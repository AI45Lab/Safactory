from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from ..binding_plan import BindingPlan
from ..http_client import HttpServiceClient
from ..types import ClusterRegistry, EnvClusterBinding, RayClusterInfo

from .base import ClusterBackend
from .rayjob import RayJobManager

log = logging.getLogger("ray_clusters")

# If you don't provide a default entrypoint, we will try per-env entrypoints only.
DEFAULT_ENTRYPOINT = "python env/app.py"


def _normalize_head_env(raw: Any) -> Dict[str, str]:
    """Normalize HeadConfig.headEnv from YAML into Dict[str, str]."""
    if raw is None:
        return {}

    if isinstance(raw, dict):
        out: Dict[str, str] = {}
        for k, v in raw.items():
            if v is None:
                continue
            out[str(k)] = str(v)
        return out

    # Compatibility: allow list style [{name: "...", value: "..."}]
    if isinstance(raw, list):
        out: Dict[str, str] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            value = item.get("value")
            if name is None or value is None:
                continue
            out[str(name)] = str(value)
        return out

    raise TypeError(f"Unsupported headEnv config type: {type(raw)!r}")


def _normalize_optional_bool(raw: Any, *, field_name: str) -> bool:
    """Normalize YAML value into bool with clear error on invalid strings."""
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        v = raw.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
    raise TypeError(f"Unsupported {field_name} config value: {raw!r}")


def build_rayjob_config(cluster_cfg: Dict[str, Any], env_name: str) -> Tuple[Optional[Any], List[Any]]:
    """
    Build rayjob_sdk.HeadConfig and a list of rayjob_sdk.Volume objects.

    Returns:
        Tuple(HeadConfig, List[Volume])
    """
    env_types = dict(cluster_cfg.get("env_types", {}) or {})
    env_cfg = dict(env_types.get(str(env_name), {}) or {})

    # Expected shape:
    #   cluster.env_types.<env>.resources.head:
    #     {
    #       cpu, gpu|nvidia.com/gpu, memory,
    #       privileged, headEnv, ...
    #     }
    head_res = dict(((env_cfg.get("resources") or {}).get("head") or {}) or {})
    raw_volumes = env_cfg.get("volumes")

    if not head_res and not raw_volumes:
        return None, []

    from rayjob_sdk import HeadConfig, Volume

    sdk_volumes = []
    if raw_volumes and isinstance(raw_volumes, list):
        for vol_data in raw_volumes:
            if isinstance(vol_data, dict):
                sdk_volumes.append(Volume(**vol_data))

    kwargs: Dict[str, Any] = {}
    resources: Dict[str, str] = {}

    # Preferred explicit nested style: resources.head.resources: { ... }
    nested_resources = head_res.get("resources")
    if isinstance(nested_resources, dict):
        for k, v in nested_resources.items():
            if v is not None:
                resources[str(k)] = str(v)

    # Legacy shorthand style: resources.head.cpu / memory / gpu
    if head_res.get("cpu") is not None:
        resources["cpu"] = str(head_res.get("cpu"))
    if head_res.get("memory") is not None:
        resources["memory"] = str(head_res.get("memory"))

    gpu = head_res.get("nvidia.com/gpu")
    if gpu is None:
        gpu = head_res.get("gpu")
    if gpu is not None:
        resources["nvidia.com/gpu"] = str(gpu)

    # Pass through all additional non-HeadConfig fields as resource keys
    # so config values under `resources.head` are effective.
    reserved = {"cpu", "memory", "gpu", "nvidia.com/gpu", "privileged", "headEnv", "head_env", "resources"}
    for k, v in head_res.items():
        key = str(k)
        if key in reserved or v is None:
            continue
        resources[key] = str(v)

    if resources:
        kwargs["resources"] = resources

    if "privileged" in head_res and head_res.get("privileged") is not None:
        kwargs["privileged"] = _normalize_optional_bool(head_res.get("privileged"), field_name="privileged")

    # Support both `headEnv` and `head_env` in config.
    if "headEnv" in head_res:
        kwargs["headEnv"] = _normalize_head_env(head_res.get("headEnv"))
    elif "head_env" in head_res:
        kwargs["headEnv"] = _normalize_head_env(head_res.get("head_env"))

    if raw_volumes:
        kwargs["volumes"] = raw_volumes

    head_config = None
    if kwargs:
        try:
            head_config = HeadConfig(**kwargs)
        except TypeError:
            kwargs.pop("volumes", None)
            if kwargs:
                head_config = HeadConfig(**kwargs)

    return head_config, sdk_volumes



def _normalize_entrypoints(raw: Any) -> Dict[str, str]:
    if raw is None:
        return {}

    if isinstance(raw, str):
        ep = raw.strip()
        return {"*": ep} if ep else {}

    if isinstance(raw, dict):
        out: Dict[str, str] = {}
        for k, v in raw.items():
            env = str(k).strip()
            ep = str(v).strip()
            if env and ep:
                out[env] = ep
        return out

    if isinstance(raw, list):
        out: Dict[str, str] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            env = str(item.get("env", "")).strip()
            ep = str(item.get("entrypoint", "")).strip()
            if env and ep:
                out[env] = ep
        return out

    raise TypeError(f"Unsupported entrypoint config type: {type(raw)!r}")


def _jobname_hint(env_name: str, idx: int) -> str:
    """Job name hint: prefix before '_' + index.

    Example:
      android_gym, idx=1 -> android-1
    """
    base = (str(env_name).split("_", 1)[0] or str(env_name)).strip()
    if not base:
        base = "rayjob"
    return f"{base}-{int(idx)}"


class RemoteRayJobBackend(ClusterBackend):
    """Remote backend that manages Ray clusters via RayJobManager (rayjob_sdk)."""

    def __init__(
        self,
        *,
        rayjob_cfg: Dict[str, Any],
        cluster_cfg: Dict[str, Any],
        http: HttpServiceClient,
        http_port: int,
    ) -> None:
        self._http = http
        self._http_port = int(http_port)

        required = ["domain", "tenant", "access_key", "secret_key"]
        missing = [k for k in required if not str(rayjob_cfg.get(k, "")).strip()]
        if missing:
            raise RuntimeError(f"Remote mode requires rayjob config keys: {missing}")

        self._rayjob_project: str = str(rayjob_cfg.get("project", "default")).strip() or "default"

        self._rayjob_manager = RayJobManager(
            domain=str(rayjob_cfg["domain"]),
            tenant=str(rayjob_cfg["tenant"]),
            access_key=str(rayjob_cfg["access_key"]),
            secret_key=str(rayjob_cfg["secret_key"]),
            token=rayjob_cfg.get("token"),
            verify=bool(rayjob_cfg.get("verify", False)),
        )

        # Keep raw cluster config for per-env env_types lookup
        self._cluster_cfg: Dict[str, Any] = dict(cluster_cfg or {})
        self._env_types: Dict[str, Any] = dict(self._cluster_cfg.get("env_types", {}) or {})

        self._quotagroup: str = str(self._cluster_cfg.get("quotagroup", "")).strip()
        # New config.yaml places description under rayjob; keep compatibility with older cluster.description.
        self._description: str = str(
            rayjob_cfg.get("description", self._cluster_cfg.get("description", "RL env Ray cluster"))
        ).strip()

        self._entrypoints: Dict[str, str] = _normalize_entrypoints(self._cluster_cfg.get("entrypoint"))
        self._default_entrypoint: str = (
            str(self._cluster_cfg.get("default_entrypoint", DEFAULT_ENTRYPOINT)).strip() or DEFAULT_ENTRYPOINT
        )

        self._poll_interval_s: float = float(self._cluster_cfg.get("head_ip_poll_interval_s", 5.0))
        self._poll_timeout_s: float = float(self._cluster_cfg.get("head_ip_poll_timeout_s", 600.0))

        # NOTE: the dict key is a *cluster id* (not necessarily image).
        # We use "{env_name}#{idx}" so ActorPool can schedule by env and pick the least-loaded job.
        self._clusters: Dict[str, RayClusterInfo] = {}

    async def start(self, plan: BindingPlan) -> ClusterRegistry:
        if not plan.env_to_image:
            return ClusterRegistry(clusters_by_id={}, env_bindings={})

        # required counts per env (computed by manager using batch_size/limit)
        required_counts = dict(getattr(plan, "env_job_counts", None) or {})
        if not required_counts:
            required_counts = {env: 1 for env in plan.env_to_image.keys()}

        # Over-provision by +30%: ceil(required * 1.3)
        # integer ceil: ceil(req * 13 / 10) = (req*13 + 9)//10
        create_counts: Dict[str, int] = {}
        for env_name, req in required_counts.items():
            req_i = max(1, int(req or 1))
            required_counts[env_name] = req_i
            # create_counts[env_name] = max(req_i, (req_i * 13 + 9) // 10)
            create_counts[env_name] = req_i

        # 1) create RayJobs (over-provisioned)
        tasks: List[asyncio.Task] = []
        for env_name, image in plan.env_to_image.items():
            image = (image or "").strip()
            if not image:
                continue

            n_create = max(1, int(create_counts.get(env_name, 1) or 1))
            for idx in range(1, n_create + 1):
                tasks.append(
                    asyncio.create_task(
                        self._ensure_cluster_for_env_job(env_name=env_name, idx=idx, image=image)
                    )
                )

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            errors = [r for r in results if isinstance(r, Exception)]
            if errors:
                # 注意：多起的那 30% 里有失败不一定是致命的，只要最终能满足 required 就继续
                for e in errors[:3]:
                    log.warning("rayjob create failed (ignored if capacity enough): %s", e)

        # sanity: ensure created >= required per env
        created_counts: Dict[str, int] = {env: 0 for env in required_counts.keys()}
        for cid in list(self._clusters.keys()):
            env = str(cid).split("#", 1)[0]
            if env in created_counts:
                created_counts[env] += 1

        lacking = {env: (created_counts.get(env, 0), req)
                   for env, req in required_counts.items()
                   if created_counts.get(env, 0) < req}
        if lacking:
            raise RuntimeError(f"Not enough RayJobs created to satisfy required counts: {lacking}")

        # 2) wait until required number of clusters per env have head_ip
        await self._wait_for_head_ips(required_counts=required_counts)

        # 3) select KEEP set: for each env keep exactly `required` clusters (prefer READY ones)
        def _cid_idx(cid: str) -> int:
            try:
                return int(str(cid).split("#", 1)[1])
            except Exception:
                return 10 ** 9

        keep_ids: set[str] = set()
        keep_infos: List[RayClusterInfo] = []
        keep_first_by_env: Dict[str, RayClusterInfo] = {}

        for env_name, req in required_counts.items():
            prefix = f"{env_name}#"
            ready = [(cid, info) for cid, info in self._clusters.items()
                     if str(cid).startswith(prefix) and getattr(info, "head_ip", "")]
            ready.sort(key=lambda x: _cid_idx(x[0]))

            if len(ready) < int(req):
                # 理论上不应发生，因为 _wait_for_required_head_ips 已满足 required
                raise RuntimeError(
                    f"Required head_ip satisfied but ready clusters still < required for env='{env_name}'. "
                    f"ready={len(ready)}, required={req}"
                )

            chosen = ready[: int(req)]
            for cid, info in chosen:
                keep_ids.add(str(cid))
                keep_infos.append(info)

            # record one binding per env (stable)
            keep_first_by_env[env_name] = chosen[0][1]

        # 4) delete the rest (ALL others, even if they already have head_ip)
        delete_ids = [cid for cid in list(self._clusters.keys()) if str(cid) not in keep_ids]

        cleanup_task = asyncio.create_task(self._stop_and_delete_clusters(delete_ids))
        http_task = asyncio.create_task(self._wait_for_head_http_services(cluster_infos=keep_infos))

        # wait both
        await asyncio.gather(http_task, cleanup_task)

        deleted_ids = cleanup_task.result() if cleanup_task.done() else []

        # remove successfully deleted from internal state (optional, avoids later noisy cleanup)
        for cid in deleted_ids:
            self._clusters.pop(cid, None)

        # 5) build registry ONLY with kept clusters (so ActorPool won't schedule onto extras)
        kept_clusters: Dict[str, RayClusterInfo] = {cid: self._clusters[cid] for cid in keep_ids if
                                                    cid in self._clusters}

        env_bindings: Dict[str, EnvClusterBinding] = {}
        for env_name, image in plan.env_to_image.items():
            info = keep_first_by_env.get(env_name)
            if not info:
                # fallback: any kept cluster for this env
                prefix = f"{env_name}#"
                for cid, v in kept_clusters.items():
                    if str(cid).startswith(prefix):
                        info = v
                        break
            if not info:
                continue

            env_bindings[env_name] = EnvClusterBinding(
                env_name=env_name,
                image=image,
                project=info.project,
                job_name=info.job_name,
                head_ip=info.head_ip,
            )

        return ClusterRegistry(clusters_by_id=kept_clusters, env_bindings=env_bindings)

    async def close(self) -> None:
        # Snapshot all cluster ids (including extras that might have failed deletion earlier)
        cluster_ids = list(self._clusters.keys())

        # Try cleanup with retries
        try:
            await self._stop_and_delete_clusters(cluster_ids)
        finally:
            # Always clear local state (remote side best-effort)
            self._clusters.clear()


    # ------------------------------------------------------------------ #

    async def _ensure_cluster_for_env_job(self, *, env_name: str, idx: int, image: str) -> None:
        env_name = str(env_name)
        image = (image or "").strip()
        if not env_name or not image:
            return

        cluster_id = f"{env_name}#{int(idx)}"
        if cluster_id in self._clusters:
            return

        env_cfg = dict(self._env_types.get(env_name, {}) or {})

        # Prefer new config: cluster.env_types.<env>.entrypoint
        entrypoint = str(
            env_cfg.get("entrypoint")
            or self._entrypoints.get(env_name)
            or self._entrypoints.get("*")
            or self._default_entrypoint
        ).strip()
        if not entrypoint:
            raise RuntimeError(
                f"No entrypoint configured for env='{env_name}'. "
                f"Provide cluster.env_types['{env_name}'].entrypoint (preferred) "
                "or cluster_cfg['entrypoint']/default_entrypoint (legacy)."
            )

        quotagroup = str(env_cfg.get("quotagroup") or self._quotagroup).strip()
        head_config ,volumes= build_rayjob_config(self._cluster_cfg, env_name)
        name_hint = _jobname_hint(env_name, idx)

        def _create_job_sync() -> str:
            return str(
                self._rayjob_manager.create(
                    project=self._rayjob_project,
                    name=name_hint,
                    image=image,
                    entrypoint=str(entrypoint),
                    quotagroup=str(quotagroup),
                    volumes=volumes,
                    description=self._description or f"Env cluster for env={env_name}",
                    head_config=head_config,
                )
            )

        max_attempts = 3
        last_err: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                job_name = await asyncio.to_thread(_create_job_sync)
                self._clusters[cluster_id] = RayClusterInfo(
                    image=image,
                    project=self._rayjob_project,
                    job_name=job_name,
                    head_ip="",
                )
                log.info(
                    "RayJob created: env='%s', idx=%s, image='%s', job_name='%s'",
                    env_name, idx, image, job_name
                )
                return
            except Exception as e:
                last_err = e
                sleep_s = min(5.0, 0.5 * (2 ** (attempt - 1)))
                log.warning(
                    "RayJob create failed (attempt %d/%d) env='%s', idx=%s, image='%s': %s. Retry in %.1fs",
                    attempt, max_attempts, env_name, idx, image, e, sleep_s
                )
                if attempt < max_attempts:
                    await asyncio.sleep(sleep_s)

        raise RuntimeError(f"RayJob create failed for env='{env_name}', idx={idx}, image='{image}': {last_err}")


    async def _wait_for_head_ips(self, required_counts: Optional[Dict[str, int]] = None) -> None:
        """Poll head IPs.

        - If required_counts is None: wait until ALL clusters have head_ip.
        - If required_counts is provided: wait until for each env we have >= required head_ip.

        required_counts key: env_name
        env_name is inferred from cluster_id format: "{env_name}#{idx}"
        """
        if not self._clusters:
            return

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._poll_timeout_s
        attempt = 0

        def _env_of_cluster_id(cid: str) -> str:
            return str(cid).split("#", 1)[0]

        while True:
            # ---------------------------
            # Check if satisfied + decide which clusters to poll next
            # ---------------------------
            if required_counts is None:
                # Wait ALL clusters
                missing_cids = [cid for cid, info in self._clusters.items() if not getattr(info, "head_ip", "")]
                if not missing_cids:
                    log.info("all clusters have head_ip")
                    return

                short_envs = None
                ready_counts = None
            else:
                # Wait per-env required counts
                # Normalize required envs: ignore <=0
                req_envs = {str(env): int(req) for env, req in (required_counts or {}).items() if int(req or 0) > 0}

                ready_counts: Dict[str, int] = {env: 0 for env in req_envs.keys()}
                for cid, info in self._clusters.items():
                    env = _env_of_cluster_id(cid)
                    if env in ready_counts and getattr(info, "head_ip", ""):
                        ready_counts[env] += 1

                short_envs = [env for env, req in req_envs.items() if ready_counts.get(env, 0) < req]
                if not short_envs:
                    log.info("required head_ip satisfied: %s", ready_counts)
                    return

                # Only poll clusters belonging to envs that are still short
                missing_cids = []
                short_env_set = set(short_envs)
                for cid, info in self._clusters.items():
                    if getattr(info, "head_ip", ""):
                        continue
                    env = _env_of_cluster_id(cid)
                    if env in short_env_set:
                        missing_cids.append(cid)

                if not missing_cids:
                    raise RuntimeError(
                        f"Still short on head_ip but no remaining clusters to poll. "
                        f"required={req_envs}, ready={ready_counts}"
                    )

            attempt += 1
            if required_counts is None:
                log.info("head_ip poll attempt=%d, missing=%s", attempt, missing_cids)
            else:
                log.info(
                    "head_ip poll attempt=%d, short_envs=%s, missing=%s, ready=%s",
                    attempt, short_envs, missing_cids, ready_counts
                )

            # ---------------------------
            # Poll head IPs for selected clusters
            # ---------------------------
            cid_list: List[str] = []
            tasks: List[Any] = []
            for cid in missing_cids:
                info = self._clusters.get(cid)
                if not info:
                    continue
                job_name = str(getattr(info, "job_name", "") or "").strip()
                if not job_name:
                    continue
                cid_list.append(cid)
                tasks.append(asyncio.to_thread(self._rayjob_manager.get_head_ip, self._rayjob_project, job_name))

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for cid, res in zip(cid_list, results):
                    if isinstance(res, Exception):
                        log.warning("get_head_ip failed for cluster='%s': %s", cid, res)
                        continue
                    ip = (res or "").strip()
                    if ip:
                        self._clusters[cid].head_ip = ip
                        log.info("head_ip resolved: cluster='%s' -> %s", cid, ip)

            # ---------------------------
            # Timeout check + sleep
            # ---------------------------
            if loop.time() >= deadline:
                if required_counts is None:
                    still_missing = [cid for cid, info in self._clusters.items() if not getattr(info, "head_ip", "")]
                    raise RuntimeError(f"Timeout waiting for head IPs: {still_missing}")
                else:
                    raise RuntimeError(f"Timeout waiting for required head IPs. required={required_counts}")

            await asyncio.sleep(self._poll_interval_s)


    async def _wait_for_head_http_services(self, cluster_infos: Optional[List[RayClusterInfo]] = None) -> None:
        if cluster_infos is None:
            cluster_infos = [info for info in self._clusters.values() if getattr(info, "head_ip", "")]
        else:
            cluster_infos = [info for info in cluster_infos if getattr(info, "head_ip", "")]

        if not cluster_infos:
            raise RuntimeError("No head_ip resolved; cannot check HTTP readiness")

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._poll_timeout_s
        attempt = 0

        infos = list(cluster_infos)  # stable snapshot

        while True:
            tasks = [self._http.check_envs_ready(info.head_ip, self._http_port) for info in infos]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            not_ready: List[RayClusterInfo] = []
            for info, res in zip(infos, results):
                if isinstance(res, Exception) or not res:
                    not_ready.append(info)

            if not not_ready:
                log.info("all head HTTP services are ready (snapshot)")
                return

            attempt += 1
            if loop.time() >= deadline:
                remaining = ", ".join(f"{getattr(i, 'job_name', '')}@{getattr(i, 'head_ip', '')}" for i in not_ready)
                raise RuntimeError(f"Timeout waiting for head HTTP services: {remaining}")

            log.info(
                "head HTTP not ready (attempt %d), retry in %.1fs. not_ready=%s",
                attempt, self._poll_interval_s, [getattr(i, 'job_name', '') for i in not_ready]
            )
            await asyncio.sleep(self._poll_interval_s)


    async def _stop_and_delete_clusters(self, cluster_ids: List[str]) -> List[str]:
        """Stop then delete clusters. Best-effort.
        Returns:
            cluster_ids that were successfully deleted.
        """
        if not cluster_ids:
            return []

        jobs: List[Tuple[str, str, str]] = []  # (cid, project, job_name)
        for cid in cluster_ids:
            info = self._clusters.get(cid)
            if not info:
                continue
            job_name = str(getattr(info, "job_name", "") or "").strip()
            if not job_name:
                continue
            project = str(getattr(info, "project", "") or self._rayjob_project).strip() or self._rayjob_project
            jobs.append((cid, project, job_name))

        if not jobs:
            return []

        async def _cleanup_one(cid: str, project: str, job_name: str) -> Optional[str]:
            try:
                await asyncio.to_thread(self._rayjob_manager.stop, project, job_name)
            except Exception as e:
                log.warning("stop rayjob failed (ignored): cluster=%s, job=%s, err=%s", cid, job_name, e)

            try:
                await asyncio.to_thread(self._rayjob_manager.delete, project, job_name)
                log.info("deleted extra rayjob: cluster=%s, job=%s", cid, job_name)
                return cid
            except Exception as e:
                log.warning("delete rayjob failed (ignored): cluster=%s, job=%s, err=%s", cid, job_name, e)
                return None

        tasks = [asyncio.create_task(_cleanup_one(cid, project, job_name)) for cid, project, job_name in jobs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        deleted: List[str] = []
        for r in results:
            if isinstance(r, str) and r:
                deleted.append(r)
        return deleted
