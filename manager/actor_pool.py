from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from env.env_factory import normalize_create_kwargs, resolve_env_class

from .http_client import HttpServiceClient
from .types import ActorKey, ActorRoute, ClusterRegistry, EnvClusterBinding, PoolEntry

log = logging.getLogger("manager.actor_pool")


class ActorPool:
    """
    Manages prewarm + close/refill via either HTTP or local in-process transport.
    """

    def __init__(
        self,
        *,
        repo,
        http: HttpServiceClient,
        pool_size: int,
        http_port: int,
        http_concurrency: int,
        startup_concurrency: Optional[int] = None,
        base_image: str,
        default_seed: int = 123,
        env_limits: Optional[Dict[str, int]] = None,
        transport_mode: str = "http",
    ) -> None:
        self._repo = repo
        self._http = http

        self._pool_size = int(pool_size)
        self._http_port = int(http_port)
        self._http_concurrency = int(http_concurrency)
        if startup_concurrency is None:
            startup_concurrency = min(self._http_concurrency, 16)
        self._startup_concurrency = max(1, int(startup_concurrency))
        self._base_image = (base_image or "").strip()
        self._default_seed = int(default_seed)
        self._transport_mode = str(transport_mode or "http").strip().lower() or "http"
        self._env_limits: Dict[str, int] = {}
        for k, v in (env_limits or {}).items():
            try:
                self._env_limits[str(k)] = int(v)
            except Exception:
                continue

        self._lock = asyncio.Lock()
        self._fill_sem = asyncio.Semaphore(self._startup_concurrency)
        self._pool: Dict[ActorKey, PoolEntry] = {}
        self._actor_routes: Dict[ActorKey, ActorRoute] = {}
        self._job_load: Dict[Tuple[str, str], int] = {}
        self._inproc_executor: Optional[ThreadPoolExecutor] = None
        if self._transport_mode == "inproc":
            self._inproc_executor = ThreadPoolExecutor(
                max_workers=max(1, self._startup_concurrency),
                thread_name_prefix="env-inproc-bg",
            )

    def _repo_fetch_batch_size(self, requested: int = 1) -> int:
        return max(1, self._pool_size, int(requested))

    async def _run_inproc_background_call(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        if self._inproc_executor is None:
            raise RuntimeError("inproc background executor is unavailable")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._inproc_executor, partial(fn, *args, **kwargs))

    async def shutdown(self) -> None:
        executor = self._inproc_executor
        if executor is None:
            return
        self._inproc_executor = None
        await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=False)

    async def reset(self) -> None:
        entries: List[PoolEntry]
        async with self._lock:
            entries = list(self._pool.values())
            self._pool.clear()
            self._actor_routes.clear()
            self._repo.reset_cursor()
            self._job_load.clear()

        inproc_entries = [
            entry
            for entry in entries
            if str(getattr(entry, "transport", "http") or "http") == "inproc"
            and getattr(entry, "local_env", None) is not None
        ]
        if not inproc_entries:
            return

        async def _close_entry(entry: PoolEntry) -> None:
            await self._close_actor_inproc(
                env=str(entry.env_name),
                env_id=str(entry.env_id),
                old_entry=entry,
            )

        await asyncio.gather(*[_close_entry(entry) for entry in inproc_entries], return_exceptions=True)

    async def list_actors(self) -> List[PoolEntry]:
        async with self._lock:
            return list(self._pool.values())

    def get_actor_route(self, env: str, env_id: str, fallback: Optional[EnvClusterBinding]) -> Optional[ActorRoute]:
        key = (str(env), str(env_id))
        entry = self._pool.get(key)
        if entry is not None and str(getattr(entry, "transport", "http") or "http") == "inproc":
            return None

        route = self._actor_routes.get(key)
        if route:
            return route
        if fallback and fallback.head_ip:
            return (fallback.head_ip, self._http_port)
        return None

    async def prewarm(self, registry: ClusterRegistry, rows: Optional[List[Dict[str, Any]]] = None) -> None:
        if self._pool_size <= 0:
            log.info("pool_size <= 0, skip prewarm")
            return

        if rows is None:
            rows = await self._repo.reserve_rows(
                self._pool_size,
                fetch_batch_size=self._repo_fetch_batch_size(self._pool_size),
            )

        if not rows:
            log.info("no active rows, skip prewarm")
            return

        log.info(
            "prewarm start: target_pool_size=%d initial_rows=%d startup_concurrency=%d http_concurrency=%d transport=%s",
            self._pool_size,
            len(rows),
            self._startup_concurrency,
            self._http_concurrency,
            self._transport_mode,
        )

        tasks = [asyncio.create_task(self._robust_fill_slot(registry, self._fill_sem, initial_row=row)) for row in rows]
        await asyncio.gather(*tasks, return_exceptions=True)

        await self.ensure_capacity(registry)

    async def ensure_capacity(self, registry: ClusterRegistry) -> None:
        """
        Continuously fill the pool until it reaches pool_size or DB is empty.
        """
        while True:
            async with self._lock:
                deficit = max(0, self._pool_size - len(self._pool))
            if deficit <= 0:
                return

            rows = await self._repo.reserve_rows(
                deficit,
                fetch_batch_size=self._repo_fetch_batch_size(deficit),
            )
            if not rows:
                log.info("[manager] ensure_capacity: no DB rows currently available to fill pool")
                return

            tasks = [asyncio.create_task(self._robust_fill_slot(registry, self._fill_sem, initial_row=row)) for row in rows]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def close_and_refill(self, *, env: str, env_id: str, registry: ClusterRegistry) -> Optional[PoolEntry]:
        """
        Close the specified actor and immediately try to create a new one to replace it.
        """
        env = str(env)
        env_id = str(env_id)
        key = (env, env_id)

        async with self._lock:
            route = self._actor_routes.get(key)
            binding = registry.env_bindings.get(env)
            old_entry = self._pool.pop(key, None)
            self._actor_routes.pop(key, None)
            if old_entry is not None:
                self._release_job_load_for_entry_locked(old_entry)

        next_row = await self._repo.reserve_one(fetch_batch_size=self._repo_fetch_batch_size())

        if self._transport_mode == "inproc":
            await self._close_actor_inproc(env=env, env_id=env_id, old_entry=old_entry)
        else:
            await self._close_actor_http(env=env, env_id=env_id, route=route, binding=binding)

        if not next_row:
            log.info("close_and_refill: no DB row currently available to refill pool")
            return None

        return await self._robust_fill_slot(registry, sem=self._fill_sem, initial_row=next_row)

    async def _robust_fill_slot(
        self,
        registry: ClusterRegistry,
        sem: Optional[asyncio.Semaphore],
        initial_row: Optional[Dict[str, Any]],
    ) -> Optional[PoolEntry]:
        """
        Attempts to fill one pool slot.
        If initial_row fails, fetch the next row and retry until success or DB runs out.
        """
        current_row = initial_row

        while True:
            if current_row is None:
                current_row = await self._repo.reserve_one(fetch_batch_size=self._repo_fetch_batch_size())
                if current_row is None:
                    log.info("robust_fill_slot: no DB row currently available, stopping slot fill.")
                    return None

            env_key = f"{current_row.get('env_name')}/{current_row.get('env_id')}"
            try:
                if sem:
                    async with sem:
                        created = await self._attempt_create_actor(current_row, registry)
                else:
                    created = await self._attempt_create_actor(current_row, registry)

                if created is not None:
                    log.info("Successfully created actor for %s", env_key)
                    return created

                log.error("Failed to create actor for %s after retries. Skipping row.", env_key)
                current_row = None
                continue

            except Exception:
                log.error(
                    "Unexpected error while creating actor for %s. Skipping row.",
                    env_key,
                    exc_info=True,
                )
                current_row = None

    async def _attempt_create_actor(
        self,
        row: Dict[str, Any],
        registry: ClusterRegistry,
    ) -> Optional[PoolEntry]:
        if self._transport_mode == "inproc":
            return await self._attempt_create_actor_inproc(row, registry)
        return await self._attempt_create_actor_http(row, registry)

    def _resolve_image(self, row: Dict[str, Any], registry: ClusterRegistry, env_name: str) -> str:
        image = (row.get("image") or "").strip()
        if image:
            return image

        binding = registry.env_bindings.get(env_name)
        if binding and binding.image:
            return str(binding.image)
        return self._base_image

    def _build_pool_entry(
        self,
        *,
        env_name: str,
        env_id: str,
        row: Dict[str, Any],
        image: str,
        job_name: str,
        head_ip: str,
        transport: str,
        local_env: Any = None,
    ) -> PoolEntry:
        return PoolEntry(
            env_name=env_name,
            env_id=env_id,
            row_id=row.get("id"),
            image=image,
            job_name=job_name,
            head_ip=head_ip,
            group_id=str(row.get("group_id") or ""),
            status="ready",
            transport=transport,
            local_env=local_env,
        )

    def _release_job_load_locked(self, *, env_name: str, job_name: str) -> None:
        env_name = str(env_name or "").strip()
        job_name = str(job_name or "").strip()
        if not env_name or not job_name:
            return

        key = (env_name, job_name)
        cur = int(self._job_load.get(key, 0) or 0)
        if cur <= 1:
            self._job_load.pop(key, None)
        else:
            self._job_load[key] = cur - 1

    def _release_job_load_for_entry_locked(self, entry: Optional[PoolEntry]) -> None:
        if entry is None:
            return
        self._release_job_load_locked(
            env_name=str(getattr(entry, "env_name", "") or ""),
            job_name=str(getattr(entry, "job_name", "") or ""),
        )

    async def _attempt_create_actor_http(
        self,
        row: Dict[str, Any],
        registry: ClusterRegistry,
    ) -> Optional[PoolEntry]:
        env_name = str(row.get("env_name", "")).strip()
        env_id = str(row.get("env_id", "")).strip()
        if not env_name or not env_id:
            log.error("Invalid DB row (missing env_name/env_id): %s", row)
            return None

        image = self._resolve_image(row, registry, env_name)
        if not image:
            log.error("Cannot resolve image for env='%s' id='%s'", env_name, env_id)
            return None

        async with self._lock:
            try:
                cluster = self._choose_cluster_and_reserve_locked(env_name=env_name, image=image, registry=registry)
                reserved_key = (env_name, str(cluster.job_name))
            except Exception:
                log.error("Reservation failed for %s/%s", env_name, env_id, exc_info=True)
                return None

        url = f"http://{cluster.head_ip}:{self._http_port}/{env_name}/{env_id}/reset"
        delete_url = f"http://{cluster.head_ip}:{self._http_port}/{env_name}/{env_id}"
        payload = {
            "env_param": row.get("env_params"),
            "seed": row.get("seed", self._default_seed),
        }

        async def cleanup_remote_actor(stage: str) -> None:
            try:
                async with await self._http.delete(delete_url) as cleanup_resp:
                    log.info(
                        "Cleanup response for %s/%s during %s: status=%s",
                        env_name,
                        env_id,
                        stage,
                        cleanup_resp.status,
                    )
            except Exception:
                log.warning(
                    "Cleanup failed for %s/%s during %s (ignoring)",
                    env_name,
                    env_id,
                    stage,
                    exc_info=True,
                )

        max_reset_attempts = 1
        cleanup_attempted_for_failure = False

        for attempt in range(1, max_reset_attempts + 1):
            cleanup_attempted_for_failure = False
            try:
                async with await self._http.post(url, json=payload) as resp:
                    if resp.status in (500, 502, 503, 504):
                        raise RuntimeError(f"Server error status={resp.status}")

                    resp.raise_for_status()

                    key: ActorKey = (env_name, env_id)
                    entry = self._build_pool_entry(
                        env_name=env_name,
                        env_id=env_id,
                        row=row,
                        image=image,
                        job_name=str(cluster.job_name),
                        head_ip=str(cluster.head_ip),
                        transport="http",
                    )
                    async with self._lock:
                        self._pool[key] = entry
                        self._actor_routes[key] = (cluster.head_ip, self._http_port)
                    return entry

            except (asyncio.TimeoutError, aiohttp.ClientError, RuntimeError):
                log.warning(
                    "Reset attempt %d/%d failed for %s/%s",
                    attempt,
                    max_reset_attempts,
                    env_name,
                    env_id,
                    exc_info=True,
                )
                await cleanup_remote_actor(f"attempt_{attempt}")
                cleanup_attempted_for_failure = True
                if attempt < max_reset_attempts:
                    await asyncio.sleep(1.0)
            except Exception:
                log.error(
                    "Non-recoverable reset failure for %s/%s on attempt %d/%d",
                    env_name,
                    env_id,
                    attempt,
                    max_reset_attempts,
                    exc_info=True,
                )
                break

        if not cleanup_attempted_for_failure:
            log.warning("Reset failed for %s/%s. Sending CLEANUP (DELETE) to %s...", env_name, env_id, cluster.head_ip)
            await cleanup_remote_actor("final_failure")
        else:
            log.warning(
                "Reset failed %d times for %s/%s. Cleanup was attempted after each failed attempt.",
                max_reset_attempts,
                env_name,
                env_id,
            )

        async with self._lock:
            self._release_job_load_locked(env_name=reserved_key[0], job_name=reserved_key[1])

        return None

    def _create_and_reset_local_env_sync(
        self,
        env_name: str,
        create_kwargs: Dict[str, Any],
        seed: Any,
    ) -> Any:
        env_cls = resolve_env_class(env_name)
        env_obj = env_cls(**create_kwargs)
        try:
            env_obj.reset(seed=seed)
        except Exception:
            close_fn = getattr(env_obj, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    log.warning("inproc env close after reset failure failed: env=%s", env_name, exc_info=True)
            raise
        return env_obj

    async def _attempt_create_actor_inproc(
        self,
        row: Dict[str, Any],
        registry: ClusterRegistry,
    ) -> Optional[PoolEntry]:
        env_name = str(row.get("env_name", "")).strip()
        env_id = str(row.get("env_id", "")).strip()
        if not env_name or not env_id:
            log.error("Invalid DB row (missing env_name/env_id): %s", row)
            return None

        image = self._resolve_image(row, registry, env_name)
        if not image:
            log.error("Cannot resolve image for env='%s' id='%s'", env_name, env_id)
            return None

        async with self._lock:
            try:
                cluster = self._choose_cluster_and_reserve_locked(env_name=env_name, image=image, registry=registry)
                reserved_key = (env_name, str(cluster.job_name))
            except Exception:
                log.error("Reservation failed for %s/%s", env_name, env_id, exc_info=True)
                return None

        try:
            create_kwargs = normalize_create_kwargs(row.get("env_params"))
        except Exception:
            log.error("Invalid env_params for %s/%s transport=inproc", env_name, env_id, exc_info=True)
            async with self._lock:
                self._release_job_load_locked(env_name=reserved_key[0], job_name=reserved_key[1])
            return None

        seed = row.get("seed", self._default_seed)
        max_reset_attempts = 3

        for attempt in range(1, max_reset_attempts + 1):
            try:
                env_obj = await self._run_inproc_background_call(
                    self._create_and_reset_local_env_sync,
                    env_name,
                    create_kwargs,
                    seed,
                )

                key: ActorKey = (env_name, env_id)
                entry = self._build_pool_entry(
                    env_name=env_name,
                    env_id=env_id,
                    row=row,
                    image=image,
                    job_name=str(cluster.job_name),
                    head_ip=str(cluster.head_ip),
                    transport="inproc",
                    local_env=env_obj,
                )
                async with self._lock:
                    self._pool[key] = entry
                return entry

            except Exception:
                log.warning(
                    "Inproc reset attempt %d/%d failed for %s/%s",
                    attempt,
                    max_reset_attempts,
                    env_name,
                    env_id,
                    exc_info=True,
                )
                if attempt < max_reset_attempts:
                    await asyncio.sleep(1.0)

        async with self._lock:
            self._release_job_load_locked(env_name=reserved_key[0], job_name=reserved_key[1])

        return None

    async def _close_actor_http(
        self,
        *,
        env: str,
        env_id: str,
        route: Optional[ActorRoute],
        binding: Optional[EnvClusterBinding],
    ) -> None:
        host = ""
        if route:
            host = route[0]
        elif binding and binding.head_ip:
            host = binding.head_ip

        if not host:
            log.warning("close skipped: no route/binding for env='%s' id='%s'", env, env_id)
            return

        delete_url = f"http://{host}:{self._http_port}/{env}/{env_id}"
        legacy_post_url = f"http://{host}:{self._http_port}/{env}/{env_id}/close"

        try:
            async with await self._http.delete(delete_url) as resp:
                status = resp.status
                if status == 404:
                    async with await self._http.post(legacy_post_url) as resp2:
                        if resp2.status >= 400:
                            log.error("close failed (fallback): %s status=%s", legacy_post_url, resp2.status)
                elif status >= 400:
                    log.error("close failed: %s status=%s", delete_url, status)

        except Exception:
            log.error(
                "close error: env='%s', id='%s'",
                env,
                env_id,
                exc_info=True,
            )

    async def _close_actor_inproc(self, *, env: str, env_id: str, old_entry: Optional[PoolEntry]) -> None:
        local_env = None if old_entry is None else getattr(old_entry, "local_env", None)
        close_fn = None if local_env is None else getattr(local_env, "close", None)
        if not callable(close_fn):
            return

        try:
            await self._run_inproc_background_call(close_fn)
        except Exception:
            log.warning(
                "close error: env='%s', id='%s' transport=inproc",
                env,
                env_id,
                exc_info=True,
            )

    def _choose_cluster_and_reserve_locked(self, *, env_name: str, image: str, registry: ClusterRegistry):
        """
        Pick the best cluster for this env and reserve one slot.
        MUST be called under self._lock.
        """
        prefix = f"{env_name}#"
        candidates = []

        clusters = registry.clusters_by_id or {}
        for cid, info in clusters.items():
            if str(cid).startswith(prefix):
                if info is not None and getattr(info, "head_ip", None):
                    candidates.append(info)

        if not candidates:
            raise RuntimeError(f"No cluster/head_ip available for env='{env_name}', image='{image}'")

        lim = int(self._env_limits.get(env_name, 0) or 0)

        def load_of(info: Any) -> int:
            job_name = str(getattr(info, "job_name", "") or "")
            return int(self._job_load.get((env_name, job_name), 0) or 0)

        if lim > 0:
            below = [candidate for candidate in candidates if load_of(candidate) < lim]
            pool = below or candidates
        else:
            pool = candidates

        chosen = min(pool, key=lambda candidate: (load_of(candidate), str(getattr(candidate, "job_name", "") or "")))
        chosen_job = str(getattr(chosen, "job_name", "") or "")
        self._job_load[(env_name, chosen_job)] = int(self._job_load.get((env_name, chosen_job), 0) or 0) + 1
        return chosen
