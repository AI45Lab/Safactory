import asyncio
import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Protocol

import aiohttp

from core.data_manager.manager import DataManager, SessionContext
from core.exp.handler import EpisodeHandler, NullEpisodeHandler
from core.http.http_client import HttpClient
from core.llm import (
    BaseURLProvider,
    LLM,
    SessionSuffixBaseURLProvider,
    resolve_llm_http_settings,
)

log = logging.getLogger("interactor")


@dataclass(slots=True, frozen=True)
class ActorHandle:
    """
    A lightweight reference to an env actor exposed via HTTP or local inproc transport.
    """

    env_name: str
    env_id: str
    group_id: str
    transport: str = "http"
    base_url: Optional[str] = None
    local_env: Optional[Any] = None


class ActorPool(Protocol):
    pool_size: int

    async def start(self) -> None: ...

    async def acquire(self) -> Optional[ActorHandle]: ...

    async def done(self, actor: ActorHandle) -> None: ...

    async def aclose(self) -> None: ...


class Interactor:
    """
    Interactor drives episodes:
      get_task_prompt -> LLM -> step (repeat) until terminated/truncated or prompt empty.
    """

    def __init__(
        self,
        pool: ActorPool,
        base_url_provider: BaseURLProvider,
        api_key: str,
        model: str,
        data_manager: DataManager,
        temperature: float = 0.3,
        max_steps: int = 1000,
        message_cut: int = 3,
        env_http_timeout_s: float = 30.0,
        max_workers: Optional[int] = None,
        http_retries: int = 2,
        http_retry_backoff_s: float = 0.5,
        verbose: bool = True,
        log_actions_preview_chars: int = 120,
        episode_handler: Optional[EpisodeHandler] = None,
    ):
        self.pool = pool
        self.max_steps = int(max_steps)
        self.message_cut = int(message_cut)
        self.max_workers = int(max_workers) if max_workers is not None else None
        self.env_request_timeout_s = float(env_http_timeout_s)

        self.http_retries = max(0, int(http_retries))
        self.http_retry_backoff_s = float(http_retry_backoff_s)
        self.verbose = bool(verbose)

        self.log_actions_preview_chars = max(0, int(log_actions_preview_chars))
        self.episode_handler = episode_handler or NullEpisodeHandler()

        self.model = model
        self.data_manager = data_manager

        self.base_url_provider = base_url_provider
        self.api_key = api_key
        self.temperature = float(temperature)

        self._worker_count = self._derive_worker_count()
        self._session_routed_llm = isinstance(self.base_url_provider, SessionSuffixBaseURLProvider)
        self._llm_http_settings = resolve_llm_http_settings(
            worker_count=self._worker_count,
            session_routed=self._session_routed_llm,
            trust_env=True,
        )
        self._llm_timeout_s = self._llm_http_settings.request_timeout_s
        self._llm_connect_timeout_s = self._llm_http_settings.connect_timeout_s
        self._llm_sock_read_timeout_s = self._llm_http_settings.sock_read_timeout_s
        self._llm_max_connections = self._llm_http_settings.max_connections
        self._llm_keepalive_connections = self._llm_http_settings.keepalive_connections
        self._llm_max_concurrency = self._llm_http_settings.max_concurrency
        self._llm_semaphore = asyncio.Semaphore(self._llm_max_concurrency) if self._llm_max_concurrency > 0 else None
        self._llm_startup_jitter_s = self._llm_http_settings.startup_jitter_s
        self._inproc_executor = ThreadPoolExecutor(
            max_workers=self._worker_count,
            thread_name_prefix="env-inproc-fg",
        )

        self.http = HttpClient(
            timeout_s=self.env_request_timeout_s,
            max_connections=max(64, pool.pool_size * 8),
            max_keepalive_connections=max(32, pool.pool_size * 4),
            trust_env=True,
        )
        self.llm_http = HttpClient(
            timeout_s=self._llm_timeout_s,
            connect_timeout_s=self._llm_connect_timeout_s,
            sock_read_timeout_s=self._llm_sock_read_timeout_s,
            max_connections=self._llm_max_connections,
            max_keepalive_connections=self._llm_keepalive_connections,
            trust_env=self._llm_http_settings.trust_env,
            ttl_dns_cache_s=300,
        )

        log.info(
            "Interactor initialized: model=%s temp=%.3f max_steps=%d message_cut=%d http_timeout=%.1fs retries=%d "
            "workers=%d llm_timeout=%.1fs llm_connect_timeout=%.1fs llm_sock_read_timeout=%.1fs "
            "llm_max_connections=%d llm_max_concurrency=%d llm_session_routed=%s",
            model,
            float(temperature),
            self.max_steps,
            self.message_cut,
            self.env_request_timeout_s,
            self.http_retries,
            self._worker_count,
            self._llm_timeout_s,
            self._llm_connect_timeout_s,
            self._llm_sock_read_timeout_s,
            self._llm_max_connections,
            self._llm_max_concurrency,
            self._session_routed_llm,
        )

    def _derive_worker_count(self) -> int:
        worker_count = max(1, int(getattr(self.pool, "pool_size", 1) or 1))
        if self.max_workers is not None:
            worker_count = max(1, min(worker_count, int(self.max_workers)))
        return worker_count

    def _create_llm_for_session(self, session: SessionContext) -> LLM:
        base_url = self.base_url_provider.get_base_url(session)
        return LLM(
            api_key=self.api_key,
            base_url=base_url,
            model=self.model,
            temperature=self.temperature,
            http_client=self.llm_http,
            request_timeout_s=self._llm_timeout_s,
            connect_timeout_s=self._llm_connect_timeout_s,
            sock_read_timeout_s=self._llm_sock_read_timeout_s,
            trust_env=self._llm_http_settings.trust_env,
        )

    def _trim_messages(self, prompt: Any) -> List[Dict[str, Any]]:
        """
        Keep system + last N turns for better LLM cost/control.
        """
        if not isinstance(prompt, list) or not prompt:
            return []
        if self.message_cut <= 0:
            return prompt

        out: List[Dict[str, Any]] = []
        start = 0
        if isinstance(prompt[0], dict) and prompt[0].get("role") == "system":
            out.append(prompt[0])
            start = 1

        tail = prompt[start:]
        out.extend(tail[-(self.message_cut * 2 - 1) :])
        return out

    def _url(self, actor: ActorHandle, suffix: str) -> str:
        if actor.transport != "http" or not actor.base_url:
            raise RuntimeError(f"HTTP URL requested for non-HTTP actor: transport={actor.transport}")
        return f"{actor.base_url.rstrip('/')}/{actor.env_name}/{actor.env_id}/{suffix.lstrip('/')}"

    async def _llm_generate(self, llm: LLM, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self._llm_semaphore is None:
            return await llm.generate(messages=messages)

        async with self._llm_semaphore:
            return await llm.generate(messages=messages)

    async def _http_request_json(
        self,
        method: str,
        url: str,
        *,
        json_body: Optional[dict] = None,
        ctx: str = "",
    ) -> Any:
        """
        Execute HTTP requests with retry logic.
        """
        await self.http.start()

        attempts = 1 + self.http_retries
        last_err: Optional[Exception] = None

        for i in range(attempts):
            t0 = time.perf_counter()
            try:
                async with self.http.request(method, url, json=json_body) as response:
                    dt = time.perf_counter() - t0
                    status = response.status

                    if 500 <= status <= 599:
                        response.raise_for_status()

                    if status >= 400:
                        response.raise_for_status()

                    log.debug("HTTP %s %s (%s) -> %d in %.3fs", method, url, ctx, status, dt)
                    return await response.json()

            except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
                dt = time.perf_counter() - t0
                last_err = exc

                status = getattr(exc, "status", None)
                msg = getattr(exc, "message", str(exc))

                if status is not None and 500 <= int(status) <= 599:
                    log.warning(
                        "HTTP %s %s (%s) -> %s (attempt %d/%d) after %.3fs msg=%r",
                        method,
                        url,
                        ctx,
                        status,
                        i + 1,
                        attempts,
                        dt,
                        msg,
                    )
                elif status is not None:
                    log.error(
                        "HTTP %s %s (%s) -> %s after %.3fs msg=%r (not retrying)",
                        method,
                        url,
                        ctx,
                        status,
                        dt,
                        msg,
                    )
                    raise
                else:
                    log.warning(
                        "HTTP %s %s (%s) transport/timeout (attempt %d/%d) after %.3fs: %s",
                        method,
                        url,
                        ctx,
                        i + 1,
                        attempts,
                        dt,
                        exc,
                    )

            if i + 1 < attempts:
                await asyncio.sleep(self.http_retry_backoff_s * (2**i))

        assert last_err is not None
        raise last_err

    def _normalize_inproc_result(self, result: Any) -> Any:
        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json")
        if isinstance(result, memoryview):
            return json.loads(result.tobytes())
        if isinstance(result, (bytes, bytearray)):
            return json.loads(result)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return result
        raise TypeError(f"Unsupported inproc response type: {type(result).__name__}")

    async def _inproc_request_json(
        self,
        actor: ActorHandle,
        operation: str,
        *,
        action: Optional[str] = None,
        ctx: str = "",
    ) -> Any:
        if actor.transport != "inproc" or actor.local_env is None:
            raise RuntimeError(f"inproc env handle is not available for transport={actor.transport}")

        def _call() -> Any:
            if operation == "get_task_prompt":
                return actor.local_env.get_task_prompt()
            if operation == "step":
                return actor.local_env.step(str(action or ""))
            raise ValueError(f"Unsupported inproc env operation: {operation}")

        t0 = time.perf_counter()
        log.debug("INPROC %s %s -> start", operation, ctx or "-")
        try:
            if self._inproc_executor is None:
                raise RuntimeError("inproc executor is unavailable")
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(self._inproc_executor, partial(_call))
            result = await asyncio.wait_for(future, timeout=self.env_request_timeout_s)
            body = self._normalize_inproc_result(result)
            log.debug("INPROC %s %s -> ok in %.3fs", operation, ctx or "-", time.perf_counter() - t0)
            return body
        except Exception:
            log.warning("INPROC %s %s -> failed in %.3fs", operation, ctx or "-", time.perf_counter() - t0, exc_info=True)
            raise

    async def _env_get_task_prompt(self, actor: ActorHandle, *, worker_id: int, env_key: str, step_i: int) -> Any:
        ctx = f"worker={worker_id} env={env_key} step={step_i} request=get_task_prompt"
        if actor.transport == "inproc":
            return await self._inproc_request_json(actor, "get_task_prompt", ctx=ctx)
        return await self._http_request_json(
            "GET",
            self._url(actor, "get_task_prompt"),
            ctx=ctx,
        )

    async def _env_step(
        self,
        actor: ActorHandle,
        *,
        worker_id: int,
        env_key: str,
        step_i: int,
        action: str,
    ) -> Any:
        ctx = f"worker={worker_id} env={env_key} step={step_i} request=step"
        if actor.transport == "inproc":
            return await self._inproc_request_json(actor, "step", action=action, ctx=ctx)
        return await self._http_request_json(
            "POST",
            self._url(actor, "step"),
            json_body={"action": action},
            ctx=ctx,
        )

    async def _run_one_environment(self, actor: ActorHandle, worker_id: int) -> float:
        env_key = f"{actor.env_name}_{actor.env_id}"
        last_info: Optional[Dict[str, Any]] = None

        session = await self.data_manager.create_session(
            env_id=actor.env_id,
            env_name=actor.env_name,
            llm_model=self.model,
            group_id=actor.group_id,
        )

        llm = self._create_llm_for_session(session)

        try:
            for step_i in range(1, self.max_steps + 1):
                try:
                    prompt_raw = await self._env_get_task_prompt(
                        actor,
                        worker_id=worker_id,
                        env_key=env_key,
                        step_i=step_i,
                    )
                except Exception as exc:
                    log.error("worker=%d env=%s: get_task_prompt FAILED: %s. Aborting episode.", worker_id, env_key, exc)
                    raise

                prompt_raw = await self.episode_handler.handle(actor.env_name, actor.env_id, step_i, prompt_raw)

                prompt = self._trim_messages(prompt_raw)
                if not prompt:
                    log.info("worker=%d env=%s: empty prompt -> end episode", worker_id, env_key)
                    break

                try:
                    llm_result = await self._llm_generate(llm, prompt)
                except Exception as exc:
                    log.error("worker=%d env=%s: LLM generation FAILED: %s. Aborting episode.", worker_id, env_key, exc)
                    raise

                action = llm_result["content"]
                finish_reason = llm_result.get("finish_reason", "stop")
                weight_version = llm_result.get("weight_version")

                env_state = json.dumps({"weight_version": weight_version}) if weight_version is not None else None

                if self.log_actions_preview_chars > 0:
                    preview = action.replace("\n", "\\n")[: self.log_actions_preview_chars]
                    log.debug(
                        "worker=%d env=%s step=%d action_preview=%r finish_reason=%s",
                        worker_id,
                        env_key,
                        step_i,
                        preview,
                        finish_reason,
                    )

                if finish_reason == "length":
                    log.info("worker=%d env=%s step=%d: LLM output truncated, terminating.", worker_id, env_key, step_i)
                    reward = -0.1
                    terminated = True
                    truncated = True
                    is_trainable = True
                    await self.data_manager.record_step(
                        session=session,
                        step_id=step_i,
                        messages=prompt,
                        response=action,
                        step_reward=reward,
                        env_state=env_state,
                        terminated=terminated,
                        truncated=truncated,
                        is_trainable=is_trainable,
                    )
                    break

                try:
                    out = await self._env_step(
                        actor,
                        worker_id=worker_id,
                        env_key=env_key,
                        step_i=step_i,
                        action=action,
                    )
                except Exception as exc:
                    log.error(
                        "worker=%d env=%s step=%d: STEP REQUEST FAILED: %s. Aborting episode.",
                        worker_id,
                        env_key,
                        step_i,
                        exc,
                    )
                    raise

                reward = float(out.get("reward", 0.0) or 0.0)
                terminated = bool(out.get("terminated", False))
                truncated = bool(out.get("truncated", False))
                raw_info = out.get("info")
                last_info = raw_info if isinstance(raw_info, dict) else None

                if self.message_cut > 0:
                    is_trainable = True
                elif self.message_cut <= 0 and (terminated or truncated):
                    is_trainable = True
                else:
                    is_trainable = False

                await self.data_manager.record_step(
                    session=session,
                    step_id=step_i,
                    messages=prompt,
                    response=action,
                    step_reward=reward,
                    env_state=env_state,
                    terminated=terminated,
                    truncated=truncated,
                    is_trainable=is_trainable,
                )

                if terminated or truncated:
                    log.info("worker=%d env=%s done: terminated=%s truncated=%s", worker_id, env_key, terminated, truncated)
                    break

        finally:
            try:
                await self.episode_handler.on_episode_end(
                    env_name=actor.env_name,
                    env_id=actor.env_id,
                    total_reward=session.total_reward,
                    info=last_info,
                )
            except Exception:
                log.exception("worker=%d env=%s: episode_handler.on_episode_end failed", worker_id, env_key)

        return session.total_reward

    async def run_all_environments(self) -> Dict[str, float]:
        """
        Run workers until pool is exhausted.
        """
        await self.http.start()
        await self.llm_http.start()
        await self.pool.start()

        results: Dict[str, float] = {}
        lock = asyncio.Lock()

        worker_count = self._worker_count
        log.info("run start: pool_size=%d workers=%d", self.pool.pool_size, worker_count)

        async def worker(worker_id: int) -> None:
            apply_startup_jitter = self._llm_startup_jitter_s > 0.0
            while True:
                actor = await self.pool.acquire()
                if actor is None:
                    log.info("worker=%d: no more actors (pool exhausted) -> exit", worker_id)
                    return

                env_key = f"{actor.env_name}_{actor.env_id}"
                t0 = time.perf_counter()
                reward = float("nan")

                try:
                    log.info(
                        "worker=%d acquired env=%s transport=%s base_url=%s",
                        worker_id,
                        env_key,
                        actor.transport,
                        actor.base_url,
                    )

                    if apply_startup_jitter:
                        # Spread the initial worker wave so startup does not hammer the LLM at once.
                        await asyncio.sleep(random.uniform(0.0, self._llm_startup_jitter_s))
                        apply_startup_jitter = False

                    reward = await self._run_one_environment(actor, worker_id)

                    async with lock:
                        results[env_key] = reward

                except Exception as exc:
                    log.warning(
                        "worker=%d env=%s FAILED (Exception): %s. Will recycle actor and continue.",
                        worker_id,
                        env_key,
                        exc,
                    )

                finally:
                    try:
                        await self.pool.done(actor)
                    except Exception as exc:
                        log.exception("worker=%d env=%s CRITICAL ERROR in pool.done(): %s", worker_id, env_key, exc)

                dt = time.perf_counter() - t0
                log.info("worker=%d env=%s finished (reward=%s) time=%.2fs", worker_id, env_key, reward, dt)

        tasks = [asyncio.create_task(worker(i), name=f"worker-{i}") for i in range(worker_count)]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            log.warning("run cancelled: cancelling workers...")
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        except Exception:
            log.exception("run failed: cancelling workers...")
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        log.info("run finished: episodes=%d", len(results))
        return results

    async def aclose(self) -> None:
        try:
            executor = self._inproc_executor
            self._inproc_executor = None
            if executor is not None:
                await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=False)
        except Exception:
            log.exception("failed to close inproc foreground executor (ignored)")
        try:
            await self.http.close()
        except Exception:
            log.exception("failed to close http client (ignored)")
        try:
            await self.llm_http.close()
        except Exception:
            log.exception("failed to close llm http client (ignored)")
