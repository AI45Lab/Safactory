import asyncio
import os
import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import logging
import aiohttp
import threading

from core.http.http_client import HttpClient

logger = logging.getLogger(__name__)

# 从环境变量读取重试配置，支持模型权重更新期间的长时间等待
# -1 表示无限重试
DEFAULT_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "-1"))  # 默认无限重试
DEFAULT_RETRY_BACKOFF = float(os.environ.get("LLM_RETRY_BACKOFF", "2.0"))  # 初始退避 2 秒
DEFAULT_MAX_RETRY_DELAY = float(os.environ.get("LLM_MAX_RETRY_DELAY", "30.0"))  # 最大退避 30 秒
DEFAULT_RETRY_JITTER_RATIO = float(os.environ.get("LLM_RETRY_JITTER_RATIO", "0.2"))
DEFAULT_REQUEST_TIMEOUT_S = 300.0
SESSION_ROUTED_REQUEST_TIMEOUT_S = 900.0
DEFAULT_CONNECT_TIMEOUT_S = 100.0
DEFAULT_STARTUP_JITTER_S = 2.0
DEFAULT_STEP_MAX_TOKENS = 768


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


@dataclass(frozen=True)
class LLMHTTPSettings:
    request_timeout_s: float
    connect_timeout_s: float
    sock_read_timeout_s: float
    max_connections: int
    keepalive_connections: int
    max_concurrency: int
    startup_jitter_s: float
    trust_env: bool = True


def resolve_llm_http_settings(
    *,
    worker_count: int = 1,
    session_routed: bool = False,
    trust_env: bool = True,
) -> LLMHTTPSettings:
    """
    Resolve all LLM HTTP/runtime settings from one place.

    Session-routed deployments default to a longer request timeout and a modest
    concurrency cap to avoid hammering a single routed backend during startup.
    """
    normalized_worker_count = max(1, int(worker_count))
    request_timeout_default = (
        SESSION_ROUTED_REQUEST_TIMEOUT_S if session_routed else DEFAULT_REQUEST_TIMEOUT_S
    )
    request_timeout_s = _env_float("AIEVOBOX_LLM_TIMEOUT_S", request_timeout_default)
    connect_timeout_s = _env_float("AIEVOBOX_LLM_CONNECT_TIMEOUT_S", DEFAULT_CONNECT_TIMEOUT_S)
    sock_read_timeout_s = _env_float("AIEVOBOX_LLM_SOCK_READ_TIMEOUT_S", request_timeout_s)
    max_connections = _env_int("AIEVOBOX_LLM_MAX_CONNECTIONS", max(128, normalized_worker_count * 8))
    max_concurrency_default = 64 if session_routed else 0
    max_concurrency = max(0, _env_int("AIEVOBOX_LLM_MAX_CONCURRENCY", max_concurrency_default))
    startup_jitter_s = max(0.0, _env_float("AIEVOBOX_LLM_STARTUP_JITTER_S", DEFAULT_STARTUP_JITTER_S))
    return LLMHTTPSettings(
        request_timeout_s=float(request_timeout_s),
        connect_timeout_s=float(connect_timeout_s),
        sock_read_timeout_s=float(sock_read_timeout_s),
        max_connections=int(max_connections),
        keepalive_connections=max(64, normalized_worker_count * 4),
        max_concurrency=int(max_concurrency),
        startup_jitter_s=float(startup_jitter_s),
        trust_env=bool(trust_env),
    )


class LLM:
    """
    基础 LLM 客户端封装，负责直接调用 OpenAI 兼容的 /chat/completions 接口。
    支持长时间重试以应对模型权重更新等场景。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 1,
        llm_proxy: Optional[str] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF,
        max_retry_delay: float = DEFAULT_MAX_RETRY_DELAY,
        retry_jitter_ratio: float = DEFAULT_RETRY_JITTER_RATIO,
        request_timeout_s: Optional[float] = None,
        connect_timeout_s: Optional[float] = None,
        sock_read_timeout_s: Optional[float] = None,
        max_tokens: Optional[int] = None,
        trust_env: bool = True,
        http_client: Optional[HttpClient] = None,
    ) -> None:
        http_settings = resolve_llm_http_settings(trust_env=trust_env)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.llm_proxy = llm_proxy if llm_proxy else None
        self.max_retries = int(max_retries)  # -1 表示无限重试
        self.retry_backoff = float(retry_backoff)
        self.max_retry_delay = float(max_retry_delay)
        self.retry_jitter_ratio = max(0.0, float(retry_jitter_ratio))
        self.request_timeout_s = max(
            1.0,
            float(http_settings.request_timeout_s if request_timeout_s is None else request_timeout_s),
        )
        self.connect_timeout_s = max(
            0.1,
            float(http_settings.connect_timeout_s if connect_timeout_s is None else connect_timeout_s),
        )
        self.sock_read_timeout_s = max(
            0.1,
            float(http_settings.sock_read_timeout_s if sock_read_timeout_s is None else sock_read_timeout_s),
        )
        self.trust_env = bool(http_settings.trust_env)
        self._http_client = http_client
        resolved_max_tokens = (
            _env_int("LLM_STEP_MAX_TOKENS", DEFAULT_STEP_MAX_TOKENS)
            if max_tokens is None
            else int(max_tokens)
        )
        self.max_tokens = resolved_max_tokens if resolved_max_tokens > 0 else None

    @staticmethod
    def _parse_response_payload(data: Dict[str, Any]) -> Dict[str, Any]:
        choice = data["choices"][0]
        content = choice["message"]["content"]
        finish_reason = choice.get("finish_reason", "stop")
        # 从 metadata 中获取 weight_version（SGLang 等引擎会返回）
        metadata = data.get("metadata") or {}
        weight_version = metadata.get("weight_version")
        return {
            "content": content,
            "finish_reason": finish_reason,
            "weight_version": weight_version,
        }

    def _retry_delay(self, attempt: int, *, rate_limited: bool = False) -> float:
        exponent = attempt if rate_limited else max(attempt - 1, 0)
        delay_base = min(self.retry_backoff * (2 ** exponent), self.max_retry_delay)
        return delay_base + random.uniform(0.0, delay_base * self.retry_jitter_ratio)

    def _retry_info(self, attempt: int) -> str:
        return "∞" if self.max_retries == -1 else f"{attempt}/{self.max_retries}"

    async def _post_with_shared_client(self, url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        request_kwargs: Dict[str, Any] = {
            "json": payload,
            "headers": headers,
        }
        if self.llm_proxy:
            request_kwargs["proxy"] = self.llm_proxy

        async with self._http_client.request("POST", url, **request_kwargs) as resp:
            logger.info("[LLM] POST %s got response status=%s", url, resp.status)
            resp.raise_for_status()
            return await resp.json()

    async def _post_with_standalone_client(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
    ) -> Dict[str, Any]:
        timeout = aiohttp.ClientTimeout(
            total=self.request_timeout_s,
            connect=self.connect_timeout_s,
            sock_read=self.sock_read_timeout_s,
        )
        async with aiohttp.ClientSession(timeout=timeout, trust_env=self.trust_env) as session:
            async with session.post(url, json=payload, headers=headers, proxy=self.llm_proxy) as resp:
                logger.info("[LLM] POST %s got response status=%s", url, resp.status)
                resp.raise_for_status()
                return await resp.json()

    async def generate(self, messages: List[dict]) -> Dict[str, Any]:
        """
        基于 OpenAI messages 列表调用 chat.completions（异步接口，内置失败重试）。

        messages 必须是形如 [{"role": "...", "content": "..."}] 的标准结构。

        返回:
            Dict 包含:
            - "content": str - LLM 生成的内容
            - "finish_reason": str - 结束原因 ("stop", "length" 等)

        重试策略：指数退避，最大延迟封顶，适应模型权重更新场景。
        """
        attempt = 0
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        last_error = None
        # max_retries == -1 表示无限重试
        while self.max_retries == -1 or attempt <= self.max_retries:
            try:
                logger.info("[LLM] POST %s (thread=%s, attempt=%d)", url, threading.current_thread().name, attempt)
                if self._http_client is not None:
                    data = await self._post_with_shared_client(url, payload, headers)
                else:
                    data = await self._post_with_standalone_client(url, payload, headers)
                logger.info("[LLM] POST %s returned", url)
                return self._parse_response_payload(data)
            except aiohttp.ClientResponseError as e:
                last_error = e
                attempt += 1
                if self.max_retries != -1 and attempt > self.max_retries:
                    break

                if e.status == 429:
                    delay = self._retry_delay(attempt, rate_limited=True)
                    logger.warning("[LLM] rate limited (429), retry in %.1fs (attempt=%s)", delay, self._retry_info(attempt))
                    await asyncio.sleep(delay)
                    continue

                if 500 <= e.status <= 599:
                    delay = self._retry_delay(attempt)
                    logger.warning(
                        "[LLM] upstream %s, retry in %.1fs (attempt=%s)",
                        e.status,
                        delay,
                        self._retry_info(attempt),
                    )
                    await asyncio.sleep(delay)
                    continue

                raise
            except (aiohttp.ServerDisconnectedError, aiohttp.ClientOSError, aiohttp.ClientConnectionError) as e:
                last_error = e
                attempt += 1
                if self.max_retries != -1 and attempt > self.max_retries:
                    break
                delay = self._retry_delay(attempt)
                logger.warning(
                    "[LLM] transport error, retry in %.1fs (attempt=%s): %s",
                    delay,
                    self._retry_info(attempt),
                    e,
                )
                await asyncio.sleep(delay)
            except asyncio.TimeoutError as e:
                last_error = e
                attempt += 1
                if self.max_retries != -1 and attempt > self.max_retries:
                    break
                delay = self._retry_delay(attempt)
                logger.warning("[LLM] timeout, retry in %.1fs (attempt=%s)", delay, self._retry_info(attempt))
                await asyncio.sleep(delay)
            except Exception as e:
                last_error = e
                attempt += 1
                if self.max_retries != -1 and attempt > self.max_retries:
                    break
                delay = self._retry_delay(attempt)
                logger.warning("[LLM] unexpected error, retry in %.1fs (attempt=%s): %s", delay, self._retry_info(attempt), e)
                await asyncio.sleep(delay)

        raise RuntimeError(
            f"[LLM] All retries exhausted (limit={self._retry_info(attempt)}). Last error: {last_error}"
        )
