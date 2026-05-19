#!/usr/bin/env python3
"""
LLM Proxy Server

Embedded in the slime_generator process.  Provides a single HTTP endpoint
consumed by AIEvoBox environments:

    POST /v1/{session_id}/chat/completions

Training data (tokens, masks, mm_train_inputs) is read directly from the
shared TrajectoryMaskBuilder in memory — no HTTP round-trip needed.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import sys
import time
from typing import Any, Optional

# Add rl directory to path for utils import
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from rl.utils import get_env

import httpx
from fastapi import FastAPI, HTTPException, Request

# Add AIEvoBox to path
AIEVOBOX_ROOT = get_env("AIEVOBOX_ROOT")
if AIEVOBOX_ROOT not in sys.path:
    sys.path.insert(0, AIEVOBOX_ROOT)

# llm_proxy runs in-process inside slime_generator. The host (slime_generator)
# installs the root file/console handlers via setup_process_logging, so this
# module just uses a named logger that propagates to root — no per-file
# handler here, which would duplicate every line into both log files.
logger = logging.getLogger("llm_proxy")
logger.info("LLM Proxy module imported (pid=%d)", os.getpid())

# Add mask directory to path
MASK_DIR = os.path.join(AIEVOBOX_ROOT, "rl", "mask")
if MASK_DIR not in sys.path:
    sys.path.insert(0, MASK_DIR)

from trajectory_mask_builder import PreparedPrompt, TrajectoryMaskBuilder

app = FastAPI(title="LLM Proxy Server", debug=True)

def _resolve_proxy_workers() -> int:
    default_workers = min(32, max(8, os.cpu_count() or 8))
    raw = os.getenv("AIEVOBOX_LLM_PROXY_WORKERS")
    if not raw:
        return default_workers
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "Invalid AIEVOBOX_LLM_PROXY_WORKERS=%r, fallback to default=%d",
            raw,
            default_workers,
        )
        return default_workers


_PROXY_WORKERS = _resolve_proxy_workers()


@app.middleware("http")
async def set_body_size(request: Request, call_next):
    # VLM requests can include base64 images in the prompt.
    request._body_size_limit = 1_073_741_824  # 1GB
    response = await call_next(request)
    return response


class ProxyState:
    """Global state for the proxy server."""

    def __init__(self):
        self.tokenizer = None
        self.processor: Optional[Any] = None
        self.trajectory_mask_builder: Optional[TrajectoryMaskBuilder] = None
        self.remote_engine_url: Optional[str] = None  # Base URL without /v1
        self._http_client: Optional[httpx.AsyncClient] = None
        self._builder_executor: Optional[ThreadPoolExecutor] = None
        self.max_length: Optional[int] = None
        # Sampling params
        self.temperature: float = 1.0
        self.top_p: float = 1.0

    def get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with connection pooling."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=2048,
                    max_keepalive_connections=512
                ),
                timeout=httpx.Timeout(
                    connect=300.0,
                    read=None,
                    write=None,
                    pool=None,
                ),
            )
        return self._http_client

    def get_builder_executor(self) -> ThreadPoolExecutor:
        """Get or create the dedicated builder executor."""
        if self._builder_executor is None:
            self._builder_executor = ThreadPoolExecutor(
                max_workers=_PROXY_WORKERS,
                thread_name_prefix="llm-proxy",
            )
            logger.info(
                "Initialized llm_proxy executor: workers=%d",
                _PROXY_WORKERS,
            )
        return self._builder_executor

    async def close(self):
        """Close the HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
        if self._builder_executor is not None:
            self._builder_executor.shutdown(wait=False, cancel_futures=True)
            self._builder_executor = None


STATE = ProxyState()


@app.post("/v1/{session_id}/chat/completions")
async def proxy_chat_completions(session_id: str, request: Request):
    """
    Proxy chat completions to the remote engine via /generate API.
    Records the trajectory (tokens, mask, logprobs) for training.
    """
    if STATE.remote_engine_url is None or STATE.tokenizer is None:
        logger.error(
            "Proxy not initialized: session=%s remote_engine_url=%s tokenizer_set=%s",
            session_id, STATE.remote_engine_url, STATE.tokenizer is not None,
        )
        raise HTTPException(status_code=503, detail="Proxy not initialized.")

    try:
        payload = await request.json()
    except Exception as e:
        logger.warning("Invalid JSON body: session=%s err=%s", session_id, repr(e))
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")

    messages = payload.get("messages", [])

    # Get sampling params from payload or use defaults
    temperature = payload.get("temperature", STATE.temperature)
    top_p = payload.get("top_p", STATE.top_p)
    max_tokens = payload.get("max_tokens")

    # Prepare input tokens (with multimodal state like cumulative image_data).
    # Run in thread pool to avoid blocking the event loop (CPU-bound tokenization)
    loop = asyncio.get_running_loop()
    builder_executor = STATE.get_builder_executor()
    prep: PreparedPrompt = await loop.run_in_executor(
        builder_executor,
        STATE.trajectory_mask_builder.prepare_generate_input,
        session_id,
        messages
    )
    input_ids = prep.input_ids
    image_data = prep.image_data

    # Calculate max_new_tokens
    if STATE.max_length is not None:
        remaining = STATE.max_length - len(input_ids)
        if max_tokens is not None:
            max_new_tokens = min(max_tokens, remaining)
        else:
            max_new_tokens = remaining

        if max_new_tokens <= 0:
            logger.warning(
                "Token budget exhausted: session=%s input_ids=%d max_length=%d "
                "requested_max_tokens=%s — returning empty response",
                session_id, len(input_ids), STATE.max_length, max_tokens,
            )
            # Return empty response
            return {
                "id": f"chatcmpl-{session_id}-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "proxy",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "length"
                }],
                "usage": {
                    "prompt_tokens": len(input_ids),
                    "completion_tokens": 0,
                    "total_tokens": len(input_ids)
                }
            }
    else:
        max_new_tokens = max_tokens or 256

    # Build /generate request
    generate_payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
        },
        "return_logprob": True,
        "stream": False,
    }
    if image_data:
        generate_payload["image_data"] = image_data

    # Call /generate endpoint
    http_client = STATE.get_http_client()
    url = f"{STATE.remote_engine_url}/generate"

    n_images = len(image_data) if image_data else 0
    req_started = time.monotonic()
    logger.debug(
        "Calling /generate: session=%s url=%s input_ids=%d max_new_tokens=%d "
        "temperature=%.3f top_p=%.3f images=%d",
        session_id, url, len(input_ids), max_new_tokens,
        temperature, top_p, n_images,
    )
    try:
        resp = await http_client.post(
            url,
            json=generate_payload,
            headers={"Content-Type": "application/json"}
        )
        resp.raise_for_status()
        resp_json = resp.json()
        elapsed_ms = (time.monotonic() - req_started) * 1000
        logger.debug(
            "Generate response: session=%s status=%d text_len=%d elapsed_ms=%.1f",
            session_id, resp.status_code, len(resp_json.get("text", "")), elapsed_ms,
        )
    except httpx.HTTPStatusError as e:
        # Upstream returned a non-2xx — capture body so we can see WHY (e.g.
        # "queue full", "weight reloading", "OOM"). Truncate to keep logs sane.
        elapsed_ms = (time.monotonic() - req_started) * 1000
        body_preview = (e.response.text or "")[:1024]
        logger.error(
            "Generate HTTP error: session=%s status=%d url=%s elapsed_ms=%.1f "
            "input_ids=%d max_new_tokens=%d images=%d body=%r",
            session_id, e.response.status_code, url, elapsed_ms,
            len(input_ids), max_new_tokens, n_images, body_preview,
        )
        raise HTTPException(
            status_code=502,
            detail=f"Upstream /generate returned {e.response.status_code}: {body_preview}",
        )
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        elapsed_ms = (time.monotonic() - req_started) * 1000
        logger.error(
            "Generate connect error: session=%s url=%s elapsed_ms=%.1f err=%s",
            session_id, url, elapsed_ms, repr(e),
        )
        raise HTTPException(status_code=502, detail=f"Cannot connect to /generate: {e}")
    except (httpx.ReadTimeout, httpx.PoolTimeout, httpx.WriteTimeout) as e:
        elapsed_ms = (time.monotonic() - req_started) * 1000
        logger.error(
            "Generate timeout: session=%s url=%s elapsed_ms=%.1f input_ids=%d "
            "max_new_tokens=%d err=%s",
            session_id, url, elapsed_ms, len(input_ids), max_new_tokens, repr(e),
        )
        raise HTTPException(status_code=504, detail=f"Timeout calling /generate: {e}")
    except Exception as e:
        import traceback
        elapsed_ms = (time.monotonic() - req_started) * 1000
        logger.error(
            "Generate failed (unexpected): session=%s url=%s elapsed_ms=%.1f err=%s\n%s",
            session_id, url, elapsed_ms, repr(e), traceback.format_exc(),
        )
        raise HTTPException(status_code=502, detail=f"Failed to call /generate: {e}")

    # Extract response data
    output_ids = resp_json.get("output_ids", [])
    output_logprobs = resp_json.get("meta_info", {}).get("output_token_logprobs", [])
    finish_reason_info = resp_json.get("meta_info", {}).get("finish_reason", {})

    # Determine finish_reason
    if isinstance(finish_reason_info, dict):
        finish_reason = finish_reason_info.get("type", "stop")
    else:
        finish_reason = "stop"

    # Get assistant_text from generate API response (already decoded)
    assistant_text = resp_json.get("text", "")

    # Save trajectory
    if STATE.trajectory_mask_builder is not None:
        try:
            await loop.run_in_executor(
                builder_executor,
                STATE.trajectory_mask_builder.record_generation,
                prep,
                output_ids,
                output_logprobs,
                assistant_text,
                finish_reason,
            )
        except Exception as e:
            import traceback
            logger.warning(
                "Failed to save trajectory: session=%s output_ids=%d finish=%s err=%s\n%s",
                session_id, len(output_ids), finish_reason, repr(e),
                traceback.format_exc(),
            )

    # Build OpenAI-compatible response
    response = {
        "id": f"chatcmpl-{session_id}-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "proxy",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": assistant_text
            },
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": len(input_ids),
            "completion_tokens": len(output_ids),
            "total_tokens": len(input_ids) + len(output_ids)
        },
        "metadata": {
            "weight_version": resp_json.get("meta_info", {}).get("weight_version")
        }
    }

    return response


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "initialized": STATE.tokenizer is not None,
        "remote_engine_url": STATE.remote_engine_url,
    }


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    await STATE.close()
