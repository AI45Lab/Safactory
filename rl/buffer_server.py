import asyncio
import copy
import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
import numpy as np
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional

# Add rl directory to path for utils import
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from utils import get_env

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

# Add AIEvoBox to path
AIEVOBOX_ROOT = get_env("AIEVOBOX_ROOT")
if AIEVOBOX_ROOT not in sys.path:
    sys.path.insert(0, AIEVOBOX_ROOT)

from core.data_manager.manager import DataManager

# Setup logging
LOG_DIR = os.path.join(AIEVOBOX_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "buffer_server.log")

logger = logging.getLogger("buffer_server")
logger.setLevel(logging.DEBUG)

# File handler with rotation
file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=50*1024*1024, backupCount=5, encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
))
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s"
))
logger.addHandler(console_handler)

logger.info(f"Buffer Server logging to: {LOG_FILE}")

app = FastAPI(title="Rollout Buffer Server", debug=True)

# Track subprocesses
aievobox_process: Optional[subprocess.Popen] = None

# DataManager for querying the database
data_manager: Optional[DataManager] = None

# LLM Proxy URL (constructed from host and port)
_llm_proxy_host = get_env("LLM_PROXY_HOST")
_llm_proxy_port = get_env("LLM_PROXY_PORT")
llm_proxy_url: str = f"http://{_llm_proxy_host}:{_llm_proxy_port}/v1"

# Track last served step ID for cursor-based pagination
last_served_id: int = 0

# Pending step rows by session_id until that session reaches terminal state.
pending_rows_by_session: Dict[str, List[Dict[str, Any]]] = {}

# Completed sessions by original GRPO group_id.
completed_sessions_by_group: Dict[str, List[Dict[str, Any]]] = {}

# Group size (set by /start_rollout)
group_size: int = 1


@app.middleware("http")
async def set_body_size(request: Request, call_next):
    request._body_size_limit = 1_073_741_824  # 1GB
    response = await call_next(request)
    return response


class BufferResponse(BaseModel):
    success: bool
    message: str = ""
    data: Optional[Dict[str, Any]] = None


def _parse_timestamp(ts: Optional[str]) -> Optional[float]:
    """Parse timestamp string to float."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts).timestamp()
    except Exception:
        try:
            return float(ts)
        except Exception:
            return None


def _normalize_reward(value: Any) -> float:
    try:
        reward = float(value or 0.0)
    except (TypeError, ValueError):
        reward = 0.0
    return max(0.0, min(1.0, reward))


def _float_or_zero(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _row_step_id(row: Dict[str, Any]) -> int:
    try:
        return int(row.get("step_id", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _build_item_from_row(
    row: Dict[str, Any],
    *,
    reward_override: Optional[float] = None,
    train_group_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a database row to the expected item format."""
    # Parse stored prompt (JSON serialized messages list)
    prompt_str = row.get("prompt", "")
    if isinstance(prompt_str, str):
        base_messages = json.loads(prompt_str) if prompt_str else []
    else:
        base_messages = prompt_str
    messages = base_messages + [{"role": "assistant", "content": row.get("response", "")}]

    session_id = row.get("session_id", "")
    env_id = row.get("env_id", "")
    group_id = row.get("group_id", "")
    is_session_completed = _as_bool(row.get("is_session_completed", False))

    # 从 env_state 中解析 weight_version
    weight_version = 0
    if env_state_raw := row.get("env_state"):
        try:
            weight_version = int(json.loads(env_state_raw).get("weight_version") or 0)
        except Exception:
            weight_version = 0

    step_id = _row_step_id(row)
    truncated = _as_bool(row.get("truncated", False))
    raw_reward = row.get("total_reward", row.get("reward", 0.0)) if reward_override is None else reward_override
    reward = _float_or_zero(raw_reward)
    model_output_truncated = truncated and _float_or_zero(row.get("step_reward", row.get("reward", 0.0))) < 0.0
    reward = max(0.0, min(1.0, reward))
    train_group_id = train_group_id or str(group_id)

    extra_info = {
        "timestamp": _parse_timestamp(row.get("session_end_time")) or _parse_timestamp(row.get("timestamp")) or time.time(),
        "steps": step_id,
        "step_id": step_id,
        # 注意：finish_reason 与 truncated 不完全等价，finish_reason 仅用于训练侧标记截断状态
        "finish_reason": "length" if model_output_truncated else "stop",
        "session_id": session_id,
        "env_id": env_id,
        "group_id": group_id,
        "train_group_id": train_group_id,
        "is_session_completed": is_session_completed,
        "weight_version": weight_version,
        "truncated": truncated,
    }

    return {
        "uid": str(uuid.uuid4()),
        "instance_id": str(session_id),
        "messages": messages,
        "reward": reward,
        "extra_info": extra_info,
    }


async def fetch_new_items_from_db(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fetch new trainable step rows from the database using cursor-based pagination."""
    global data_manager, last_served_id

    if data_manager is None:
        return []

    rows = []
    try:
        fetched_rows = await data_manager.fetch_done_steps_with_context(
            after_id=last_served_id,
            limit=limit or max(100, group_size * 16)
        )
    except Exception as e:
        logger.error(f"fetch_done_steps_with_context error: {e}")
        return []

    for row in fetched_rows:
        step_pk = row.get("step_pk")
        try:
            # Always advance the cursor for rows returned by the storage layer.
            if step_pk is not None and (not last_served_id or step_pk > last_served_id):
                last_served_id = step_pk

            rows.append(row)
        except Exception as e:
            logger.error(f"Error reading row: {e}")
            continue

    return rows


def _build_completed_session(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None

    sorted_rows = sorted(rows, key=_row_step_id)
    terminal_row = next((row for row in reversed(sorted_rows) if _as_bool(row.get("is_session_completed", False))), sorted_rows[-1])
    session_id = str(terminal_row.get("session_id") or "")
    group_id = str(terminal_row.get("group_id") or "")
    if not session_id or not group_id:
        return None

    terminal_reward = _normalize_reward(terminal_row.get("total_reward", terminal_row.get("reward", 0.0)))
    items_by_step: Dict[int, Dict[str, Any]] = {}
    for row in sorted_rows:
        step_id = _row_step_id(row)
        if step_id <= 0:
            continue
        train_group_id = f"{group_id}:step:{step_id}"
        items_by_step[step_id] = _build_item_from_row(
            row,
            reward_override=terminal_reward,
            train_group_id=train_group_id,
        )

    if not items_by_step:
        return None

    return {
        "session_id": session_id,
        "group_id": group_id,
        "items_by_step": items_by_step,
    }


def accumulate_and_pop_ready_groups(new_rows: List[Dict[str, Any]]) -> tuple:
    """Accumulate step rows and return complete per-step GRPO groups."""
    global pending_rows_by_session, completed_sessions_by_group, group_size

    ready_groups = []
    ready_group_ids = []
    finished_session_ids = []

    for row in new_rows:
        session_id = str(row.get("session_id") or "")
        if not session_id:
            continue
        pending_rows_by_session.setdefault(session_id, []).append(row)
        if not _as_bool(row.get("is_session_completed", False)):
            continue

        session_rows = pending_rows_by_session.pop(session_id, [])
        completed_session = _build_completed_session(session_rows)
        if completed_session is None:
            continue

        group_id = completed_session["group_id"]
        bucket = completed_sessions_by_group.setdefault(group_id, [])
        if any(existing["session_id"] == session_id for existing in bucket):
            continue
        bucket.append(completed_session)

    # Check for complete groups
    to_delete = []
    for group_id, bucket in completed_sessions_by_group.items():
        while len(bucket) >= group_size:
            session_group = bucket[:group_size]
            del bucket[:group_size]

            common_steps = set(session_group[0]["items_by_step"].keys())
            for completed_session in session_group[1:]:
                common_steps.intersection_update(completed_session["items_by_step"].keys())

            for step_id in sorted(common_steps):
                train_group_id = f"{group_id}:step:{step_id}"
                ready_groups.append((
                    train_group_id,
                    [completed_session["items_by_step"][step_id] for completed_session in session_group],
                ))

            ready_group_ids.append(group_id)
            finished_session_ids.extend(completed_session["session_id"] for completed_session in session_group)

        if not bucket:
            to_delete.append(group_id)

    for k in to_delete:
        completed_sessions_by_group.pop(k, None)

    return ready_groups, ready_group_ids, finished_session_ids


@app.post("/get_rollout_data", response_model=BufferResponse)
async def get_rollout_data(request: Request):
    global pending_rows_by_session, completed_sessions_by_group

    # Fetch new step rows from database and accumulate completed sessions.
    new_rows = await fetch_new_items_from_db(limit=None)
    ready_groups, finished_group_ids, finished_session_ids = accumulate_and_pop_ready_groups(new_rows)

    # Log pending status
    pending_step_counts = {k: len(v) for k, v in pending_rows_by_session.items()}
    completed_session_counts = {k: len(v) for k, v in completed_sessions_by_group.items()}
    logger.info(
        "new_rows=%d, ready_groups=%d, pending_steps=%s, completed_sessions=%s",
        len(new_rows),
        len(ready_groups),
        pending_step_counts,
        completed_session_counts,
    )

    # Flatten groups to items
    ready_items = [item for _, group in ready_groups for item in group]
    rewards = [float(item.get("reward", 0.0)) for item in ready_items]

    total_samples = len(ready_items)
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

    # 统计权重版本信息，用于后续在 Slime 侧计算数据 age
    weight_versions: List[int] = []
    for item in ready_items:
        extra = item.get("extra_info") or {}
        wv = extra.get("weight_version", 0)
        try:
            weight_versions.append(int(wv))
        except Exception:
            weight_versions.append(0)

    if weight_versions:
        max_wv = max(weight_versions)
        mean_wv = sum(weight_versions) / len(weight_versions)
    else:
        max_wv = 0.0
        mean_wv = 0.0
    finished_groups = list(sorted(set(finished_group_ids)))
    finished_sessions = list(sorted(set(finished_session_ids)))

    meta_info = {
        "total_samples": total_samples,
        "avg_reward": avg_reward,
        "finished_groups": finished_groups,
        "finished_sessions": finished_sessions,
        "avg_weight_version": mean_wv,
        "max_weight_version": max_wv,
    }

    if total_samples == 0:
        return BufferResponse(
            success=False,
            message="No data available to read",
            data={"data": [], "meta_info": meta_info},
        )

    logger.info(f"Returning {total_samples} items")

    return BufferResponse(
        success=True,
        message=f"Successfully read {total_samples} items",
        data={"data": ready_items, "meta_info": meta_info},
    )


async def init_data_manager(job_session: str, storage_type: str, db_url: str, restart_training: bool = False):
    """Initialize the DataManager for querying the database."""
    global data_manager, last_served_id
    data_manager = DataManager(job_id=job_session, storage_type=storage_type, db_url=db_url)
    await data_manager.init()
    logger.info(f"DataManager initialized with {storage_type} DB: {db_url}, job_session: {job_session}")

    # Initialize cursor based on restart_training flag
    if restart_training:
        last_served_id = await data_manager.get_max_step_id()
        logger.info(f"restart_training=True, initialized last_served_id={last_served_id}")


def start_aievobox_process(data: dict):
    """Start AIEvoBox launcher.py as a subprocess.

    NOTE: LLM Proxy is now hosted in-process by slime_generator.
    It must already be running before this function is called.
    """
    global aievobox_process, group_size, last_served_id, pending_rows_by_session, completed_sessions_by_group, data_manager

    # Set group size (num_repeat_per_sample)
    group_size = int(data.get("num_repeat_per_sample", 16))

    # Clear state for new rollout
    restart_training = data.get("restart_training", False)
    if restart_training:
        pending_rows_by_session.clear()
        completed_sessions_by_group.clear()
        logger.info("restart_training=True, cleared pending rollout state")

    # Keep a single job_session for both reader and writer process.
    job_session = str(data.get("job_session") or uuid.uuid4().hex)
    
    # Mode
    mode = os.environ.get("AIEVOBOX_MODE", "local")

    # Database path
    storage_type = os.environ.get("STORAGE_TYPE", "sqlite")
    db_url = os.environ.get("AIEVOBOX_DB_URL", f"sqlite:///{AIEVOBOX_ROOT}/rl/rl.db")

    # Run async init in a new event loop (since we're in a thread)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            init_data_manager(job_session=job_session, storage_type=storage_type, db_url=db_url, restart_training=restart_training)
        )
    finally:
        loop.close()

    # Build launcher.py command line arguments
    aievobox_root = os.environ.get("AIEVOBOX_ROOT", "/root/AIEvoBox")
    launcher_script = os.path.join(aievobox_root, "launcher.py")
    env_root = get_env("AIEVOBOX_ENV_ROOT")
    env_config = os.environ.get("AIEVOBOX_ENV_CONFIG")
    max_steps = int(get_env("AIEVOBOX_MAX_STEPS") or 10)
    message_cut = int(get_env("AIEVOBOX_MESSAGE_CUT") or 0)
    llm_model = get_env("RL_MODEL") or "default"
    llm_temperature = float(get_env("LLM_TEMPERATURE") or 1.0)
    pool_size = int(get_env("AIEVOBOX_POOL_SIZE") or 16)
    rl_epoch = int(get_env("RL_EPOCH") or 1)
    env_transport = os.environ.get("AIEVOBOX_ENV_TRANSPORT", "http")
    multiplier = os.environ.get("AIEVOBOC_MULTIPLIER", 1.2)

    cmd = [
        "python3", launcher_script,
        "--mode", mode,
        "--db-path", db_url,
        "--storage-type", storage_type,
        *(["--env-config", env_config] if env_config else ["--env-root", env_root]),
        "--llm-base-url", llm_proxy_url,
        "--llm-model", llm_model,
        "--llm-temperature", str(llm_temperature),
        "--max-steps", str(max_steps),
        "--message-cut", str(message_cut),
        "--pool-size", str(pool_size),
        "--multiplier", str(multiplier),
        "--job-id", job_session,
        "--no-rebuild-table",
        "--rl-use-session-suffix-url",
        "--rl-group-size", str(group_size),
        "--rl-epoch", str(rl_epoch),
        "--env-transport", env_transport,
        "--env-http-timeout-s", "600",
    ]

    logger.info(f"Starting launcher.py: {' '.join(cmd)}")
    logger.info(f"Config: group_size={group_size}, db_url={db_url}")
    logger.info(f"LLM Proxy URL: {llm_proxy_url}")

    try:
        aievobox_process = subprocess.Popen(
            cmd,
            cwd=aievobox_root,
            stdout=None,  # Inherit stdout
            stderr=None,  # Inherit stderr
        )
        logger.info(f"launcher.py started with PID: {aievobox_process.pid}")
    except Exception as e:
        logger.error(f"Failed to start launcher.py: {e}")
        raise


@app.post("/start_rollout")
async def start_rollout(request: Request):
    global aievobox_process

    payload = await request.json()

    # Check if AIEvoBox is already running
    if aievobox_process is not None and aievobox_process.poll() is None:
        return {"message": "AIEvoBox is already running", "pid": aievobox_process.pid}

    # Start AIEvoBox in a background thread
    thread = threading.Thread(target=start_aievobox_process, args=(payload,), daemon=True)
    thread.start()

    return {"message": "Rollout started"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "aievobox_running": aievobox_process is not None and aievobox_process.poll() is None,
        "llm_proxy_running": llm_proxy_process is not None and llm_proxy_process.poll() is None,
        "data_manager_initialized": data_manager is not None,
    }


if __name__ == "__main__":
    port = int(get_env("BUFFER_SERVER_PORT"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        limit_concurrency=1000,  # Connection concurrency limit
        # limit_max_requests=1000000,  # Maximum request limit
        timeout_keep_alive=5,  # Keep-alive timeout,
    )
