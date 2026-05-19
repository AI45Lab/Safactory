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
from typing import Any, Dict, List, Optional

# Add rl directory to path for utils import
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from utils import get_env, setup_process_logging, start_debugpy

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

# Add AIEvoBox to path
AIEVOBOX_ROOT = get_env("AIEVOBOX_ROOT")
if AIEVOBOX_ROOT not in sys.path:
    sys.path.insert(0, AIEVOBOX_ROOT)

from core.data_manager.manager import DataManager

# buffer_server is the session leader: it creates the per-run folder under
# logs/<timestamp>/ and registers it via .current_run + AIEVOBOX_RUN_DIR so
# llm_proxy / slime_generator / launcher land in the same dir. Logs from
# every named logger in this process flow through the root handler — chatty
# deps are pinned at WARNING by setup_process_logging.
RUN_DIR = setup_process_logging(
    "buffer_server",
    logs_root=os.path.join(AIEVOBOX_ROOT, "logs"),
    create_new_run_dir=True,
    run_name=os.environ.get("AIEVOBOX_RUN_NAME"),
)
LOG_FILE = os.path.join(RUN_DIR, "buffer_server.log")

logger = logging.getLogger("buffer_server")
logger.info("Buffer Server logging to: %s", LOG_FILE)
logger.info("Run directory: %s", RUN_DIR)

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

# Pending items by instance_id (for grouping)
pending_items_by_instance: Dict[str, List[Dict[str, Any]]] = {}

# Monotonic ready-group sequence per instance_id. This is separate from
# group_id: group_id identifies the prompt/task, while rollout_group_key
# identifies one popped batch of group_size samples for GRPO.
ready_group_sequence_by_instance: Dict[str, int] = {}

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


def _build_item_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
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

    # 从 env_state 中解析 weight_version
    weight_version = 0
    if env_state_raw := row.get("env_state"):
        weight_version = int(json.loads(env_state_raw)["weight_version"])

    extra_info = {
        "timestamp": _parse_timestamp(row.get("session_end_time")) or _parse_timestamp(row.get("timestamp")) or time.time(),
        "steps": row.get("step_id", 0),
        # 注意：finish_reason 与 truncated 不完全等价，finish_reason 仅用于训练侧标记截断状态
        "finish_reason": "length" if row.get("truncated", False) else "stop",
        "session_id": session_id,
        "env_id": env_id,
        "group_id": group_id,
        "weight_version": weight_version,
        "truncated": row.get("truncated", False),
        "is_session_completed": row.get("is_session_completed", False),
    }

    return {
        "uid": str(uuid.uuid4()),
        "instance_id": str(group_id),
        "messages": messages,
        "reward": float(row.get("reward", 0.0)),
        "extra_info": extra_info,
    }


async def fetch_new_items_from_db(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fetch new completed steps from the database using cursor-based pagination."""
    global data_manager, last_served_id

    if data_manager is None:
        return []

    items = []
    try:
        rows = await data_manager.fetch_done_steps_with_context(
            after_id=last_served_id,
            limit=limit or 100
        )
    except Exception as e:
        logger.error(f"fetch_done_steps_with_context error: {e}")
        return []

    for row in rows:
        step_pk = row.get("step_pk")
        try:
            item = _build_item_from_row(row)
            items.append(item)
            # Update cursor to the latest processed id
            if not last_served_id or step_pk > last_served_id:
                last_served_id = step_pk
        except Exception as e:
            logger.error(f"Error building item from row: {e}")
            continue

    return items


def accumulate_and_pop_ready_groups(new_items: List[Dict[str, Any]]) -> tuple:
    """Accumulate items and return ready groups."""
    global pending_items_by_instance, ready_group_sequence_by_instance, group_size

    ready_groups = []
    finished_instance_ids = []

    # Add new items to pending
    for item in new_items:
        instance_id = str(item.get("instance_id", ""))
        if not instance_id:
            continue
        pending_items_by_instance.setdefault(instance_id, []).append(item)

    # Check for complete groups
    to_delete = []
    for instance_id, bucket in pending_items_by_instance.items():
        while len(bucket) >= group_size:
            group = bucket[:group_size]
            del bucket[:group_size]
            seq = ready_group_sequence_by_instance.get(instance_id, 0)
            ready_group_sequence_by_instance[instance_id] = seq + 1
            rollout_group_key = f"{instance_id}:{seq}"
            for item in group:
                extra_info = item.setdefault("extra_info", {})
                extra_info["rollout_group_key"] = rollout_group_key
            ready_groups.append((instance_id, rollout_group_key, list(group)))
            finished_instance_ids.append(instance_id)
        if not bucket:
            to_delete.append(instance_id)

    for k in to_delete:
        pending_items_by_instance.pop(k, None)

    return ready_groups, finished_instance_ids


@app.post("/get_rollout_data", response_model=BufferResponse)
async def get_rollout_data(request: Request):
    global pending_items_by_instance

    # Fetch new items from database and accumulate groups
    new_items = await fetch_new_items_from_db(limit=None)
    ready_groups, finished_ids = accumulate_and_pop_ready_groups(new_items)

    # Log pending status
    pending_counts = {k: len(v) for k, v in pending_items_by_instance.items()}
    logger.info(f"new_items={len(new_items)}, ready_groups={len(ready_groups)}, pending={pending_counts}")

    # Flatten groups to items
    ready_items = [item for _, _, group in ready_groups for item in group]
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
    finished_groups = list(sorted(set(finished_ids)))

    meta_info = {
        "total_samples": total_samples,
        "avg_reward": avg_reward,
        "finished_groups": finished_groups,
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
    global aievobox_process, group_size, last_served_id, pending_items_by_instance, ready_group_sequence_by_instance, data_manager

    # Set group size (num_repeat_per_sample)
    group_size = int(data.get("num_repeat_per_sample", 16))

    # Clear state for new rollout
    restart_training = data.get("restart_training", False)
    if restart_training:
        pending_items_by_instance.clear()
        ready_group_sequence_by_instance.clear()
        logger.info("restart_training=True, cleared pending items")

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
    start_debugpy("buffer_server", default_port=5678)
    port = int(get_env("BUFFER_SERVER_PORT"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        limit_concurrency=1000,  # Connection concurrency limit
        # limit_max_requests=1000000,  # Maximum request limit
        timeout_keep_alive=5,  # Keep-alive timeout,
    )
