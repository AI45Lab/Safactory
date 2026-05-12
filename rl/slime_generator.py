import atexit
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional
import copy

# Add rl directory to path for utils import
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

_MASK_DIR = os.path.join(_SCRIPT_DIR, "mask")
if _MASK_DIR not in sys.path:
    sys.path.insert(0, _MASK_DIR)

from utils import get_env, AggType, MetricsRecorder

import aiohttp
import requests
import uvicorn
from transformers import AutoTokenizer
from transformers import AutoProcessor, PreTrainedTokenizerBase, ProcessorMixin

from slime.utils.async_utils import run
from slime.utils.types import Sample

import llm_proxy as _llm_proxy_module
from trajectory_mask_builder import TrajectoryMaskBuilder
from opd.teacher_log_probs import attach_teacher_log_probs

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)

# Global variables
TOKENIZER = None
TRAJECTORY_MASK_BUILDER = None
START_ROLLOUT = True
_LLM_PROXY_STARTED = False
_TRAININFO_EXECUTOR: Optional[ThreadPoolExecutor] = None

_llm_proxy_port = int(get_env("LLM_PROXY_PORT"))


def _resolve_traininfo_workers() -> int:
    default_workers = min(16, max(4, (os.cpu_count() or 8) // 2))
    raw = os.getenv("AIEVOBOX_TRAININFO_WORKERS")
    if not raw:
        return default_workers
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "Invalid AIEVOBOX_TRAININFO_WORKERS=%r, fallback to default=%d",
            raw,
            default_workers,
        )
        return default_workers


_TRAININFO_WORKERS = _resolve_traininfo_workers()


def _get_traininfo_executor() -> ThreadPoolExecutor:
    global _TRAININFO_EXECUTOR

    if _TRAININFO_EXECUTOR is None:
        _TRAININFO_EXECUTOR = ThreadPoolExecutor(
            max_workers=_TRAININFO_WORKERS,
            thread_name_prefix="traininfo",
        )
        logger.info(
            "Initialized traininfo executor: workers=%d",
            _TRAININFO_WORKERS,
        )
    return _TRAININFO_EXECUTOR


def _shutdown_traininfo_executor() -> None:
    global _TRAININFO_EXECUTOR

    if _TRAININFO_EXECUTOR is not None:
        _TRAININFO_EXECUTOR.shutdown(wait=False, cancel_futures=True)
        _TRAININFO_EXECUTOR = None


atexit.register(_shutdown_traininfo_executor)


def decode_tokens_with_mask_debug(tokenizer, token_ids, loss_mask):
    """
    token_ids: 全部 tokens
    loss_mask: 从第一个 1 开始的 mask（已去掉尾部 0）

    返回 [(text, 0/1), ...] 合并连续相同 mask 的段
    """
    if not token_ids:
        return []

    # 前面没有 mask 覆盖的部分，mask 都是 0
    prefix_len = len(token_ids) - len(loss_mask)
    full_mask = [0] * prefix_len + loss_mask

    # 合并连续相同 mask 的段
    segments = []
    current_tokens = [token_ids[0]]
    current_mask = full_mask[0]

    for tid, mask in zip(token_ids[1:], full_mask[1:]):
        if mask == current_mask:
            current_tokens.append(tid)
        else:
            segments.append((tokenizer.decode(current_tokens), current_mask))
            current_tokens = [tid]
            current_mask = mask

    if current_tokens:
        segments.append((tokenizer.decode(current_tokens), current_mask))

    return segments


def write_debug_to_file(
    tokenizer,
    rollout_id: int,
    record: Dict,
    oai_messages: List[Dict],
    token_ids: List[int],
    loss_mask: List[int],
    response_length: int,
):
    """将训练数据的调试信息写入文件。"""
    debug_dir = os.path.join(get_env("AIEVOBOX_ROOT"), "logs")
    os.makedirs(debug_dir, exist_ok=True)

    debug_segments = decode_tokens_with_mask_debug(tokenizer, token_ids, loss_mask)

    # 解析 messages 中的 JSON content
    oai_messages_parsed = copy.deepcopy(oai_messages)
    for msg in oai_messages_parsed:
        if isinstance(msg.get("content"), str):
            try:
                msg["content"] = json.loads(msg["content"])
            except json.JSONDecodeError:
                pass

    debug_file = os.path.join(debug_dir, f"train_{rollout_id}.log")
    with open(debug_file, "a+", encoding="utf-8") as f:
        f.write(json.dumps({
            "messages": oai_messages_parsed,
            "debug_segments": debug_segments,
            "index": record["instance_id"],
            "prompt": record["uid"],
            "tokens": token_ids,
            "response_length": response_length,
            "reward": record["reward"],
            "status": (
                "completed"
                if "finish_reason" not in record["extra_info"]
                or record["extra_info"]["finish_reason"] != "length"
                else "truncated"
            ),
            "loss_mask": loss_mask,
            "metadata": record["extra_info"],
        }, ensure_ascii=False) + "\n")

def _init_llm_proxy_server(args):
    """Initialize tokenizer, processor, TrajectoryMaskBuilder, and start the
    llm_proxy HTTP server in a background daemon thread.

    This is called once (idempotent).  The HTTP server is needed so that
    AIEvoBox environments can hit ``/v1/{session_id}/chat/completions``.
    Training data is read directly from TRAJECTORY_MASK_BUILDER in memory.
    """
    global TOKENIZER, TRAJECTORY_MASK_BUILDER, _LLM_PROXY_STARTED

    if _LLM_PROXY_STARTED:
        return

    # 1. Tokenizer
    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

    # 2. Processor (optional, VLM only)
    processor = None
    try:
        proc = AutoProcessor.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        if isinstance(proc, PreTrainedTokenizerBase) or not isinstance(proc, ProcessorMixin):
            proc = None
        processor = proc
    except Exception:
        processor = None

    # 3. TrajectoryMaskBuilder
    TRAJECTORY_MASK_BUILDER = TrajectoryMaskBuilder(TOKENIZER, processor)

    # 4. Wire into llm_proxy module STATE (shared in-process)
    state = _llm_proxy_module.STATE
    state.tokenizer = TOKENIZER
    state.processor = processor
    state.trajectory_mask_builder = TRAJECTORY_MASK_BUILDER

    remote_engine_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    state.remote_engine_url = remote_engine_url
    max_length_str = os.environ.get("LLM_MAX_LENGTH")
    state.max_length = int(max_length_str) if max_length_str else None
    state.temperature = float(os.environ.get("LLM_TEMPERATURE", "1.0"))
    state.top_p = float(os.environ.get("LLM_TOP_P", "1.0"))

    # 5. Start uvicorn in a daemon thread
    def _run_server():
        uvicorn.run(
            _llm_proxy_module.app,
            host="0.0.0.0",
            port=_llm_proxy_port,
            timeout_keep_alive=5,
            log_level="warning",
        )

    server_thread = threading.Thread(target=_run_server, daemon=True)
    server_thread.start()

    # 6. Wait until the server is ready
    for _ in range(30):
        try:
            r = requests.get(f"http://127.0.0.1:{_llm_proxy_port}/health", timeout=2)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        raise RuntimeError("LLM proxy HTTP server failed to start within 30 s")

    logger.info(f"LLM proxy server started in-process on port {_llm_proxy_port}")
    _LLM_PROXY_STARTED = True


def build_loss_mask_from_response_mask(
    tokens: List[int],
    response_mask: List[int],
) -> tuple[List[int], List[int], int]:
    """Convert full-length token-level response_mask to (token_ids, loss_mask, response_length).

    Returns:
        - token_ids: 完整的 tokens（未截取）
        - loss_mask: 从第一个 mask=1 的位置开始截取到末尾
        - response_length: loss_mask 的长度

    注意：token_ids 和 loss_mask 的长度不同，loss_mask 对应 token_ids 的后半部分。
    """
    # 边界情况：空输入
    if not tokens or not response_mask:
        return list(tokens or []), [], 0

    # 验证长度一致性
    if len(tokens) != len(response_mask):
        raise ValueError(f"Tokens and response_mask have different lengths: {len(tokens)} != {len(response_mask)}")

    # 找到第一个生成 token 的位置（mask=1）
    try:
        first_generated_idx = response_mask.index(1)
    except ValueError:
        # 没有生成内容（全是 mask=0）
        return list(tokens), [], 0

    # 从第一个生成 token 开始截取 loss_mask
    loss_mask = response_mask[first_generated_idx:]
    response_length = len(loss_mask)

    return tokens, loss_mask, response_length


def _get_record_training_info(record: Dict[str, Any]) -> Dict[str, Any]:
    oai_messages = record["messages"]
    session_id = record["extra_info"].get("session_id", "")
    tokens, response_mask, _image_data, messages_str, mm_train_inputs = TRAJECTORY_MASK_BUILDER.get_training_info(
        session_id,
        oai_messages,
    )
    return {
        "tokens": tokens,
        "response_mask": response_mask,
        "messages_str": messages_str,
        "mm_train_inputs": mm_train_inputs,
    }


def group_by_train_group_id(results: List[Dict]) -> List[List[Dict]]:
    """按 extra_info.train_group_id 将样本分组。

    Args:
        results: 样本列表，每个样本必须包含 extra_info.train_group_id

    Returns:
        分组后的样本列表 List[List[Dict]]
    """
    if not results:
        return []

    groups = {}
    for item in results:
        extra_info = item.get("extra_info") or {}
        train_group_id = str(extra_info.get("train_group_id") or extra_info.get("group_id") or "")
        if not train_group_id:
            raise ValueError("extra_info.train_group_id must be in item")
        if train_group_id not in groups:
            groups[train_group_id] = []
        groups[train_group_id].append(item)

    return list(groups.values())


async def get_rollout_data(api_base_url: str) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        while True:
            async with session.post(
                f"{api_base_url}/get_rollout_data", json={}, timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                response.raise_for_status()
                resp_json = await response.json()
                if resp_json["success"]:
                    break
            await asyncio.sleep(3)
            if time.time() - start_time > 30:
                print("rollout data is not ready, have been waiting for 30 seconds")
                # Reset start_time to continue waiting or handle timeout differently
                start_time = time.time()  # Or raise an exception, or return empty list

        data = resp_json["data"]
        meta_info = {}
        if isinstance(data, list):
            if "data" in data[0]:
                data = [item["data"] for item in data]
        elif isinstance(data, dict):
            if "data" in data:
                meta_info = data["meta_info"]
                data = data["data"]
        print(f"Meta info: {meta_info}")
        required_keys = {"uid", "instance_id", "messages", "reward", "extra_info"}
        for item in data:
            if not required_keys.issubset(item.keys()):
                raise ValueError(f"Missing required keys in response item: {item}")

        return data, meta_info


def start_rollout(api_base_url: str, args, metadata):
    url = f"{api_base_url}/start_rollout"
    print(f"metadata: {metadata}")
    finished_group_id_list = [item for sublist in metadata.values() for item in sublist]
    restart_training = os.environ.get("SLIME_ROLLBUF_RESTART_TRAINING", "True").strip().lower() == "true"
    payload = {
        "num_process": str(getattr(args, "rollout_num_process", 100)),
        "num_epoch": str(args.num_epoch or 3),
        "remote_engine_url": f"http://{args.sglang_router_ip}:{args.sglang_router_port}",
        "remote_buffer_url": args.rollout_buffer_url,
        "task_type": args.rollout_task_type,
        "num_repeat_per_sample": int(args.n_samples_per_prompt),
        "max_tokens": int(args.rollout_max_response_len),
        "sampling_params": {
            "max_tokens": int(args.rollout_max_response_len),
            "temperature": args.rollout_temperature,
            "top_p": args.rollout_top_p,
        },
        "tokenizer_path": args.hf_checkpoint,
        "skip_instance_ids": finished_group_id_list,
        "restart_training": restart_training,
    }
    print("start rollout with payload: ", payload)

    while True:
        try:
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            print(f"[start_rollout] Success: {data}")
            return data
        except Exception as e:
            print(f"[start_rollout] Failed to send rollout config: {e}")


def filter_by_weight_version(data_buffer, current_version: int, off_by_n: int = 0):
    """根据权重版本过滤 buffer 中的数据。

    过滤掉那些权重版本与当前版本差距超过 off_by_n 的样本。

    Args:
        data_buffer: 数据 buffer
        current_version: 当前权重版本（当前 pipeline 中通常是 rollout_id + 1）
        off_by_n: 允许的最大权重差，默认为 0（只保留当前版本的数据）
    """
    buffer_length = data_buffer.get_buffer_length()
    if buffer_length == 0:
        return

    # 获取所有样本
    all_samples = data_buffer.get_samples(buffer_length)

    # 过滤样本
    filtered_samples = []
    for sample_group in all_samples:
        filtered_group = []
        len_sample_group = len(sample_group)
        for sample in sample_group:
            metadata = getattr(sample, "metadata", None) or {}
            sample_version = metadata.get("weight_version", 0)
            try:
                sample_version = int(sample_version)
            except (ValueError, TypeError):
                sample_version = 0

            # 检查权重版本差距是否在允许范围内
            if current_version - sample_version <= off_by_n:
                filtered_group.append(sample)
            else:
                logger.debug(
                    f"Filtered out sample with weight_version={sample_version}, "
                    f"current_version={current_version}, off_by_n={off_by_n}"
                )

        if filtered_group and len(filtered_group) == len_sample_group:
            filtered_samples.append(filtered_group)

    if filtered_samples:
        data_buffer.add_samples(filtered_samples)

async def generate_rollout_async(args, rollout_id: int, data_buffer, evaluation: bool = False) -> Dict[str, Any]:
    if evaluation:
        raise NotImplementedError("Evaluation rollout is not implemented")

    metrics = MetricsRecorder()
    print("rollout_id: ", rollout_id)
    current_version = rollout_id + 1

    # 根据weight_version过滤已完成的数据
    off_by_n = int(get_env("RL_OFF_BY_N"))
    dapo_filter_enabled = os.environ.get("DAPO_filter", "true").strip().lower() in ("1", "true", "yes", "on")
    filter_by_weight_version(data_buffer, current_version=current_version, off_by_n=off_by_n)
    buffer_length = data_buffer.get_buffer_length()
    needed_groups = max(0, args.rollout_batch_size - buffer_length)
    data_number_to_fetch = needed_groups * args.n_samples_per_prompt
    print(f"INFO: buffer length: {buffer_length}, data_number_to_fetch: {data_number_to_fetch}")
    if needed_groups <= 0:
        print(
            f"❕buffer length: {data_buffer.get_buffer_length()}, buffer has enough data, return {args.rollout_batch_size} prompts"
        )
        final_return_results = data_buffer.get_samples(args.rollout_batch_size)
        for group in final_return_results:
            for sample in group:
                if sample.reward is None:
                    raise RuntimeError(
                        "Encountered reward=None after rollout assembly. "
                        "The rollout buffer is likely underfilled."
                    )
                metrics.record("used/reward", float(sample.reward), AggType.MEAN)
                metrics.record("used/response_length", float(sample.response_length), AggType.MEAN)
                meta = getattr(sample, "metadata", {}) or {}
                metrics.record("used/weight_version", float(meta.get("weight_version", 0)), AggType.MEAN)
        metrics.record("used/count", float(sum(len(g) for g in final_return_results)), AggType.SUM)
        metrics.push(step=rollout_id)
        return final_return_results
    base_url = args.rollout_buffer_url
    tokenizer = TOKENIZER
    retry_times = 0
    all_meta_info = []

    if args.fetch_trajectory_retry_times == -1:
        print(
            f"⚠️  [get_rollout_data] Fetch trajectory retry times set to -1, will retry indefinitely until sufficient data is collected"
        )

    # 持续获取数据，直到 buffer 中有足够的可训练 groups
    while data_buffer.get_buffer_length() < args.rollout_batch_size and (
        args.fetch_trajectory_retry_times == -1 or retry_times < args.fetch_trajectory_retry_times
    ):
        retry_times += 1
        try:
            remaining_groups = args.rollout_batch_size - data_buffer.get_buffer_length()
            fetch_sample_count = remaining_groups * args.n_samples_per_prompt
            print(f"need sample count: fetch_sample_count: {fetch_sample_count}")
            raw_results = []

            while len(raw_results) < fetch_sample_count:
                await asyncio.sleep(5)
                data, meta_info = await get_rollout_data(api_base_url=base_url)
                raw_results.extend(data)
                if meta_info:
                    all_meta_info.append(meta_info)
                print(f"get rollout data with length: {len(raw_results)}")

            # 从 extra_info 中获取 weight_version，记录 fetched metrics
            for record in raw_results:
                extra_info = record.get("extra_info") or {}
                record["weight_version"] = extra_info.get("weight_version", 0)
                metrics.record("fetched/reward", float(record.get("reward", 0)), AggType.MEAN)
                metrics.record("fetched/weight_version", float(record["weight_version"]), AggType.MEAN)
            metrics.record("fetched/count", float(len(raw_results)), AggType.SUM)

            # 按训练 group_id 分组；instance_id 保持为唯一 session_id。
            grouped_results = group_by_train_group_id(raw_results)

            # 按 group 过滤：group 中所有 sample 都必须符合版本要求
            valid_groups = []
            for group in grouped_results:
                if len(group) != args.n_samples_per_prompt:
                    train_group_id = (group[0].get("extra_info") or {}).get("train_group_id") if group else None
                    logger.warning(
                        "Drop rollout group with unexpected size: train_group_id=%s size=%d expected=%d",
                        train_group_id,
                        len(group),
                        args.n_samples_per_prompt,
                    )
                    continue
                rewards = [record.get("reward") for record in group]
                if dapo_filter_enabled and len(set(rewards)) == 1:
                    logger.info(
                        f"Filtered out group with rewards={rewards}, "
                        f"current_version={current_version}"
                    )
                    continue
                if all(current_version - record.get("weight_version", 0) <= off_by_n for record in group):
                    valid_groups.append(group)
                else:
                    # 记录被过滤的 group 信息
                    versions = [record.get("weight_version", 0) for record in group]
                    logger.info(
                        f"Filtered out group with weight_versions={versions}, "
                        f"current_version={current_version}, off_by_n={off_by_n}"
                    )

            print(f"✅ Valid groups collected this round: {len(valid_groups)}")

            sample_results = []
            touched_session_ids = set()
            try:
                flat_records = [record for group_record in valid_groups for record in group_record]
                for record in flat_records:
                    session_id = record["extra_info"].get("session_id", "")
                    if session_id:
                        touched_session_ids.add(session_id)

                loop = asyncio.get_running_loop()
                traininfo_executor = _get_traininfo_executor()
                training_info_results = await asyncio.gather(
                    *(loop.run_in_executor(traininfo_executor, _get_record_training_info, record) for record in flat_records),
                    return_exceptions=True,
                )

                result_offset = 0
                for group_record in valid_groups:
                    group_results = []
                    drop_group = False
                    group_training_infos = training_info_results[result_offset:result_offset + len(group_record)]
                    result_offset += len(group_record)

                    for record, training_info in zip(group_record, group_training_infos):
                        oai_messages = record["messages"]
                        session_id = record["extra_info"].get("session_id", "")
                        group_id = record["extra_info"].get("group_id", "")
                        if isinstance(training_info, Exception):
                            logger.error(
                                "Drop rollout group due to sample assembly error: instance_id=%s session_id=%s group_id=%s",
                                record.get("instance_id"),
                                session_id,
                                group_id,
                                exc_info=(type(training_info), training_info, training_info.__traceback__),
                            )
                            drop_group = True
                            break

                        try:
                            max_length = _llm_proxy_module.STATE.max_length
                            tokens = training_info["tokens"]
                            response_mask = training_info["response_mask"]
                            _messages_str = training_info["messages_str"]
                            mm_train_inputs = training_info["mm_train_inputs"]
                            if not tokens and not response_mask and _messages_str == "":
                                logger.warning(
                                    "Drop rollout group due to unmatched trajectory: instance_id=%s session_id=%s group_id=%s",
                                    record.get("instance_id"),
                                    session_id,
                                    group_id,
                                )
                                drop_group = True
                                break
                            if max_length is not None and len(tokens) > max_length:
                                tokens = tokens[:max_length]
                                response_mask = response_mask[:max_length]

                            token_ids, loss_mask, response_length = build_loss_mask_from_response_mask(tokens, response_mask)
                            write_debug_to_file(tokenizer, rollout_id, record, oai_messages, token_ids, loss_mask, response_length)

                            metadata = dict(record["extra_info"])
                            sample = Sample(
                                index=record["instance_id"],
                                prompt=record["uid"],
                                tokens=token_ids,
                                response_length=response_length,
                                reward=record["reward"],
                                status=(
                                    Sample.Status.COMPLETED
                                    if "finish_reason" not in record["extra_info"]
                                    or record["extra_info"]["finish_reason"] != "length"
                                    else Sample.Status.TRUNCATED
                                ),
                                loss_mask=loss_mask,
                                metadata=metadata,
                            )

                            if mm_train_inputs is not None:
                                sample.multimodal_train_inputs = mm_train_inputs

                            group_results.append(sample)
                        except Exception:
                            logger.exception(
                                "Drop rollout group due to sample assembly error: instance_id=%s session_id=%s group_id=%s",
                                record.get("instance_id"),
                                session_id,
                                group_id,
                            )
                            drop_group = True
                            break


                    if drop_group:
                        continue
                    sample_results.append(group_results)
            finally:
                if TRAJECTORY_MASK_BUILDER is not None:
                    for session_id in touched_session_ids:
                        TRAJECTORY_MASK_BUILDER.clear_session(session_id)
            data_buffer.add_samples(sample_results)
            print(
                "✅ Trainable groups added this round: "
                f"{len(sample_results)}, buffer length: {data_buffer.get_buffer_length()}/{args.rollout_batch_size}"
            )

        except Exception as err:
            print(f"[get_rollout_data] Failed to get rollout data: {err}, retry times: {retry_times}")

    if len(all_meta_info) > 0 and "finished_groups" in all_meta_info[0]:
        finished_group_id_list = []
        for item in all_meta_info:
            finished_group_id_list.extend(item["finished_groups"])

        data_buffer.update_metadata({str(rollout_id): finished_group_id_list})

    print("finally buffered trainable group count: ", data_buffer.get_buffer_length())
    if data_buffer.get_buffer_length() < args.rollout_batch_size:
        raise RuntimeError(
            "Insufficient trainable rollout groups after filtering and trajectory matching: "
            f"buffer_length={data_buffer.get_buffer_length()}, required={args.rollout_batch_size}"
        )

    final_return_results = data_buffer.get_samples(args.rollout_batch_size)
    for group in final_return_results:
        for sample in group:
            if sample.reward is None:
                raise RuntimeError(
                    "Encountered reward=None after rollout assembly. "
                    "The rollout buffer is likely underfilled."
                )
            metrics.record("used/reward", float(sample.reward), AggType.MEAN)
            metrics.record("used/response_length", float(sample.response_length), AggType.MEAN)
            meta = getattr(sample, "metadata", {}) or {}
            metrics.record("used/weight_version", float(meta.get("weight_version", 0)), AggType.MEAN)
    metrics.record("used/count", float(sum(len(g) for g in final_return_results)), AggType.SUM)
    metrics.push(step=rollout_id)

    return final_return_results


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """Generate rollout for both training and evaluation."""
    global START_ROLLOUT

    # Initialize tokenizer + processor + llm_proxy HTTP server (once).
    # Must happen BEFORE start_rollout, because buffer_server will launch
    # AIEvoBox environments that immediately connect to the HTTP server.
    _init_llm_proxy_server(args)

    if START_ROLLOUT:
        metadata = data_buffer.get_metadata()
        start_inform = start_rollout(args.rollout_buffer_url, args, metadata)
        print(f"start rollout with payload: {start_inform}")
        print(f"start rollout id: {rollout_id}")
        START_ROLLOUT = False

    sample_groups = run(generate_rollout_async(args, rollout_id, data_buffer, evaluation))
    if evaluation:
        return sample_groups
    return run(attach_teacher_log_probs(args, sample_groups))
