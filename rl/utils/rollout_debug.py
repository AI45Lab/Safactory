"""Debug 埋点：记录 rollout 阶段从 buffer 拉到的每个 group 的 reward 分布、
过滤原因，以及每个 sample 的明细。每个 rollout_id 一个 JSONL 文件，
便于排查"全 0 / 全 1 组过多"等问题。

每行的 schema 之一：
- group 记录：见 RolloutGroupDebugLogger.log_group
- summary：rollout 结束写一行 type="summary" 的汇总
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import statistics
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional


logger = logging.getLogger(__name__)


# group 状态枚举（写到 JSONL 的 status 字段，并用作 counter key 的后缀）
STATUS_KEPT = "kept"
STATUS_DAPO_ALL_SAME = "dapo_all_same"
STATUS_WEIGHT_VERSION_SKEW = "weight_version_skew"
STATUS_DROP_UNMATCHED_TRAJECTORY = "drop_unmatched_trajectory"
STATUS_DROP_ASSEMBLY_ERROR = "drop_assembly_error"

ALL_STATUSES = (
    STATUS_KEPT,
    STATUS_DAPO_ALL_SAME,
    STATUS_WEIGHT_VERSION_SKEW,
    STATUS_DROP_UNMATCHED_TRAJECTORY,
    STATUS_DROP_ASSEMBLY_ERROR,
)

# extra_info 中可能有用、需要原样落盘的字段
_EXTRA_INFO_PASSTHROUGH_KEYS = (
    "task",
    "task_id",
    "task_type",
    "problem_id",
    "data_source",
    "split",
    "subset",
    "difficulty",
)

# 匹配 OpenAI 风格 base64 图片 data URI 前缀，如 data:image/png;base64,
_BASE64_IMAGE_RE = re.compile(r"^data:image/[^;,]+;base64,", re.IGNORECASE)


def _is_base64_image(value: Any) -> bool:
    return isinstance(value, str) and bool(_BASE64_IMAGE_RE.match(value))


def _summarize_base64_image(value: str) -> str:
    match = _BASE64_IMAGE_RE.match(value)
    if not match:
        return f"<base64 image stripped: bytes={len(value)}>"
    mime_part = match.group(0)  # e.g. "data:image/png;base64,"
    payload_len = len(value) - match.end()
    return f"<base64 image stripped: prefix={mime_part!r} payload_bytes={payload_len}>"


def _sanitize_content_item(item: Any) -> Any:
    """处理 content list 中的单个 part，把 base64 图片替换为占位符字符串。"""
    if not isinstance(item, dict):
        return item
    new_item = dict(item)

    image_url = new_item.get("image_url")
    if isinstance(image_url, dict):
        url = image_url.get("url")
        if _is_base64_image(url):
            new_image_url = dict(image_url)
            new_image_url["url"] = _summarize_base64_image(url)
            new_item["image_url"] = new_image_url
    elif _is_base64_image(image_url):
        new_item["image_url"] = _summarize_base64_image(image_url)

    image = new_item.get("image")
    if _is_base64_image(image):
        new_item["image"] = _summarize_base64_image(image)

    return new_item


def _sanitize_message_content(content: Any) -> Any:
    """content 可能是 str / list[dict] / 嵌套 dict；对其中的 base64 图片打码。"""
    if isinstance(content, list):
        return [_sanitize_content_item(item) for item in content]
    return content


def sanitize_messages_for_log(messages: Any) -> List[Dict[str, Any]]:
    """返回 messages 的深拷贝副本，去除其中的 base64 图片内容。

    - 若 message["content"] 是 JSON 编码的字符串（如 ``[{"type": ...}]``），
      会先尝试解码再打码；解码失败则原样保留。
    - 仅打码满足 ``data:image/<mime>;base64,...`` 前缀的字符串，避免误伤
      普通文本/URL/路径。
    """
    if not isinstance(messages, list):
        return []
    sanitized: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            sanitized.append(msg)
            continue
        new_msg = copy.deepcopy(msg)
        content = new_msg.get("content")
        if isinstance(content, str):
            stripped = content.lstrip()
            if stripped.startswith("[") or stripped.startswith("{"):
                try:
                    content = json.loads(content)
                except (json.JSONDecodeError, ValueError):
                    pass
        new_msg["content"] = _sanitize_message_content(content)
        sanitized.append(new_msg)
    return sanitized


def extract_assistant_replies(messages: Iterable[Dict[str, Any]]) -> List[Any]:
    """收集所有 assistant 角色的 content（已经过 sanitize），用于快速看模型的回复。"""
    replies: List[Any] = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            replies.append(msg.get("content"))
    return replies


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _reward_stats(rewards: List[Optional[float]]) -> Dict[str, Any]:
    clean = [r for r in rewards if r is not None]
    unique = sorted(set(clean))
    if clean:
        return {
            "min": min(clean),
            "max": max(clean),
            "mean": sum(clean) / len(clean),
            "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
            "unique_rewards": unique,
            "unique_count": len(unique),
            "all_same": len(unique) <= 1,
            "all_zero": all(r == 0 for r in clean),
            "all_one": all(r == 1 for r in clean),
            "missing_count": len(rewards) - len(clean),
        }
    return {
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "unique_rewards": [],
        "unique_count": 0,
        "all_same": False,
        "all_zero": False,
        "all_one": False,
        "missing_count": len(rewards),
    }


class RolloutGroupDebugLogger:
    """每个 rollout_id 一个 JSONL 文件，按 group 写入 reward 分布与过滤状态。

    使用方式：
        log_dir = os.path.join(AIEVOBOX_ROOT, "logs", "rollout_debug")
        logger = RolloutGroupDebugLogger(log_dir, rollout_id)
        logger.start_round()           # 每次 fetch 调一次
        logger.log_group(group, status="dapo_all_same", reason=...)
        ...
        logger.write_summary(buffer_length=..., target=...)
    """

    def __init__(self, log_dir: str, rollout_id: int) -> None:
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError as err:
            logger.warning("Failed to create rollout debug log dir %s: %s", log_dir, err)
        self.path = os.path.join(log_dir, f"rollout_groups_debug_{rollout_id}.jsonl")
        self.rollout_id = rollout_id
        self._round = 0
        self._lock = Lock()
        self._counters: Dict[str, int] = {
            "groups_total": 0,
            "groups_all_zero": 0,
            "groups_all_one": 0,
            "groups_all_same_other": 0,
            "groups_mixed": 0,
        }
        for status in ALL_STATUSES:
            self._counters[f"groups_{status}"] = 0

    def start_round(self) -> int:
        with self._lock:
            self._round += 1
            return self._round

    @property
    def counters(self) -> Dict[str, int]:
        return dict(self._counters)

    def log_group(
        self,
        group: List[Dict[str, Any]],
        status: str,
        reason: Optional[str] = None,
        n_samples_per_prompt: Optional[int] = None,
        current_version: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """写一条 group 级别的 JSONL 记录。

        status: 见模块顶部的 STATUS_* 常量。未识别的 status 仍会被写入，
        只是不会被汇总到固定 counter。
        """
        if status not in ALL_STATUSES:
            logger.warning("Unknown rollout debug group status: %s", status)

        record = self._build_group_record(
            group=group,
            status=status,
            reason=reason,
            n_samples_per_prompt=n_samples_per_prompt,
            current_version=current_version,
            extra=extra,
        )
        with self._lock:
            self._update_counters(record, status)
        self._write_line(record)

    def write_summary(self, **extra: Any) -> None:
        """rollout 结束时写一行 summary。"""
        with self._lock:
            payload: Dict[str, Any] = {
                "type": "summary",
                "rollout_id": self.rollout_id,
                "rounds": self._round,
                "counters": dict(self._counters),
            }
        if extra:
            payload.update(extra)
        self._write_line(payload)

    def _build_group_record(
        self,
        *,
        group: List[Dict[str, Any]],
        status: str,
        reason: Optional[str],
        n_samples_per_prompt: Optional[int],
        current_version: Optional[int],
        extra: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        rewards = [_safe_float(r.get("reward")) for r in group]
        finish_reasons: List[Optional[str]] = []
        weight_versions: List[int] = []
        session_ids: List[Optional[str]] = []
        group_ids: List[Optional[str]] = []
        instance_ids: List[Any] = []
        samples: List[Dict[str, Any]] = []

        for record in group:
            extra_info = record.get("extra_info") or {}
            try:
                weight_version = int(extra_info.get("weight_version", 0) or 0)
            except (TypeError, ValueError):
                weight_version = 0
            finish_reason = extra_info.get("finish_reason")
            session_id = extra_info.get("session_id")
            group_id = extra_info.get("group_id")
            instance_id = record.get("instance_id")

            finish_reasons.append(finish_reason)
            weight_versions.append(weight_version)
            session_ids.append(session_id)
            group_ids.append(group_id)
            instance_ids.append(instance_id)

            sample = {
                "uid": record.get("uid"),
                "instance_id": instance_id,
                "reward": record.get("reward"),
                "finish_reason": finish_reason,
                "weight_version": weight_version,
                "session_id": session_id,
                "group_id": group_id,
            }
            for key in _EXTRA_INFO_PASSTHROUGH_KEYS:
                if key in extra_info:
                    sample[key] = extra_info[key]

            # 输入给模型的 prompt + 模型回复（OpenAI 风格 messages，base64 图片打码）
            sanitized_messages = sanitize_messages_for_log(record.get("messages"))
            sample["messages"] = sanitized_messages
            sample["assistant_replies"] = extract_assistant_replies(sanitized_messages)

            samples.append(sample)

        primary_instance_id = instance_ids[0] if instance_ids else None

        record_payload: Dict[str, Any] = {
            "type": "group",
            "rollout_id": self.rollout_id,
            "round": self._round,
            "instance_id": primary_instance_id,
            "group_size": len(group),
            "expected_group_size": n_samples_per_prompt,
            "status": status,
            "reason": reason,
            "current_version": current_version,
            "rewards": rewards,
            "reward_stats": _reward_stats(rewards),
            "finish_reasons": finish_reasons,
            "weight_versions": weight_versions,
            "session_ids": session_ids,
            "group_ids": group_ids,
            "samples": samples,
        }
        if extra:
            record_payload["extra"] = extra
        return record_payload

    def _update_counters(self, record: Dict[str, Any], status: str) -> None:
        self._counters["groups_total"] += 1
        stats = record["reward_stats"]
        if stats["all_zero"]:
            self._counters["groups_all_zero"] += 1
        elif stats["all_one"]:
            self._counters["groups_all_one"] += 1
        elif stats["all_same"]:
            self._counters["groups_all_same_other"] += 1
        else:
            self._counters["groups_mixed"] += 1

        key = f"groups_{status}"
        if key in self._counters:
            self._counters[key] += 1

    def _write_line(self, payload: Dict[str, Any]) -> None:
        try:
            line = json.dumps(payload, ensure_ascii=False, default=str)
        except (TypeError, ValueError) as err:
            logger.warning("Failed to serialize rollout debug payload: %s", err)
            return
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as err:
            logger.warning("Failed to write rollout debug log %s: %s", self.path, err)


def iter_group_records(path: str) -> Iterable[Dict[str, Any]]:
    """读取一个 rollout_groups_debug_*.jsonl 文件的所有 group 记录（跳过 summary）。

    给离线分析脚本用。
    """
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "group":
                yield obj
