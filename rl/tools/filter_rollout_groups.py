#!/usr/bin/env python3
"""按 reward 条件筛选 rollout_groups_debug_*.jsonl 中的 group。

默认筛选"平均奖励 = 1.0"的 group，输出到 stdout（每行一个完整 group JSON）。
也支持指定输出文件、不同的目标均值、容差、是否只输出 summary。

示例：
    # 默认：mean = 1.0 全 1 组，全部输出到 stdout（含 messages，体积大）
    python filter_rollout_groups.py /path/to/rollout_groups_debug_0.jsonl

    # 写到文件并打印简要 summary
    python filter_rollout_groups.py /path/to/rollout_groups_debug_0.jsonl \
        -o /tmp/all_one_groups.jsonl

    # 只想要紧凑摘要，去掉 messages 字段
    python filter_rollout_groups.py /path/to/rollout_groups_debug_0.jsonl \
        --compact -o /tmp/all_one_summary.jsonl

    # 不指定 mean，按 all_one 标志过滤（等价于默认）
    python filter_rollout_groups.py /path/to/rollout_groups_debug_0.jsonl --all-one

    # 平均奖励在 [0.5, 1.0] 之间
    python filter_rollout_groups.py /path/to/rollout_groups_debug_0.jsonl \
        --mean-min 0.5 --mean-max 1.0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from typing import Any, Dict, Iterator, Optional, TextIO


def _iter_lines(path: str) -> Iterator[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                yield line


def _parse_record(line: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def _matches(
    record: Dict[str, Any],
    *,
    target_mean: Optional[float],
    tol: float,
    mean_min: Optional[float],
    mean_max: Optional[float],
    all_one: bool,
    status_filter: Optional[set],
) -> bool:
    stats = record.get("reward_stats") or {}
    mean = stats.get("mean")

    if status_filter is not None:
        if record.get("status") not in status_filter:
            return False

    if all_one:
        if not stats.get("all_one"):
            return False

    if target_mean is not None:
        if mean is None:
            return False
        if not math.isclose(mean, target_mean, abs_tol=tol):
            return False

    if mean_min is not None:
        if mean is None or mean < mean_min - tol:
            return False
    if mean_max is not None:
        if mean is None or mean > mean_max + tol:
            return False

    return True


def _compact(record: Dict[str, Any]) -> Dict[str, Any]:
    """去掉 samples 中的 messages（最占体积），只保留 reward 等关键字段。"""
    out = dict(record)
    samples = out.get("samples")
    if isinstance(samples, list):
        compact_samples = []
        for s in samples:
            if not isinstance(s, dict):
                compact_samples.append(s)
                continue
            sc = {k: v for k, v in s.items() if k not in ("messages", "assistant_replies")}
            # 保留 assistant_replies 的预览（前 200 字符），方便快速看
            replies = s.get("assistant_replies") or []
            sc["assistant_replies_preview"] = [
                (r if isinstance(r, str) else json.dumps(r, ensure_ascii=False))[:200]
                for r in replies
            ]
            compact_samples.append(sc)
        out["samples"] = compact_samples
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="筛选 rollout_groups_debug_*.jsonl 中符合 reward 条件的 group。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="输入 JSONL 文件路径")
    parser.add_argument(
        "-o", "--output",
        help="输出 JSONL 文件路径（默认写到 stdout）",
    )
    parser.add_argument(
        "--mean", dest="target_mean", type=float, default=1.0,
        help="目标平均奖励值（默认 1.0）。设为 None/-1 则不限制 mean。",
    )
    parser.add_argument(
        "--no-mean-filter", action="store_true",
        help="禁用 --mean，仅依赖 --all-one / --mean-min / --mean-max / --status 过滤",
    )
    parser.add_argument(
        "--tol", type=float, default=1e-9,
        help="--mean / --mean-min / --mean-max 的浮点容差（默认 1e-9）",
    )
    parser.add_argument(
        "--mean-min", type=float, default=None,
        help="平均奖励下界（包含）",
    )
    parser.add_argument(
        "--mean-max", type=float, default=None,
        help="平均奖励上界（包含）",
    )
    parser.add_argument(
        "--all-one", action="store_true",
        help="额外要求 reward_stats.all_one == true（即所有 sample reward 都是 1）",
    )
    parser.add_argument(
        "--status", action="append", default=None,
        help="只保留指定 status 的 group，可多次指定。"
             "可选值：kept / dapo_all_same / weight_version_skew / drop_unmatched_trajectory / drop_assembly_error",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="最多输出多少条匹配的 group",
    )
    parser.add_argument(
        "--compact", action="store_true",
        help="输出时去掉每个 sample 的 messages 字段，仅保留 assistant_replies_preview，体积更小",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="不打印 summary 到 stderr",
    )

    args = parser.parse_args()

    target_mean: Optional[float]
    if args.no_mean_filter:
        target_mean = None
    else:
        target_mean = args.target_mean

    status_filter = set(args.status) if args.status else None

    out_stream: TextIO
    if args.output and args.output != "-":
        out_stream = open(args.output, "w", encoding="utf-8")
    else:
        out_stream = sys.stdout

    total = 0
    matched = 0
    skipped_summary = 0
    status_tally: Counter = Counter()
    matched_status_tally: Counter = Counter()

    try:
        for line in _iter_lines(args.input):
            record = _parse_record(line)
            if record is None:
                continue
            rec_type = record.get("type", "group")
            if rec_type != "group":
                # summary 行直接跳过
                skipped_summary += 1
                continue

            total += 1
            status_tally[record.get("status", "unknown")] += 1

            if not _matches(
                record,
                target_mean=target_mean,
                tol=args.tol,
                mean_min=args.mean_min,
                mean_max=args.mean_max,
                all_one=args.all_one,
                status_filter=status_filter,
            ):
                continue

            matched += 1
            matched_status_tally[record.get("status", "unknown")] += 1

            payload = _compact(record) if args.compact else record
            out_stream.write(json.dumps(payload, ensure_ascii=False) + "\n")

            if args.limit is not None and matched >= args.limit:
                break
    finally:
        if out_stream is not sys.stdout:
            out_stream.close()

    if not args.quiet:
        sys.stderr.write(
            f"[filter_rollout_groups] scanned: {total} group(s), "
            f"matched: {matched}, skipped summary lines: {skipped_summary}\n"
        )
        sys.stderr.write(f"[filter_rollout_groups] all-status tally: {dict(status_tally)}\n")
        sys.stderr.write(f"[filter_rollout_groups] matched-status tally: {dict(matched_status_tally)}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
