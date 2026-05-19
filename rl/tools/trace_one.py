#!/usr/bin/env python3
"""Trace one rollout end-to-end across all log files in a run dir.

Usage:
    python rl/tools/trace_one.py --id <session_id_or_instance_id>
    python rl/tools/trace_one.py --id <id> --run-dir logs/20260517-220930
    python rl/tools/trace_one.py --list-sessions <instance_id>

The script joins existing logs by ID — no extra instrumentation required.

Inputs accepted:
  * session_id: single rollout (one trajectory). The strongest correlation key.
  * instance_id: a whole group (group_size rollouts of the same task).
                 The script lists the contained session_ids; pass one back
                 with --id to drill in.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple


# --- Run dir resolution -----------------------------------------------------

def _read_marker(logs_root: str) -> Optional[str]:
    p = os.path.join(logs_root, ".current_run")
    try:
        with open(p) as f:
            v = f.read().strip()
        return v if v and os.path.isdir(v) else None
    except OSError:
        return None


def resolve_run_dir(arg: Optional[str]) -> str:
    if arg:
        return os.path.abspath(arg)
    env = os.environ.get("AIEVOBOX_RUN_DIR")
    if env and os.path.isdir(env):
        return env
    aievobox_root = os.environ.get(
        "AIEVOBOX_ROOT", "/mnt/shared-storage-user/chenxinquan/Safactory"
    )
    marker = _read_marker(os.path.join(aievobox_root, "logs"))
    if marker:
        return marker
    raise SystemExit(
        "Could not resolve run dir. Pass --run-dir or set AIEVOBOX_RUN_DIR."
    )


# --- Timestamp parsing ------------------------------------------------------
# Handles both formats we use:
#   buffer_server / llm_proxy:  "2026-05-17 22:29:28,446 [LEVEL] logger: msg"
#   main (launcher):            "2026-05-17 22:10:42 | LEVEL | logger | msg"

_TS_RE_COMMA = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})\s+\[(\w+)\]\s+([^:]+):\s*(.*)$")
_TS_RE_PIPE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+\|\s+(\w+)\s+\|\s+([^|]+?)\s+\|\s*(.*)$")


def parse_line(line: str) -> Optional[Tuple[str, str, str, str]]:
    """Return (timestamp_iso, level, logger, message) or None if unparseable.

    Timestamp is ISO-like with millisecond precision when available.
    """
    line = line.rstrip("\n")
    m = _TS_RE_COMMA.match(line)
    if m:
        date, ms, level, logger, msg = m.groups()
        return f"{date}.{ms}", level.strip(), logger.strip(), msg
    m = _TS_RE_PIPE.match(line)
    if m:
        date, level, logger, msg = m.groups()
        return f"{date}.000", level.strip(), logger.strip(), msg
    return None


# --- Per-source readers -----------------------------------------------------

@dataclass(order=True)
class Event:
    ts: str
    source: str = field(compare=False)
    level: str = field(compare=False)
    logger: str = field(compare=False)
    msg: str = field(compare=False)

    def render(self, max_msg: int = 400) -> str:
        msg = self.msg if len(self.msg) <= max_msg else self.msg[: max_msg - 1] + "…"
        return f"{self.ts}  [{self.source:<13}]  {self.logger:<28} | {msg}"


def grep_log(path: str, needles: List[str], source: str) -> List[Event]:
    """Stream a log file, keep only lines containing any needle."""
    if not os.path.isfile(path):
        return []
    out: List[Event] = []
    needles_b = needles  # plain string substring match is cheaper than regex here
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not any(n in line for n in needles_b):
                continue
            parsed = parse_line(line)
            if parsed is None:
                # keep unparseable lines anyway, attach a fallback timestamp
                out.append(Event("0000-00-00 00:00:00.000", source, "?", "?", line.strip()))
                continue
            ts, level, logger, msg = parsed
            out.append(Event(ts, source, level, logger, msg))
    return out


# --- Rollout debug lookup ---------------------------------------------------

def load_rollout_debug(run_dir: str) -> List[dict]:
    path = os.path.join(run_dir, "rollout_debug", "rollout_groups_debug_0.jsonl")
    if not os.path.isfile(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def find_context(debug: List[dict], the_id: str) -> Tuple[Optional[str], List[str], Optional[dict], Optional[dict]]:
    """Resolve `the_id` against the rollout-debug records.

    Returns (instance_id, [session_ids], group_record, sample_record_if_session).
    """
    for rec in debug:
        if rec.get("instance_id") == the_id:
            sids = [s.get("session_id") for s in rec.get("samples", []) if s.get("session_id")]
            return the_id, sids, rec, None
        for s in rec.get("samples", []):
            if s.get("session_id") == the_id:
                return rec.get("instance_id"), [the_id], rec, s
            if s.get("uid") == the_id:
                # match by uid: still drill into one sample
                return rec.get("instance_id"), [s.get("session_id")], rec, s
    return None, [], None, None


# --- Training record lookup (train_0.log / .slim.jsonl) ---------------------

def find_train_records(run_dir: str, instance_id: str) -> List[dict]:
    """Pull lightweight summaries of train records matching this instance."""
    candidates = [
        os.path.join(run_dir, "train_0.slim.jsonl"),
        os.path.join(run_dir, "train_0.log"),
    ]
    out = []
    for path in candidates:
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                # cheap pre-filter to avoid JSON-parsing every line
                if instance_id not in line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("index") != instance_id:
                    continue
                lm = d.get("loss_mask") or []
                tokens = d.get("tokens") or []
                rl = d.get("response_length") or 0
                out.append({
                    "prompt_uid": d.get("prompt"),
                    "reward": d.get("reward"),
                    "status": d.get("status"),
                    "msg_count": len(d.get("messages", [])),
                    "prompt_tokens": len(tokens) - rl,
                    "response_length": rl,
                    "loss_mask_ones": sum(1 for x in lm if x),
                    "finish_reason": (d.get("metadata") or {}).get("finish_reason"),
                })
        if out:
            # Stop after the first source that has data
            break
    return out


# --- Main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True, help="session_id, instance_id, or sample uid")
    ap.add_argument("--run-dir", default=None, help="run directory (default: $AIEVOBOX_RUN_DIR or .current_run)")
    ap.add_argument("--list-sessions", action="store_true",
                    help="when --id is an instance_id, just list the contained session_ids and exit")
    ap.add_argument("--max-events", type=int, default=2000, help="cap on rendered events (oldest kept)")
    ap.add_argument("--max-msg", type=int, default=400, help="truncate each event message to this length")
    args = ap.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    print(f"# run_dir: {run_dir}", file=sys.stderr)

    debug = load_rollout_debug(run_dir)
    inst_id, session_ids, group_rec, sample_rec = find_context(debug, args.id)

    if inst_id is None:
        print(f"# Could not resolve {args.id!r} in rollout_debug. Treating as raw needle.",
              file=sys.stderr)
        session_ids = [args.id]
        inst_id = None

    # If the user gave an instance_id and there are multiple sessions, list them.
    if args.list_sessions or (inst_id == args.id and len(session_ids) > 1 and not sample_rec):
        print(f"instance_id: {inst_id}")
        if group_rec:
            print(f"  group_size : {group_rec.get('group_size')}")
            rewards = group_rec.get("rewards") or []
            print(f"  rewards    : {rewards}")
            print(f"  status     : {group_rec.get('status')}  reason: {group_rec.get('reason')}")
        for i, sid in enumerate(session_ids):
            sample = next(
                (s for s in (group_rec or {}).get("samples", []) if s.get("session_id") == sid),
                None,
            )
            if sample:
                replies = len(sample.get("assistant_replies", []) or [])
                print(f"  [{i}] session_id={sid}  reward={sample.get('reward')}  "
                      f"replies={replies}  finish={sample.get('finish_reason')}")
            else:
                print(f"  [{i}] session_id={sid}")
        if args.list_sessions:
            return

    # Collect events from each log, filtered by all relevant IDs.
    needles = list({*session_ids, *(s for s in [inst_id] if s)})
    sources = {
        "buffer_server": os.path.join(run_dir, "buffer_server.log"),
        "main":          os.path.join(run_dir, "main.log"),
        "llm_proxy":     os.path.join(run_dir, "llm_proxy.log"),
    }
    events: List[Event] = []
    for src, path in sources.items():
        events.extend(grep_log(path, needles, src))

    events.sort()

    # --- Render -------------------------------------------------------------
    if inst_id:
        print(f"\n=== TRACE for instance={inst_id} | sessions={len(session_ids)} ===")
    else:
        print(f"\n=== TRACE for id={args.id} ===")

    if sample_rec:
        replies = sample_rec.get("assistant_replies") or []
        print(f"\n## Group sample summary (single rollout)")
        print(f"  session_id    : {sample_rec.get('session_id')}")
        print(f"  reward        : {sample_rec.get('reward')}")
        print(f"  finish_reason : {sample_rec.get('finish_reason')}")
        print(f"  weight_version: {sample_rec.get('weight_version')}")
        print(f"  uid           : {sample_rec.get('uid')}")
        print(f"  assistant_replies: {len(replies)} step(s)")
        for i, r in enumerate(replies):
            preview = (r if isinstance(r, str) else json.dumps(r))[:200]
            print(f"    step {i}: {preview!r}")

    print(f"\n## Events ({len(events)} matched, showing oldest {min(len(events), args.max_events)}):\n")
    for ev in events[: args.max_events]:
        print(ev.render(max_msg=args.max_msg))

    # --- Training records ---------------------------------------------------
    if inst_id:
        train = find_train_records(run_dir, inst_id)
        print(f"\n## Training records (train_0.log filtered by instance_id={inst_id}): {len(train)} hit(s)")
        if train:
            print(f"  {'prompt_uid':<36}  {'reward':>6}  {'msgs':>4}  {'p_tok':>5}  {'r_len':>5}  {'mask1s':>6}  finish")
            for t in train:
                print(f"  {t['prompt_uid']!s:<36}  {t['reward']!s:>6}  {t['msg_count']:>4}  "
                      f"{t['prompt_tokens']:>5}  {t['response_length']:>5}  {t['loss_mask_ones']:>6}  {t['finish_reason']}")


if __name__ == "__main__":
    main()
