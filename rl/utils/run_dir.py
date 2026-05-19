"""Shared per-run log directory.

`buffer_server`, `llm_proxy`, and `slime_generator` are started by separate
shell scripts (sometimes on different nodes), but they belong to the same
training session. They agree on a single ``logs/<timestamp>[-<run_name>]/``
folder via a marker file ``logs/.current_run`` and the ``AIEVOBOX_RUN_DIR``
env var.

Order of precedence on resolution:
  1. ``AIEVOBOX_RUN_DIR`` env var — set by the session leader and inherited
     by subprocesses; also lets the user pin a folder explicitly. Honored
     for both reader and leader (``create_new`` does not override an
     explicit user pin).
  2. Marker file at ``<logs_root>/.current_run`` (one line: absolute path).
  3. Fallback: create a new ``logs/<timestamp>[-<run_name>]/`` folder.

The leader (``create_new=True``) creates a fresh dir unless the env var is
already set externally. Readers (``create_new=False``) discover the leader's
dir; if no leader has registered yet they fall back to a fresh dir so
startup never crashes — see ``allow_fallback`` if you want to opt out.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

_RUN_DIR_ENV = "AIEVOBOX_RUN_DIR"
_MARKER_FILENAME = ".current_run"
_RUN_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_run_name(run_name: Optional[str]) -> str:
    if not run_name:
        return ""
    cleaned = _RUN_NAME_RE.sub("-", run_name.strip())
    return cleaned.strip("._-")


def _build_timestamp_dir(logs_root: str, run_name: Optional[str] = None) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = _sanitize_run_name(run_name)
    base = f"{stamp}-{suffix}" if suffix else stamp
    candidate = os.path.join(logs_root, base)
    # Same-second collision (re-launching the leader): append a short
    # disambiguator instead of silently sharing the directory.
    if os.path.exists(candidate):
        for n in range(1, 100):
            alt = os.path.join(logs_root, f"{base}-{n:02d}")
            if not os.path.exists(alt):
                return alt
    return candidate


def _read_marker(logs_root: str) -> Optional[str]:
    marker_path = os.path.join(logs_root, _MARKER_FILENAME)
    try:
        with open(marker_path, "r", encoding="utf-8") as f:
            value = f.read().strip()
    except OSError:
        return None
    if value and os.path.isdir(value):
        return value
    return None


def _write_marker(logs_root: str, run_dir: str) -> None:
    marker_path = os.path.join(logs_root, _MARKER_FILENAME)
    tmp_path = f"{marker_path}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(run_dir + "\n")
        os.replace(tmp_path, marker_path)
    except OSError as err:
        logger.warning("failed to update run-dir marker %s: %s", marker_path, err)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def get_or_create_run_dir(
    logs_root: str,
    *,
    create_new: bool = False,
    run_name: Optional[str] = None,
) -> str:
    """Return the per-run log directory under ``logs_root``.

    Args:
        logs_root: parent directory (e.g. ``$AIEVOBOX_ROOT/logs``).
        create_new: if True and ``AIEVOBOX_RUN_DIR`` is not set externally,
            create a fresh ``logs/<timestamp>[-<run_name>]/`` and overwrite
            the marker file. Use from the *session leader* (buffer_server).
            Reader processes (llm_proxy, slime_generator) should pass False
            so they discover the leader's directory.
        run_name: optional human-readable suffix appended to the timestamp
            ("``20260518-093000-osgym``"). Ignored when an existing run dir
            is reused.

    Side effects:
        - Sets ``AIEVOBOX_RUN_DIR`` in the current process environment so
          subprocesses inherit it.
        - When a new directory is created, writes
          ``<logs_root>/.current_run``.
    """
    os.makedirs(logs_root, exist_ok=True)

    # 1. Always honor an explicit pin. Even leaders should not clobber a
    # user-set AIEVOBOX_RUN_DIR — that's the documented escape hatch.
    env_val = os.environ.get(_RUN_DIR_ENV)
    if env_val:
        os.makedirs(env_val, exist_ok=True)
        if create_new:
            # Re-register the marker so readers landing later see the same
            # dir even if they read the marker before this process exports
            # env vars to them.
            _write_marker(logs_root, env_val)
        return env_val

    if create_new:
        run_dir = _build_timestamp_dir(logs_root, run_name)
        os.makedirs(run_dir, exist_ok=True)
        os.environ[_RUN_DIR_ENV] = run_dir
        _write_marker(logs_root, run_dir)
        return run_dir

    # 2. Reader path: prefer the leader's marker.
    marker_val = _read_marker(logs_root)
    if marker_val:
        os.environ[_RUN_DIR_ENV] = marker_val
        return marker_val

    # 3. No leader has registered yet — fall back to a fresh dir so we
    # never crash on startup. This may diverge from the leader by a few
    # seconds; if you see unexpected splitting, start buffer_server *first*.
    logger.warning(
        "No leader run dir found under %s (no AIEVOBOX_RUN_DIR, no marker); "
        "falling back to a fresh dir. Start buffer_server before reader "
        "processes to avoid log splitting.",
        logs_root,
    )
    run_dir = _build_timestamp_dir(logs_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.environ[_RUN_DIR_ENV] = run_dir
    _write_marker(logs_root, run_dir)
    return run_dir
