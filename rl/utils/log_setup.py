"""Per-process root-logger setup shared by RL workers.

``buffer_server``, ``llm_proxy``, and ``slime_generator`` each run in a
separate Python process and historically configured logging by hand —
three different formats, no handler on the root logger. As a result,
``logging.getLogger(__name__)`` calls in helper modules (e.g.
``trajectory_mask_builder``) wrote nowhere.

This module provides a single ``setup_process_logging`` entry point that:

  * resolves the shared run dir via :mod:`run_dir`;
  * attaches a rotating file handler at the *root* logger so every named
    logger in the process is captured;
  * attaches a console handler (optional) with sensible defaults;
  * uses the same ``"%(asctime)s | %(levelname)s | %(name)s | %(message)s"``
    format as the launcher, so a single regex parses every log file.

Idempotent: calling twice from the same process replaces the previous
handlers cleanly.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional, Sequence

from .run_dir import get_or_create_run_dir

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_INSTALLED_TAG_ATTR = "_aievobox_process_log_tag"

# Dependencies that flood the file log unless pinned to WARNING.
_DEFAULT_QUIET_LOGGERS = (
    "httpx",
    "urllib3",
    "uvicorn.access",
    "asyncio",
)


def _build_formatter() -> logging.Formatter:
    return logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)


def _clear_existing_handlers(root: logging.Logger) -> None:
    """Remove any handlers we installed previously in this process.

    Other libraries' handlers are left alone — we only drop ones tagged
    with our marker attribute, so a second ``setup_process_logging`` call
    cleanly replaces the previous setup (e.g. an in-process import doing
    its own setup, then the host process re-configuring later).
    """
    for handler in list(root.handlers):
        if getattr(handler, _INSTALLED_TAG_ATTR, None) is not None:
            root.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass


def _parse_level(level: str, default: int = logging.INFO) -> int:
    value = getattr(logging, str(level).upper(), None)
    return value if isinstance(value, int) else default


def setup_process_logging(
    process_name: str,
    *,
    logs_root: Optional[str] = None,
    create_new_run_dir: bool = False,
    run_name: Optional[str] = None,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    enable_console: bool = True,
    quiet_loggers: Sequence[str] = _DEFAULT_QUIET_LOGGERS,
    file_max_bytes: int = 50 * 1024 * 1024,
    file_backup_count: int = 5,
) -> str:
    """Install a per-process file/console handler pair on the root logger.

    Args:
        process_name: identifies the file (``<process_name>.log``) and tags
            installed handlers so re-init doesn't duplicate them.
        logs_root: parent ``logs/`` directory; defaults to
            ``$AIEVOBOX_ROOT/logs``.
        create_new_run_dir: pass ``True`` from the session leader.
        run_name: optional suffix on a freshly created run dir.
        console_level / file_level: per-handler levels (root stays DEBUG so
            the file handler can capture everything).
        enable_console: turn off for daemon processes that already mirror
            stdout to a parent process.
        quiet_loggers: logger names pinned at WARNING so chatty deps don't
            spam the run log.

    Returns:
        Absolute path of the run directory used.
    """
    if logs_root is None:
        aievobox_root = os.environ.get("AIEVOBOX_ROOT") or os.getcwd()
        logs_root = os.path.join(aievobox_root, "logs")

    run_dir = get_or_create_run_dir(
        logs_root, create_new=create_new_run_dir, run_name=run_name
    )
    log_path = os.path.join(run_dir, f"{process_name}.log")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    _clear_existing_handlers(root)

    formatter = _build_formatter()

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=file_max_bytes,
        backupCount=file_backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(_parse_level(file_level, logging.DEBUG))
    file_handler.setFormatter(formatter)
    setattr(file_handler, _INSTALLED_TAG_ATTR, process_name)
    root.addHandler(file_handler)

    if enable_console:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(_parse_level(console_level, logging.INFO))
        console.setFormatter(formatter)
        setattr(console, _INSTALLED_TAG_ATTR, process_name)
        root.addHandler(console)

    for name in quiet_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.captureWarnings(True)

    root.info(
        "logging initialized: process=%s run_dir=%s log=%s "
        "console_level=%s file_level=%s pid=%d",
        process_name,
        run_dir,
        log_path,
        console_level.upper(),
        file_level.upper(),
        os.getpid(),
    )
    return run_dir
