"""Tiny debugpy launcher used by the OSGym RL entrypoints.

Each entrypoint calls ``start_debugpy(name, default_port)`` early in startup.
The listener is gated by the ``DEBUGPY_<NAME>=1`` env var so production runs
are unaffected. ``DEBUGPY_<NAME>_PORT`` overrides the port and
``DEBUGPY_WAIT_FOR_CLIENT=1`` blocks until VS Code attaches.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_STARTED: set[str] = set()


def _truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in ("1", "true", "yes", "on")


def start_debugpy(name: str, default_port: int) -> None:
    key = name.upper()
    if not _truthy(os.getenv(f"DEBUGPY_{key}")):
        return
    if name in _STARTED:
        return

    try:
        import debugpy  # type: ignore
    except ImportError:
        logger.warning(
            "DEBUGPY_%s=1 but the debugpy package is not installed; "
            "run `pip install debugpy` to enable.", key,
        )
        return

    port = int(os.getenv(f"DEBUGPY_{key}_PORT", str(default_port)))
    try:
        debugpy.listen(("0.0.0.0", port))
    except Exception as exc:
        logger.warning("debugpy.listen failed for %s on port %d: %s", name, port, exc)
        return
    _STARTED.add(name)

    banner = f"[debugpy] {name} listening on 0.0.0.0:{port} (pid={os.getpid()})"
    print(banner, flush=True)
    logger.info(banner)

    if _truthy(os.getenv("DEBUGPY_WAIT_FOR_CLIENT")):
        print(
            f"[debugpy] {name} waiting for VS Code to attach on port {port} ...",
            flush=True,
        )
        debugpy.wait_for_client()
        print(f"[debugpy] {name} client attached", flush=True)
