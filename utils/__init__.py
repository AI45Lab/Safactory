from .actor_pool_runtime import ActorPoolRuntimeState, discover_ready_actor_from_snapshot

# Re-export from rl.utils for backward compatibility
try:
    from rl.utils import (
        get_env,
        AggType,
        MetricsRecorder,
        RolloutGroupDebugLogger,
        STATUS_KEPT,
        STATUS_DAPO_ALL_SAME,
        STATUS_WEIGHT_VERSION_SKEW,
        STATUS_DROP_UNMATCHED_TRAJECTORY,
        STATUS_DROP_ASSEMBLY_ERROR,
        get_or_create_run_dir,
        setup_process_logging,
        start_debugpy,
    )
except ImportError:
    # Fallback: try direct import when rl is not in PYTHONPATH
    import sys
    import os
    _RL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rl")
    if _RL_DIR not in sys.path:
        sys.path.insert(0, _RL_DIR)
    from utils import (
        get_env,
        AggType,
        MetricsRecorder,
        RolloutGroupDebugLogger,
        STATUS_KEPT,
        STATUS_DAPO_ALL_SAME,
        STATUS_WEIGHT_VERSION_SKEW,
        STATUS_DROP_UNMATCHED_TRAJECTORY,
        STATUS_DROP_ASSEMBLY_ERROR,
        get_or_create_run_dir,
        setup_process_logging,
        start_debugpy,
    )

__all__ = [
    "ActorPoolRuntimeState",
    "discover_ready_actor_from_snapshot",
    "get_env",
    "AggType",
    "MetricsRecorder",
    "RolloutGroupDebugLogger",
    "STATUS_KEPT",
    "STATUS_DAPO_ALL_SAME",
    "STATUS_WEIGHT_VERSION_SKEW",
    "STATUS_DROP_UNMATCHED_TRAJECTORY",
    "STATUS_DROP_ASSEMBLY_ERROR",
    "get_or_create_run_dir",
    "setup_process_logging",
    "start_debugpy",
]
