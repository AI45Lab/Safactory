# -------------------------------------------
# AIEvobox (rollout) Settings
# -------------------------------------------
export AIEVOBOX_ROOT=/mnt/shared-storage-user/chenxinquan/Safactory
export AIEVOBOX_MODE=remote
export STORAGE_TYPE=sqlite
export AIEVOBOX_DB_URL=sqlite:///mnt/shared-storage-user/evobox-share-gpfs2/chenxinquan/rl_db/osgym.db
export AIEVOBOX_MAX_STEPS=30
export AIEVOBOX_MESSAGE_CUT=1
export AIEVOBOX_ENV_CONFIG=/mnt/shared-storage-user/chenxinquan/Safactory/env/osgym/os_config.yaml
export AIEVOBOX_POOL_SIZE=128
export AIEVOBOC_MULTIPLIER=1.0
export AIEVOBOX_ENV_TRANSPORT=http
export AIEVOBOX_LLM_MAX_CONCURRENCY=$AIEVOBOX_POOL_SIZE
export AIEVOBOX_LLM_PROXY_WORKERS=128
export AIEVOBOX_LLM_STARTUP_JITTER_S=0
export AIEVOBOX_TRAININFO_WORKERS=32
export AIEVOBOX_SQLITE_BULK_INSERT_BATCH_SIZE=128
export AIEVOBOX_SQLITE_BULK_INSERT_PAUSE_S=0.01

# -------------------------------------------
# RL Settings
# -------------------------------------------
export RL_GROUP_SIZE=8
export RL_EPOCH=10
export RL_OFF_BY_N=0
# DAPO filter: when true, drops groups where all samples have the same reward.
# Disabled for osgym cold-start: with sparse terminal rewards on 15-20 step
# trajectories, almost every group is all-zero early in training, so the
# filter discards ~96% of rollouts and starves the learner.
# export DAPO_filter="${DAPO_filter:-false}"

# no use, will be removed
export RL_MODEL=model
export RL_API_KEY=openai_api_key


# -------------------------------------------
# Buffer Server Settings (run_buffer_server.sh)
# -------------------------------------------
# Buffer Server 由 run_buffer_server.sh 启动，负责管理 rollout 数据并拉起 AIEvoBox launcher。
# HOST 是其他服务连接 Buffer Server 用的地址（服务本身始终监听 0.0.0.0）。
# Slime Generator 通过此地址调用 /get_rollout_data 和 /start_rollout。
# 如果 Buffer Server 和 Slime Generator 运行在不同机器上，改为 Buffer Server 所在机器的 IP。
export BUFFER_SERVER_HOST=127.0.0.1
export BUFFER_SERVER_PORT=18889

# -------------------------------------------
# LLM Proxy Settings (hosted in-process by Slime Generator)
# -------------------------------------------
# LLM Proxy 由 Slime Generator (run_slime_generator*.sh) 在进程内启动，提供 /v1 chat completions 接口。
# HOST 是其他服务连接 LLM Proxy 用的地址（服务本身始终监听 0.0.0.0）。
# AIEvoBox launcher（由 Buffer Server 拉起）通过此地址调用 LLM。
# 如果 Buffer Server 和 Slime Generator 运行在不同机器上，改为 Slime Generator 所在机器的 IP。
export LLM_PROXY_HOST=127.0.0.1
export LLM_PROXY_PORT=18890
export LLM_MAX_LENGTH=28672
export LLM_TEMPERATURE=1.0

# -------------------------------------------
# Slime Training Settings (reference RL values)
# -------------------------------------------
export SLIME_ROLLBUF_RESTART_TRAINING=True
export SLIME_N_SAMPLES_PER_PROMPT=$RL_GROUP_SIZE
export SLIME_GLOBAL_BATCH_SIZE=256
export SLIME_ROLLOUT_BATCH_SIZE=$((SLIME_GLOBAL_BATCH_SIZE / RL_GROUP_SIZE))

# -------------------------------------------
# Debug (debugpy / VS Code)
# -------------------------------------------
# Set DEBUGPY_<NAME>=1 to start a debugpy listener inside the matching process.
# Ports default to those used by .vscode/launch.json. Override with DEBUGPY_<NAME>_PORT.
# Set DEBUGPY_WAIT_FOR_CLIENT=1 to block startup until VS Code attaches —
# useful when you need to break inside early init (e.g. tokenizer load).
#
# slime_rollout is a single python process that also hosts the in-process
# llm_proxy thread, so one attach covers breakpoints in both slime_generator.py
# and llm_proxy.py.
export DEBUGPY_BUFFER_SERVER="${DEBUGPY_BUFFER_SERVER:-0}"
export DEBUGPY_BUFFER_SERVER_PORT="${DEBUGPY_BUFFER_SERVER_PORT:-5678}"
export DEBUGPY_SLIME_ROLLOUT="${DEBUGPY_SLIME_ROLLOUT:-0}"
export DEBUGPY_SLIME_ROLLOUT_PORT="${DEBUGPY_SLIME_ROLLOUT_PORT:-5681}"
export DEBUGPY_WAIT_FOR_CLIENT="${DEBUGPY_WAIT_FOR_CLIENT:-0}"