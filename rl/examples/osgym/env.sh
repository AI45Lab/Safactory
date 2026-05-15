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