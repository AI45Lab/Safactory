#!/usr/bin/env bash

# Increase file descriptor limit for high concurrency
ulimit -n 65536 2>/dev/null || echo "Warning: Could not set ulimit -n 65536 (current: $(ulimit -n))"

# Kill existing processes
pkill -9 sglang || true
sleep 2
ray stop --force || true
pkill -9 ray || true
if [[ "${KILL_PYTHON_ON_START:-0}" == "1" ]]; then
   pkill -9 python || true
   sleep 2
else
   echo "Skipping pkill -9 python. Set KILL_PYTHON_ON_START=1 to restore old cleanup behavior."
fi

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

source "${SCRIPT_DIR}/env.sh"

# Construct URLs from host and port
ROLLOUT_BUFFER_URL="http://${BUFFER_SERVER_HOST}:${BUFFER_SERVER_PORT}"
LLM_PROXY_URL="http://${LLM_PROXY_HOST}:${LLM_PROXY_PORT}"

export WANDB_MODE=offline
export PYTHONBUFFERED=16
NUM_GPUS=${NUM_GPUS:-4}

SLIME_HOME=${SLIME_HOME:-/root/slime}
HF_CKPT_DIR="/mnt/shared-storage-user/evobox-share/hf-hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
SAVE_DIR="/mnt/shared-storage-user/evobox-share/chenxinquan/slime_checkpoints/Qwen3-VL-8B-Instruct_megatron"
MODEL_ARGS_ROTARY_BASE=5000000 source "${SLIME_HOME}/scripts/models/qwen3-8B.sh"

CKPT_ARGS=(
   --hf-checkpoint ${HF_CKPT_DIR}
   --load ${HF_CKPT_DIR}
   --save ${SAVE_DIR}
   --save-interval 20
)

# 实际上这里很多值都没有使用
ROLLOUT_ARGS=(
   --rollout-function-path rl.slime_generator.generate_rollout
   --rollout-buffer-url ${ROLLOUT_BUFFER_URL}
   --disable-rollout-global-dataset
   --num-rollout 300
   --rollout-batch-size ${SLIME_ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt ${SLIME_N_SAMPLES_PER_PROMPT}
   --rollout-max-response-len ${LLM_MAX_LENGTH}
   --rollout-temperature ${LLM_TEMPERATURE}
   --global-batch-size ${SLIME_GLOBAL_BATCH_SIZE}
   --loss-mask-type qwen
)

MEGATRON_ARGS=(
   --train-backend megatron
   --megatron-to-hf-mode bridge
   --tensor-model-parallel-size 2
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   # --recompute-granularity full
   # --recompute-method uniform
   # --recompute-num-layers 1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

TRAIN_ARGS=(
   --use-dynamic-batch-size
   --max-tokens-per-gpu 40000
   --calculate-per-token-loss
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.2
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
    --use-wandb
    --wandb-project slime
   #  --wandb-team aievobox
    --wandb-group slime
    --wandb-dir /mnt/shared-storage-user/evobox-share/chenxinquan/wandb_logs
    # --wandb-always-use-train-step
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.6
   --sglang-attention-backend fa3
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   --sglang-log-level info
   --sglang-log-level-http error
)

# Start Ray
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats

export SGLANG_LOGGING_CONFIG_PATH=${SGLANG_LOGGING_CONFIG_PATH:-"/mnt/shared-storage-user/chenxinquan/Safactory/rl/sglang_logging.json"}

RUNTIME_ENV_JSON="{\
  \"env_vars\": {\
    \"PYTHONPATH\": \"${SLIME_HOME}:${AIEVOBOX_ROOT}/rl:${AIEVOBOX_ROOT}:/root/Megatron-LM\",\
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",\
    \"LLM_PROXY_URL\": \"${LLM_PROXY_URL}\",\
    \"ROLLOUT_BUFFER_URL\": \"${ROLLOUT_BUFFER_URL}\",\
    \"SLIME_OFF_BY_N\": \"${SLIME_OFF_BY_N:-0}\"\
  }\
}"

TRAIN_ENTRYPOINT=(python3)

if [[ "${DEBUG_SLIME_TRAIN:-0}" == "1" ]]; then
   TRAIN_ENTRYPOINT=(
      python3 -m debugpy
      --listen "0.0.0.0:${DEBUGPY_TRAIN_PORT:-5679}"
      --configure-subProcess True
   )
   if [[ "${DEBUGPY_WAIT_FOR_CLIENT:-1}" == "1" ]]; then
      TRAIN_ENTRYPOINT+=(--wait-for-client)
   fi
fi

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- "${TRAIN_ENTRYPOINT[@]}" ${SLIME_HOME}/train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${MEGATRON_ARGS[@]} \
   ${CKPT_ARGS[@]} \
    ${ROLLOUT_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${GRPO_ARGS[@]} \
    ${WANDB_ARGS[@]} \
    ${TRAIN_ARGS[@]} \
    ${SGLANG_ARGS[@]}
