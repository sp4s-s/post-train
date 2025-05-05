#!/usr/bin/env bash
set -euo pipefail


MODE=${1:-train}
shift || true

if [[ "$MODE" == "vllm" ]]; then
    echo "Starting vLLM server..."
    exec python3 -m vllm.entrypoints.api_server --model /workspace/src/checkpoints/policy --port 8000 --tensor-parallel-size 4
else
    echo "Starting training with args: $@"
    exec python3 -m torch.distributed.run \
        --nproc_per_node=${NGPU:-4} \
        --nnodes=${NNODES:-1} \
        --node_rank=${RANK:-0} \
        --master_addr=${MASTER_ADDR:-localhost} \
        --master_port=${MASTER_PORT:-6000} \
        src/megatron_verl.py "$@"
fi
