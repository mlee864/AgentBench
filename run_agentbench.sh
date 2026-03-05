#!/bin/bash
#SBATCH --job-name=agentbench-vllm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

unset SSL_CERT_FILE
unset SSL_CERT_DIR

export HF_HOME=/net/projects/notebook/mlee864/hf_cache
export HF_HUB_CACHE=/net/projects/notebook/mlee864/hf_cache/hub
export TRANSFORMERS_CACHE=/net/projects/notebook/mlee864/hf_cache/transformers
export HF_XET_CACHE=/net/projects/notebook/mlee864/hf_cache/xet

# vLLM
export OPENAI_API_KEY=""
export WOLFRAM_ALPHA_APPID=""
PORT=8001

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --host 127.0.0.1 \
  --port $PORT \
  --api-key $OPENAI_API_KEY \
  --dtype half \
  --gpu-memory-utilization 0.7 \
  --max-num-seqs 1 \
  > vllm_server.log 2>&1 &



VLLM_PID=$!

for i in {1..60}; do
  if curl -s http://127.0.0.1:$PORT/v1/models -H "Authorization: Bearer $OPENAI_API_KEY" >/dev/null; then
    echo "vLLM is up."
    break
  fi
  sleep 2
done

# AgentBench 
python agent_bench.py --agent react_hotpotqa --config config.yaml > react_hotpotqa.log
python agent_bench.py --agent react_math --config config.yaml > react_math.log
python agent_bench.py --agent react_humaneval --config config.yaml > react_code.log

kill $VLLM_PID