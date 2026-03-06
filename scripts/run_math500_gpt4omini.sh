#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate GMemory

cd "$(dirname "$0")/.." || exit 1

export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.openai.com/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-your-openai-api-key-here}"

python3 tasks/run.py \
    --task math \
    --reasoning io \
    --mas_memory skill-rl \
    --mas_type skill-mas \
    --model gpt-4o-mini \
    --max_trials 20 \
    --successful_topk 3 \
    --insights_topk 2
