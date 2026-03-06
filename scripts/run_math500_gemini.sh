#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate GMemory

cd "$(dirname "$0")/.." || exit 1

export GOOGLE_API_KEY="${GOOGLE_API_KEY:-your-google-api-key-here}"

python3 tasks/run.py \
    --task math \
    --reasoning io \
    --mas_memory skill-rl \
    --mas_type math-gcn \
    --model gemini-2.5-flash \
    --max_trials 20 \
    --successful_topk 3 \
    --failed_topk 0 \
    --insights_topk 2
