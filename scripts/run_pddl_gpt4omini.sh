#!/bin/bash

# Ensure we're in the repository root
cd "$(dirname "$0")/.." || exit 1

echo "Starting PDDL benchmarking with Goal-GCN MAS and Goal-RL Memory (model: gpt-4o-mini)..."

python3 tasks/run.py \
    --task pddl \
    --reasoning io \
    --mas_memory skill-rl \
    --mas_type skill-mas \
    --model gpt-4o-mini \
    --max_trials 30 \
    --successful_topk 1 \
    --insights_topk 3

echo "Finished benchmarking."
