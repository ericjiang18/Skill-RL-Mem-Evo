#!/bin/bash

# Explicitly set the path to where AgentMemory-new2-zyk is located
# Adjust this if you are running from a different location
PROJECT_ROOT="/local3/ericjiang/AgentMemory-new2-zyk"
cd "$PROJECT_ROOT"

# OpenAI API config (Using environment variables or defaults)
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.openai.com/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-your-openai-api-key-here}"

python3 tasks/run.py \
    --task math \
    --reasoning io \
    --mas_memory skill-rl \
    --mas_type math-gcn \
    --model gpt-4o-mini \
    --max_trials 20 \
    --successful_topk 3 \
    --failed_topk 0 \
    --insights_topk 2
