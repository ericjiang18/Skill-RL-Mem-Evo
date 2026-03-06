#!/bin/bash

# Ensure we're in the repository root
cd "$(dirname "$0")/.." || exit 1

if [ -z "$1" ]; then
    echo "Usage: $0 [task_name | all]"
    echo "Example: $0 pddl"
    echo "         $0 alfworld"
    echo "         $0 all       (to clear all memories)"
    exit 1
fi

if [ "$1" == "all" ]; then
    echo "Cleaning all memory in .db/ ..."
    rm -rf .db/*
    echo "All memories have been deleted."
else
    TARGET_TASK="$1"
    echo "Cleaning memory for task: $TARGET_TASK across all models..."
    
    FOUND=0
    # Find and remove task directories under each model in .db/
    for model_dir in .db/*; do
        if [ -d "$model_dir/$TARGET_TASK" ]; then
            rm -rf "$model_dir/$TARGET_TASK"
            echo "Removed $model_dir/$TARGET_TASK"
            FOUND=1
        fi
    done
    
    if [ $FOUND -eq 0 ]; then
        echo "No memories found for task: $TARGET_TASK"
    else
        echo "Memory for $TARGET_TASK cleaned successfully."
    fi
fi
