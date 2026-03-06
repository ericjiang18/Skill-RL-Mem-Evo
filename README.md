# Skill-Conditioned RL with Evolving Memory for LLM Agents

A **test-time learning** framework for LLM-based autonomous agents. Instead of retraining, agents improve across tasks through three progressively deeper layers of memory that accumulate experience, distill insights, and discover reusable skills.

## Three-Layer Learning Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Layer 1: Experience Memory (Immediate)                         │
│  ├── Successful trajectory storage + vector retrieval           │
│  │   (GMemory Task Layer)                                       │
│  ├── ExpRAG: lightweight experience retrieval with              │
│  │   embedding similarity + goal-type boosting                  │
│  └── Effect: Task N succeeds → Task N+1 can directly            │
│      reference the execution pattern                            │
│                                                                 │
│  Layer 2: Insight Evolution (Mid-term Accumulation)             │
│  ├── LLM Refine: one post-task LLM call produces               │
│  │   structured insight per task                                │
│  ├── Insight scoring: success +0.5, failure -1                  │
│  │   (auto-prune low-scoring rules)                             │
│  ├── Periodic LLM merge/prune to consolidate insights           │
│  └── Effect: Distill general rules from many experiences        │
│                                                                 │
│  Layer 3: Skill Emergence + RL Selection (Long-term)            │
│  ├── Skill Miner: cluster successful trajectories →             │
│  │   extract reusable parameterized procedures                  │
│  ├── Skill-Conditioned RL: Q(goal_type, skill_id)               │
│  │   with UCB exploration to select the best skill              │
│  ├── LLM-driven skill step evolution on failure                 │
│  └── Effect: Automatically discover and optimize                │
│      procedural knowledge ("how to do it")                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### How the Layers Interact

```
Task Start
  │
  ▼
┌──────────────────────────────────────┐
│  Retrieval Phase (one-time)          │
│  Layer 1 → past trajectories         │
│  Layer 2 → insight rules             │
│  Layer 3 → skills + RL skill select  │
│         ↓ all injected into prompt   │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Execution: Think-Act Loop           │
│  • System prompt set once            │
│  • Sliding window (last 7 steps)     │
│  • Skill-guided action hints         │
│  • Proactive loop detection          │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Post-Task Learning                  │
│  Layer 1 → store trajectory + ExpRAG │
│  Layer 2 → LLM Refine → insight     │
│  Layer 3 → RL Q-update + skill mine  │
└──────────────────────────────────────┘
```

## Key Components

| Component | Layer | Description |
|-----------|-------|-------------|
| **G-Memory Task Layer** | 1 | Graph-based trajectory storage with k-hop retrieval and vector similarity |
| **ExpRAG** | 1 | Lightweight experience store with embedding retrieval, goal-type boosting, and success weighting |
| **LLM Refine** | 2 | Post-task LLM call that extracts INSIGHT, SKILL_UPDATE, and AVOID rules |
| **Insight Manager** | 2 | Scored rule system with periodic LLM-driven merge/prune |
| **Skill Miner** | 3 | Clusters successful trajectories → LLM synthesizes parameterized skills with goal patterns |
| **Skill-Conditioned RL** | 3 | Tabular Q(goal_type, skill_id) with UCB exploration bonus for skill selection |
| **Skill MAS** | Exec | Think-Act-Refine loop with sliding window prompts and loop detection |

## Supported Environments

| Environment | Benchmark | Task Types |
|------------|-----------|------------|
| **ALFWorld** | Household text simulation | put, clean, heat, cool, examine, puttwo |
| **PDDL** | Classical planning | blocksworld, barman, gripper, tyreworld |
| **Math** | Mathematical reasoning | MATH-500, AIME |
| **SciWorld** | Science experiments | Various science tasks |

## Installation

```bash
git clone https://github.com/ericjiang18/Goal-RL-Mem-Evo.git
cd Goal-RL-Mem-Evo
pip install -r requirements.txt
```

### Environment-specific setup

```bash
# ALFWorld
pip install alfworld
export ALFWORLD_DATA=~/.cache/alfworld
alfworld-download

# SciWorld
pip install scienceworld
```

## Configuration

```bash
# OpenAI
export OPENAI_API_KEY="your-key"
export OPENAI_API_BASE="https://api.openai.com/v1"

# Gemini (optional)
export GOOGLE_API_KEY="your-key"
```

## Quick Start

```bash
# ALFWorld
bash scripts/run_alfworld_gpt4omini.sh      # GPT-4o-mini
bash scripts/run_alfworld_gemini.sh          # Gemini 2.5 Flash

# PDDL Planning
bash scripts/run_pddl_gpt4omini.sh
bash scripts/run_pddl_gemini.sh

# Math Reasoning
bash scripts/run_math500_gpt4omini.sh
bash scripts/run_math500_gemini.sh

# SciWorld
bash scripts/run_sciworld_gpt4omini.sh
bash scripts/run_sciworld_gemini.sh
```

### Command-line Options

```bash
python3 tasks/run.py \
    --task alfworld \           # alfworld | pddl | math | sciworld
    --mas_type skill-mas \      # Skill MAS execution loop
    --mas_memory skill-rl \     # Skill-Conditioned RL + ExpRAG + LLM Refine
    --model gpt-4o-mini \       # gpt-4o-mini | gemini-2.5-flash | etc.
    --reasoning io \            # Reasoning module
    --max_trials 30 \           # Max steps per task
    --successful_topk 1 \       # Trajectories to retrieve from Layer 1
    --insights_topk 3           # Insights to retrieve from Layer 2
```

## Token Efficiency

Compared to standard ReAct agents that include full history in every prompt:

- **System prompt** (few-shots, insights, skills): set **once** per task
- **Step prompt**: sliding window of last 7 action-observation pairs
- **Think isolation**: think actions excluded from sliding window
- **LLM Refine**: single compact call (~300 tokens) per task for Layer 2 learning

## Project Structure

```
├── mas/
│   ├── llm.py                      # LLM backends (OpenAI, Gemini, Qwen)
│   ├── module_map.py               # Module registry
│   └── memory/mas_memory/
│       ├── GMemory.py              # Layer 1: Graph memory + Layer 2: Insights
│       ├── gmemory_plus.py         # G-Memory++ (goal module + skill integration)
│       ├── skill_memory.py         # Integration hub: RL + ExpRAG + LLM Refine
│       ├── skill_rl.py             # Layer 3: Skill-Conditioned RL (Q-table)
│       ├── skill_miner.py          # Layer 3: Skill extraction from clusters
│       └── goal_module.py          # Goal parsing (StructuredGoal)
├── tasks/
│   ├── run.py                      # Main entry point
│   ├── mas_workflow/
│   │   ├── skill_mas/skill_mas.py  # Execution loop (Think-Act-Refine)
│   │   └── format.py              # Prompt construction
│   ├── envs/                       # Environment implementations
│   └── prompts/                    # Task-specific prompts
└── scripts/                        # Run scripts for each benchmark
```

## License

MIT
