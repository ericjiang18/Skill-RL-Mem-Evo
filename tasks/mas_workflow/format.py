"""
Prompt formatting for MAS workflow.

Two-tier prompt structure:
  1. System context — set once at task start (few_shots, memory, insights, skills)
  2. Step prompt   — updated each step (sliding window + current observation)
"""

# ======================== System Context (set once) ========================

_system_context_template = """\
## Successful Examples (Reference Cases)
Below are some examples of similar tasks that were successfully completed.
Use these as references to guide your thinking and approach:

{few_shots}
---

## Your Own Past Successes (Execution Patterns)
Past successful execution processes on similar tasks.
Pay attention to step-by-step procedures and strategies:

{memory_few_shots}
---

## Key Insights from Related Tasks
{insights}
---
{skills_section}\
"""

# ======================== Step Prompt (per step) ========================

_step_prompt_template = """\
## Task Environment
{initial_observation}

## Current Task
{task_goal}

## Recent History (last {window_size} steps)
{sliding_window}

## Current Observation
{current_observation}
{extra_context}"""


# ======================== Public API ========================

def build_system_context(
    few_shots: list[str],
    memory_few_shots: list[str],
    insights: list[str],
    skills_text: str = "",
) -> str:
    """Build the system-level context string. Called once per task."""
    insights_text = '\n'.join(
        f'{i}. {r}' for i, r in enumerate(insights, 1)
    ) if insights else "(No insights available yet)"

    memory_text = '\n\n'.join(
        f"Task {i+1}:\n{shot}" for i, shot in enumerate(memory_few_shots)
    ) if memory_few_shots else "(No past successes retrieved)"

    skills_section = ""
    if skills_text:
        skills_section = f"\n{skills_text}\n---\n"

    return _system_context_template.format(
        few_shots='\n'.join(few_shots) if few_shots else "(No reference examples)",
        memory_few_shots=memory_text,
        insights=insights_text,
        skills_section=skills_section,
    )


def build_step_prompt(
    task_goal: str,
    action_obs_window: list[tuple[str, str]],
    current_observation: str,
    initial_observation: str = "",
    loop_warning: str = "",
    policy_hint: str = "",
    think_buffer: str = "",
) -> str:
    """Build the per-step user prompt with a sliding window of recent history."""
    window_lines = []
    for i, (act, obs) in enumerate(action_obs_window, 1):
        window_lines.append(f"[Step {i}] Action: {act}")
        obs_short = obs[:500] if len(obs) > 500 else obs
        window_lines.append(f"  Observation: {obs_short}")

    sliding_text = '\n'.join(window_lines) if window_lines else "(Task just started — no history yet)"

    extra_parts = []
    if think_buffer:
        extra_parts.append(f"\n## Your Recent Thoughts\n{think_buffer}")
    if policy_hint:
        extra_parts.append(f"\n{policy_hint}")
    if loop_warning:
        extra_parts.append(f"\n{loop_warning}")

    return _step_prompt_template.format(
        task_goal=task_goal,
        initial_observation=initial_observation,
        window_size=len(action_obs_window),
        sliding_window=sliding_text,
        current_observation=current_observation,
        extra_context=''.join(extra_parts),
    )


# ======================== Legacy helpers (backward compat) ========================

task_solve_with_insights = """
## Successful Examples (Reference Cases)
Below are some examples of similar tasks that were successfully completed.  
Please use these as references to guide your thinking and approach to the current task:

{few_shots}
---

## Your Own Past Successes (Execution Patterns)
Here are examples of successful execution processes you've previously used on similar tasks.  
Pay special attention to the step-by-step procedures and strategies, especially when encountering obstacles:

{memory_few_shots}
---

## Key Insights from Related Tasks
The following are insights gathered during the execution of similar tasks. You may refer to them during your task execution to improve problem-solving accuracy.

{insights}
---
{skills_section}
## Your Turn: Take Action!
Use the above examples and insights as a foundation, and now work on the following task:
{task_description}
"""

task_format: str = """
### Task description:   
{task_description}    

### Key steps:
{key_steps}

### Detailed trajectory:
{trajectory}
"""

temp = """NOTE: You must use the command `think: <your thoughts here>` if you want to think!!!
    - Right output: think: To solve the task, ...
    - Wrong output: To solve the task, ... """

def format_task_prompt_with_insights(
    few_shots: list[str],
    memory_few_shots: list[str],
    insights: list[str],
    task_description: str,
    skills: str = "",
) -> str: 
    
    existing_rules_text: str = '\n'.join([f'{i}. {r}' for i, r in enumerate(insights, 1)])
    memory_few_shots: str = '\n\n'.join([f"Task {i+1}:\n{shot}" for i, shot in enumerate(memory_few_shots)])

    skills_section = ""
    if skills:
        skills_section = f"\n{skills}\n---\n"

    user_prompt: str = task_solve_with_insights.format(
        few_shots='\n'.join(few_shots),
        memory_few_shots=memory_few_shots,
        task_description=task_description,   
        insights=existing_rules_text,
        skills_section=skills_section,
    )

    return user_prompt

def format_task_context(task_description: str, task_traj: str, key_steps: str = None) -> str:

    return task_format.format(
        task_description=task_description,
        key_steps=key_steps,
        trajectory=task_traj
    )
