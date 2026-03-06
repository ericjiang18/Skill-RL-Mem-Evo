"""
Skill-Conditioned RL + LLM-Driven Memory Evolution

Architecture:
1. Skill-Conditioned RL — Q(goal_type, skill_id) selects which skill to use
2. ExpRAG — lightweight experience storage & retrieval
3. LLM-Driven Refine — after each task, one LLM call to:
   - Extract insight from the experience
   - Update skill steps (if skill was used)
   - Generate avoidance rules (if failed)
"""

import os
import json
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

from .gmemory_plus import GMemoryPlus, GMemoryPlusConfig
from .skill_rl import (
    SkillRL,
    SkillRLConfig,
    Experience,
    Episode,
    ReplayStrategy,
    NO_SKILL,
)
from .goal_module import GoalParser, GoalMatcher, StructuredGoal
from ..common import MASMessage, AgentMessage, StateChain

from mas.llm import LLMCallable, Message
from mas.utils import EmbeddingFunc


# ================================ LLM Refine Prompt ================================

_REFINE_PROMPT = """\
You are a memory refinement module. After a task attempt, analyze the outcome and produce structured learnings.

Task Goal: {goal}
Goal Type: {goal_type}
Outcome: {outcome}
Total Steps: {total_steps}
Skill Used: {skill_used}
Key Actions Taken:
{key_actions}
Environment Feedback: {feedback}

Based on this experience, provide EXACTLY:

1. INSIGHT: One concise sentence about what works or doesn't work for this type of task. Be specific and actionable.

2. SKILL_UPDATE: If a skill was used, suggest how its steps should be improved. If no skill was used or no update needed, write "NONE".

3. AVOID: If the task failed, one concise sentence about what to avoid next time. If succeeded, write "NONE".

Format your response EXACTLY as:
INSIGHT: <your insight>
SKILL_UPDATE: <step improvements or NONE>
AVOID: <avoidance rule or NONE>"""


# ================================ ExpRAG ================================

class ExpRAGMemory:
    """Lightweight experience retrieval replacing complex RL pipeline."""

    def __init__(self, embedding_func, working_dir: str):
        self.embedding_func = embedding_func
        self.working_dir = working_dir
        self.experiences: List[Dict[str, Any]] = []
        self._exp_path = os.path.join(working_dir, "exprag_experiences.json")
        self._load()

    def store_experience(
        self,
        task_goal: str,
        goal_type: str,
        trajectory_summary: str,
        key_actions: List[str],
        success: bool,
        total_steps: int = 0,
        skill_used: str = "",
        refined_insight: str = "",
    ):
        embedding = self.embedding_func.embed_query(task_goal)
        exp = {
            "task_goal": task_goal,
            "goal_type": goal_type,
            "trajectory_summary": trajectory_summary,
            "key_actions": key_actions,
            "success": success,
            "total_steps": total_steps,
            "skill_used": skill_used,
            "refined_insight": refined_insight,
            "embedding": embedding,
        }
        self.experiences.append(exp)
        if len(self.experiences) > 500:
            self.experiences = self.experiences[-500:]
        self._save()

    def retrieve_experience(
        self,
        task_goal: str,
        goal_type: str = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        if not self.experiences:
            return []

        query_embedding = self.embedding_func.embed_query(task_goal)
        scored = []
        for exp in self.experiences:
            if exp.get("embedding") is None:
                continue
            emb = np.array(exp["embedding"])
            sim = np.dot(query_embedding, emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-8
            )
            if goal_type and exp.get("goal_type") == goal_type:
                sim *= 1.2
            if exp.get("success"):
                sim *= 1.3
            scored.append((exp, float(sim)))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for exp, score in scored[:top_k]:
            result = {k: v for k, v in exp.items() if k != "embedding"}
            result["relevance_score"] = score
            results.append(result)
        return results

    def _save(self):
        os.makedirs(self.working_dir, exist_ok=True)
        save_data = [{k: v for k, v in exp.items() if k != "embedding"} for exp in self.experiences]
        with open(self._exp_path, 'w') as f:
            json.dump(save_data, f, indent=2)

    def _load(self):
        if not os.path.exists(self._exp_path):
            return
        try:
            with open(self._exp_path, 'r') as f:
                data = json.load(f)
            for d in data:
                d["embedding"] = self.embedding_func.embed_query(d.get("task_goal", ""))
                self.experiences.append(d)
        except Exception as e:
            print(f"Failed to load ExpRAG: {e}")


# ================================ Configuration ================================

@dataclass
class SkillMemoryConfig(GMemoryPlusConfig):
    enable_skill_rl: bool = True
    rl_learning_rate: float = 0.3
    rl_exploration_bonus: float = 0.2
    enable_exprag: bool = True
    enable_llm_refine: bool = True

GoalRLMemoryConfig = SkillMemoryConfig  # backward compat


# ================================ Skill Memory ================================

@dataclass
class SkillMemory(GMemoryPlus):
    """
    Skill-Conditioned RL + ExpRAG + LLM-Driven Refine.

    Key flow:
    1. init_task → RL selects skill via Q(goal_type, skill_id)
    2. execution → record steps (lightweight)
    3. save_task → RL updates Q-value + LLM refines memory + ExpRAG stores
    """

    rl_config: SkillMemoryConfig = field(default_factory=SkillMemoryConfig)

    def __post_init__(self):
        super().__post_init__()
        if not hasattr(self, 'rl_config') or self.rl_config is None:
            self.rl_config = SkillMemoryConfig()

        if self.rl_config.enable_skill_rl:
            self._init_skill_rl()
        else:
            self.skill_rl = None

        if self.rl_config.enable_exprag:
            self.exprag = ExpRAGMemory(
                embedding_func=self.embedding_func,
                working_dir=os.path.join(self.persist_dir, "exprag"),
            )
        else:
            self.exprag = None

        self._current_state: str = ""
        self._step_history: List[Dict[str, Any]] = []
        self._selected_skill_id: str = NO_SKILL

        print(f"SkillMemory initialized (SkillRL: {self.rl_config.enable_skill_rl}, "
              f"ExpRAG: {self.rl_config.enable_exprag}, "
              f"LLM-Refine: {self.rl_config.enable_llm_refine})")

    def _init_skill_rl(self):
        rl_config = SkillRLConfig(
            learning_rate=self.rl_config.rl_learning_rate,
            exploration_bonus=self.rl_config.rl_exploration_bonus,
        )
        self.skill_rl = SkillRL(
            llm_model=self.llm_model,
            embedding_func=self.embedding_func,
            goal_parser=self.goal_parser,
            working_dir=os.path.join(self.persist_dir, "skill_rl"),
            config=rl_config,
        )
        self.skill_rl.load()

    # ======================== Skill Selection via RL ========================

    def select_skill_for_task(self, available_skills) -> Tuple[Any, str, float]:
        """
        Use Skill-Conditioned RL to pick the best skill for the current task.

        Returns:
            (skill_object_or_None, skill_id, q_value)
        """
        if not self.skill_rl or not self._current_goal or not available_skills:
            self._selected_skill_id = NO_SKILL
            return None, NO_SKILL, 0.0

        active = [s for s in available_skills if s.active]
        if not active:
            self._selected_skill_id = NO_SKILL
            return None, NO_SKILL, 0.0

        skill_ids = [s.skill_id for s in active]
        best_id, q_value = self.skill_rl.select_skill(self._current_goal, skill_ids)

        if best_id is None:
            self._selected_skill_id = NO_SKILL
            return None, NO_SKILL, q_value

        self._selected_skill_id = best_id

        for s in active:
            if s.skill_id == best_id:
                return s, best_id, q_value

        self._selected_skill_id = NO_SKILL
        return None, NO_SKILL, q_value

    # ======================== Task Context ========================

    def init_task_context(
        self,
        task_main: str,
        task_description: str = None,
        screenshot: Any = None,
        domain: str = None,
    ) -> MASMessage:
        mas_message = super().init_task_context(
            task_main, task_description, screenshot, domain
        )

        if self.skill_rl and self._current_goal:
            self.skill_rl.start_episode(
                task_id=self._current_task_id,
                goal=self._current_goal,
                initial_state=task_description or task_main,
                skill_id=self._selected_skill_id,
            )

        self._current_state = task_description or task_main
        self._step_history = []

        return mas_message

    # ======================== Step Processing ========================

    def process_step(self, state, action, next_state, reward, done):
        if self.skill_rl and self._current_goal:
            self.skill_rl.step(state, action, next_state, reward, done)

        self._current_state = next_state
        self._step_history.append({
            "state": state, "action": action,
            "next_state": next_state, "reward": reward, "done": done,
        })

    def move_memory_state(self, action: str, observation: str, **kwargs):
        reward = kwargs.get('reward', 0.0)
        done = kwargs.get('done', False)
        self.process_step(self._current_state, action, observation, reward, done)
        super().move_memory_state(action, observation, **kwargs)

    # ======================== Retrieval ========================

    def retrieve_memory(self, query_task, successful_topk=2, failed_topk=1,
                        insight_topk=10, skill_topk=3, threshold=0.3,
                        screenshot=None, **kwargs):
        result = super().retrieve_memory(
            query_task=query_task, successful_topk=successful_topk,
            failed_topk=failed_topk, insight_topk=insight_topk,
            skill_topk=skill_topk, threshold=threshold,
            screenshot=screenshot, **kwargs
        )
        return (result[0][:successful_topk], result[1][:failed_topk], result[2], result[3])

    # ======================== ExpRAG ========================

    def get_exprag_experiences(self, top_k=3) -> List[Dict[str, Any]]:
        if not self.exprag or not self._current_goal:
            return []
        return self.exprag.retrieve_experience(
            self._current_goal.raw_task, self._current_goal.verb, top_k
        )

    # ======================== LLM-Driven Refine ========================

    def _llm_refine(self, label: bool, feedback: str = None) -> Dict[str, str]:
        """
        One compact LLM call to extract learnings from the task experience.

        Returns dict with keys: insight, skill_update, avoid
        """
        if not self.rl_config.enable_llm_refine:
            return {"insight": "", "skill_update": "", "avoid": ""}

        if not self._current_goal:
            return {"insight": "", "skill_update": "", "avoid": ""}

        key_actions = [
            h["action"] for h in self._step_history
            if not h["action"].lower().startswith(('think', 'thought'))
        ][:15]

        skill_name = NO_SKILL
        if self._selected_skill_id != NO_SKILL and self.skill_miner:
            sk = self.skill_miner.get_skill_by_id(self._selected_skill_id)
            if sk:
                skill_name = sk.name

        prompt_text = _REFINE_PROMPT.format(
            goal=self._current_goal.raw_task,
            goal_type=self._current_goal.verb,
            outcome="SUCCESS" if label else "FAILURE",
            total_steps=len(self._step_history),
            skill_used=skill_name if skill_name != NO_SKILL else "None (free action)",
            key_actions='\n'.join(f"- {a}" for a in key_actions) if key_actions else "(no actions recorded)",
            feedback=feedback or "(no feedback)",
        )

        try:
            response = self.llm_model(
                messages=[
                    Message("system", "You are a concise memory refinement module. Follow the format exactly."),
                    Message("user", prompt_text),
                ],
                temperature=0.2,
                max_tokens=300,
            )
            return self._parse_refine_response(response)
        except Exception as e:
            print(f"LLM refine failed: {e}")
            return {"insight": "", "skill_update": "", "avoid": ""}

    def _parse_refine_response(self, response: str) -> Dict[str, str]:
        result = {"insight": "", "skill_update": "", "avoid": ""}

        insight_match = re.search(r'INSIGHT:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if insight_match:
            result["insight"] = insight_match.group(1).strip()

        skill_match = re.search(r'SKILL_UPDATE:\s*(.+?)(?:\nAVOID|\Z)', response, re.IGNORECASE | re.DOTALL)
        if skill_match:
            val = skill_match.group(1).strip()
            if val.upper() != "NONE":
                result["skill_update"] = val

        avoid_match = re.search(r'AVOID:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if avoid_match:
            val = avoid_match.group(1).strip()
            if val.upper() != "NONE":
                result["avoid"] = val

        return result

    def _apply_refine_results(self, refine: Dict[str, str], label: bool):
        """Apply LLM refine results to memory components."""
        # 1. Add refined insight to insights layer
        if refine.get("insight"):
            insight_text = refine["insight"]
            if refine.get("avoid"):
                insight_text += f" AVOID: {refine['avoid']}"
            try:
                self.insights_layer.insights_memory.append({
                    'rule': insight_text,
                    'score': 3 if label else 2,
                    'positive_correlation_tasks': [self._current_goal.raw_task] if label else [],
                    'negative_correlation_tasks': [] if label else [self._current_goal.raw_task],
                })
                self.insights_layer._index_done()
            except Exception:
                pass

        # 2. Update skill steps if skill was used and LLM suggests improvements
        if refine.get("skill_update") and self._selected_skill_id != NO_SKILL and self.skill_miner:
            sk = self.skill_miner.get_skill_by_id(self._selected_skill_id)
            if sk and not label:
                # On failure with update suggestion, append the suggestion to description
                sk.description += f" | Refinement: {refine['skill_update'][:200]}"
                self.skill_miner._save()

    # ======================== Save Task Context ========================

    def save_task_context(self, label: bool, feedback: str = None) -> MASMessage:
        # 1. Skill-Conditioned RL update
        if self.skill_rl:
            if self.skill_rl.current_episode:
                self.skill_rl.current_episode.skill_id_used = self._selected_skill_id
            episode_stats = self.skill_rl.end_episode(success=label)
            if self.current_task_context:
                self.current_task_context.add_extra_field('rl_episode_stats', episode_stats)

        # 2. LLM-Driven Refine (one compact LLM call)
        refine_results = self._llm_refine(label, feedback)
        self._apply_refine_results(refine_results, label)

        # 3. Store in ExpRAG
        if self.exprag and self._current_goal:
            key_actions = [
                h["action"] for h in self._step_history
                if not h["action"].lower().startswith(('think', 'thought'))
            ]
            self.exprag.store_experience(
                task_goal=self._current_goal.raw_task,
                goal_type=self._current_goal.verb,
                trajectory_summary=self.current_task_context.task_trajectory[:500] if self.current_task_context else "",
                key_actions=key_actions[:20],
                success=label,
                total_steps=len(self._step_history),
                skill_used=self._selected_skill_id,
                refined_insight=refine_results.get("insight", ""),
            )

        # 4. Update skill stats
        if self._selected_skill_id != NO_SKILL and self.skill_miner:
            self.skill_miner.update_skill_stats(self._selected_skill_id, label)

        # Reset
        self._selected_skill_id = NO_SKILL

        return super().save_task_context(label=label, feedback=feedback)

    def add_memory(self, mas_message: MASMessage) -> None:
        super().add_memory(mas_message)
        if self.skill_rl and self.memory_size % 10 == 0:
            self.skill_rl.save()

    # ======================== Stats ========================

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        if self.skill_rl:
            stats["skill_rl"] = self.skill_rl.get_statistics()
        if self.exprag:
            stats["exprag_experiences"] = len(self.exprag.experiences)
        return stats


# ================================ Factory ================================

def create_skill_memory(
    llm_model: LLMCallable,
    embedding_func: EmbeddingFunc,
    working_dir: str,
    enable_all_features: bool = True,
    **kwargs,
) -> SkillMemory:
    config = SkillMemoryConfig(
        enable_goal_module=enable_all_features,
        enable_skill_miner=enable_all_features,
        enable_skill_rl=True,
        rl_learning_rate=kwargs.get('learning_rate', 0.3),
        rl_exploration_bonus=kwargs.get('exploration_bonus', 0.2),
        enable_exprag=True,
        enable_llm_refine=True,
    )
    return SkillMemory(
        namespace="skill_memory",
        global_config={"working_dir": working_dir},
        llm_model=llm_model,
        embedding_func=embedding_func,
        config=config,
        rl_config=config,
    )

GoalRLMemory = SkillMemory  # backward compat
create_goal_rl_memory = create_skill_memory  # backward compat
