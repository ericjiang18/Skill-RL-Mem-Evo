"""
Skill Miner Module for G-Memory++

Extracts reusable skills/macro-procedures from successful trajectories.
Clusters similar trajectories and synthesizes generalizable skills.

Enhancements (Claude Skills–inspired):
- Lowered mining thresholds (min_cluster_size=2, trigger at 4 trajectories)
- Parameterized steps with {object}/{target} placeholders
- Execution tracking (total_uses, success_count, avg_steps_saved)
- Auto-deactivation of poorly performing skills
- Goal-pattern matching for fast skill lookup
"""

import os
import json
import re
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict

try:
    from finch import FINCH
    HAS_FINCH = True
except ImportError:
    HAS_FINCH = False

try:
    from sklearn.cluster import AgglomerativeClustering, DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from mas.llm import LLMCallable, Message
from .goal_module import StructuredGoal


# ================================ Skill Prompts ================================

SKILL_EXTRACTION_SYSTEM = """You are an expert at extracting reusable procedures from task execution logs.
Given a set of similar successful task trajectories, synthesize a generalizable "skill" — a reusable procedure that can be applied to similar future tasks.

A skill must include:
1. Name: A short, descriptive name
2. Description: What the skill accomplishes and when to use it
3. Goal Pattern: A pattern describing which tasks this skill applies to (use * as wildcard, e.g. "put * in/on *")
4. Preconditions: What must be true before applying the skill
5. Steps: The sequence of actions to perform. Use {object}, {target}, {location} as placeholders for task-specific items.
6. Postconditions: What will be true after the skill succeeds

The skill should be general enough to apply to similar situations, not specific to one instance.
Use placeholders like {object}, {target}, {location} in steps to make them parameterized.
"""

SKILL_EXTRACTION_USER = """## Goal Type: {goal_type}

## Successful Trajectories:
{trajectories}

## Common Patterns Observed:
{patterns}

Generate a reusable skill based on these trajectories. Output in this exact format:
Name: <skill name>
Description: <when to use this skill>
Goal Pattern: <pattern with * wildcards>
Preconditions:
- <condition 1>
- <condition 2>
Steps:
1. <step with {{object}}/{{target}} placeholders>
2. <step>
Postconditions:
- <expected result>
"""

SKILL_MATCHING_PROMPT = """Given the current task goal and available skills, determine which skill (if any) is applicable.

Current Goal:
{goal}

Available Skills:
{skills}

If a skill is applicable, respond with the skill name. If no skill applies, respond with "NONE".
Your response (skill name or NONE):
"""


# ================================ Data Classes ================================

@dataclass
class Skill:
    """Represents a reusable skill/macro-procedure with execution tracking."""
    
    skill_id: str
    name: str
    description: str
    goal_type: str
    goal_pattern: str = ""
    
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    
    # Execution statistics
    total_uses: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_steps_saved: float = 0.0
    
    active: bool = True
    
    supporting_trajectories: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        return self.success_count / total
    
    def record_use(self, success: bool, steps_saved: float = 0.0):
        """Record a skill usage and auto-deactivate if performing poorly."""
        self.total_uses += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        
        if steps_saved > 0:
            alpha = 1.0 / self.total_uses
            self.avg_steps_saved = (1 - alpha) * self.avg_steps_saved + alpha * steps_saved
        
        # Auto-deactivate poorly performing skills
        if self.total_uses >= 5 and self.success_rate < 0.3:
            self.active = False
    
    def matches_goal(self, task_description: str) -> bool:
        """Check if this skill's goal_pattern matches a task description."""
        if not self.goal_pattern:
            return False
        
        pattern = self.goal_pattern.replace('*', '.*')
        try:
            return bool(re.search(pattern, task_description, re.IGNORECASE))
        except re.error:
            return self.goal_pattern.lower() in task_description.lower()
    
    def instantiate_steps(self, task_objects: Dict[str, str]) -> List[str]:
        """Replace placeholders in steps with actual task objects."""
        result = []
        for step in self.steps:
            instantiated = step
            for key, value in task_objects.items():
                instantiated = instantiated.replace(f'{{{key}}}', value)
            result.append(instantiated)
        return result
    
    def to_str(self) -> str:
        parts = [
            f"### {self.name}",
            f"Description: {self.description}",
            f"Goal Type: {self.goal_type}",
        ]
        
        if self.goal_pattern:
            parts.append(f"Applies to: {self.goal_pattern}")
        
        if self.preconditions:
            parts.append(f"Preconditions: {', '.join(self.preconditions)}")
        
        parts.append("Steps:")
        for i, step in enumerate(self.steps, 1):
            parts.append(f"  {i}. {step}")
        
        if self.postconditions:
            parts.append(f"Postconditions: {', '.join(self.postconditions)}")
        
        parts.append(f"Success Rate: {self.success_rate:.1%} ({self.total_uses} uses)")
        
        return "\n".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "goal_type": self.goal_type,
            "goal_pattern": self.goal_pattern,
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "steps": self.steps,
            "total_uses": self.total_uses,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "avg_steps_saved": self.avg_steps_saved,
            "active": self.active,
            "supporting_trajectories": self.supporting_trajectories,
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Skill":
        return Skill(
            skill_id=data["skill_id"],
            name=data["name"],
            description=data["description"],
            goal_type=data["goal_type"],
            goal_pattern=data.get("goal_pattern", ""),
            preconditions=data.get("preconditions", []),
            postconditions=data.get("postconditions", []),
            steps=data.get("steps", []),
            total_uses=data.get("total_uses", 0),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            avg_steps_saved=data.get("avg_steps_saved", 0.0),
            active=data.get("active", True),
            supporting_trajectories=data.get("supporting_trajectories", []),
        )


@dataclass
class TrajectoryRecord:
    """A trajectory to be analyzed for skill extraction."""
    
    task_id: str
    goal: StructuredGoal
    trajectory: str
    key_steps: List[str] = field(default_factory=list)
    success: bool = True
    embedding: Optional[np.ndarray] = None


# ================================ Skill Miner ================================

class SkillMiner:
    """
    Mines reusable skills from successful trajectories.
    
    Process:
    1. Collect successful trajectories with their goals
    2. Cluster trajectories by similarity (lowered thresholds)
    3. For each cluster, extract a generalizable skill with parameterized steps
    4. Track skill usage and auto-deactivate poor performers
    """
    
    def __init__(
        self,
        llm_model: LLMCallable,
        embedding_func: Any,
        working_dir: str,
        min_cluster_size: int = 2,
        similarity_threshold: float = 0.65
    ):
        self.llm_model = llm_model
        self.embedding_func = embedding_func
        self.working_dir = working_dir
        self.min_cluster_size = min_cluster_size
        self.similarity_threshold = similarity_threshold
        
        self.skills: Dict[str, Skill] = {}
        self.pending_trajectories: List[TrajectoryRecord] = []
        
        self.trajectory_embeddings: List[np.ndarray] = []
        self.trajectory_ids: List[str] = []
        
        self.skills_path = os.path.join(working_dir, "skills.json")
        self.pending_path = os.path.join(working_dir, "pending_trajectories.json")
        
        self._load()
    
    def add_trajectory(
        self,
        task_id: str,
        goal: StructuredGoal,
        trajectory: str,
        key_steps: List[str],
        success: bool = True
    ):
        """Add a trajectory for potential skill extraction."""
        if not success:
            return
        
        embedding = self.embedding_func.embed_query(trajectory)
        
        record = TrajectoryRecord(
            task_id=task_id,
            goal=goal,
            trajectory=trajectory,
            key_steps=key_steps,
            success=success,
            embedding=embedding
        )
        
        self.pending_trajectories.append(record)
        self.trajectory_embeddings.append(embedding)
        self.trajectory_ids.append(task_id)
        
        # Lowered trigger threshold: mine at 4 trajectories (was 6)
        if len(self.pending_trajectories) >= self.min_cluster_size * 2:
            self.mine_skills()
        
        self._save()
    
    def mine_skills(self) -> List[Skill]:
        """Cluster trajectories and extract skills from each cluster."""
        if len(self.pending_trajectories) < self.min_cluster_size:
            return []
        
        clusters = self._cluster_trajectories()
        
        new_skills = []
        for cluster_id, indices in clusters.items():
            if len(indices) >= self.min_cluster_size:
                cluster_trajectories = [self.pending_trajectories[i] for i in indices]
                
                skill = self._extract_skill_from_cluster(cluster_trajectories)
                if skill:
                    # Check for duplicate skills before adding
                    if not self._is_duplicate_skill(skill):
                        self.skills[skill.skill_id] = skill
                        new_skills.append(skill)
        
        self._prune_trajectories()
        self._save()
        
        return new_skills
    
    def _is_duplicate_skill(self, new_skill: Skill) -> bool:
        """Check if a similar skill already exists."""
        if not new_skill.embedding:
            return False
        
        for existing in self.skills.values():
            if existing.embedding is not None:
                sim = np.dot(new_skill.embedding, existing.embedding) / (
                    np.linalg.norm(new_skill.embedding) * np.linalg.norm(existing.embedding) + 1e-8
                )
                if sim > 0.9 and existing.goal_type == new_skill.goal_type:
                    return True
        return False
    
    def _cluster_trajectories(self) -> Dict[int, List[int]]:
        """Cluster trajectories by embedding similarity."""
        if len(self.trajectory_embeddings) == 0:
            return {}
        
        embeddings = np.array(self.trajectory_embeddings)
        
        goal_types = [t.goal.verb for t in self.pending_trajectories]
        unique_goals = list(set(goal_types))
        
        clusters = defaultdict(list)
        
        for goal_type in unique_goals:
            goal_indices = [i for i, g in enumerate(goal_types) if g == goal_type]
            
            if len(goal_indices) < self.min_cluster_size:
                continue
            
            goal_embeddings = embeddings[goal_indices]
            sub_clusters = self._cluster_embeddings(goal_embeddings)
            
            for sub_cluster_id, sub_indices in sub_clusters.items():
                original_indices = [goal_indices[i] for i in sub_indices]
                cluster_key = f"{goal_type}_{sub_cluster_id}"
                clusters[cluster_key] = original_indices
        
        return clusters
    
    def _cluster_embeddings(self, embeddings: np.ndarray) -> Dict[int, List[int]]:
        """Cluster embeddings using available clustering method."""
        n_samples = len(embeddings)
        
        if n_samples < 2:
            return {0: list(range(n_samples))}
        
        if HAS_FINCH:
            try:
                c, num_clust, _ = FINCH(embeddings, distance='cosine', verbose=False)
                labels = c[:, 0] if c.ndim > 1 else c
            except Exception:
                labels = self._simple_clustering(embeddings)
        elif HAS_SKLEARN:
            try:
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=1 - self.similarity_threshold,
                    metric='cosine',
                    linkage='average'
                )
                labels = clustering.fit_predict(embeddings)
            except Exception:
                labels = self._simple_clustering(embeddings)
        else:
            labels = self._simple_clustering(embeddings)
        
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[label].append(i)
        
        return clusters
    
    def _simple_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """Simple greedy clustering when no libraries available."""
        n = len(embeddings)
        labels = np.full(n, -1)
        current_label = 0
        
        for i in range(n):
            if labels[i] >= 0:
                continue
            
            labels[i] = current_label
            
            for j in range(i + 1, n):
                if labels[j] >= 0:
                    continue
                
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8
                )
                
                if sim >= self.similarity_threshold:
                    labels[j] = current_label
            
            current_label += 1
        
        return labels
    
    def _extract_skill_from_cluster(
        self,
        trajectories: List[TrajectoryRecord]
    ) -> Optional[Skill]:
        """Extract a skill from a cluster of similar trajectories."""
        if not trajectories:
            return None
        
        goal_type = trajectories[0].goal.verb
        
        trajectory_texts = "\n---\n".join([
            f"Task: {t.goal.raw_task}\nSteps:\n" + "\n".join(f"- {s}" for s in t.key_steps)
            for t in trajectories[:5]
        ])
        
        all_steps = [step for t in trajectories for step in t.key_steps]
        step_counts = defaultdict(int)
        for step in all_steps:
            step_counts[step.lower().strip()] += 1
        
        common_patterns = [
            step for step, count in step_counts.items()
            if count >= len(trajectories) * 0.4
        ]
        patterns_text = "\n".join(f"- {p}" for p in common_patterns[:10]) if common_patterns else "No clear common patterns"
        
        try:
            response = self.llm_model(
                messages=[
                    Message("system", SKILL_EXTRACTION_SYSTEM),
                    Message("user", SKILL_EXTRACTION_USER.format(
                        goal_type=goal_type,
                        trajectories=trajectory_texts,
                        patterns=patterns_text
                    ))
                ],
                temperature=0.3,
                max_tokens=1024
            )
            
            skill = self._parse_skill_response(response, goal_type, trajectories)
            return skill
            
        except Exception as e:
            print(f"Skill extraction failed: {e}")
            return None
    
    def _parse_skill_response(
        self,
        response: str,
        goal_type: str,
        trajectories: List[TrajectoryRecord]
    ) -> Optional[Skill]:
        """Parse LLM response into a Skill object."""
        name = ""
        description = ""
        goal_pattern = ""
        preconditions = []
        postconditions = []
        steps = []
        
        name_match = re.search(r'(?:Name|Skill):\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if name_match:
            name = name_match.group(1).strip()
        else:
            name = f"skill_{goal_type}"
        
        desc_match = re.search(r'Description:\s*(.+?)(?:\n\n|Preconditions|Goal Pattern|Steps)', response, re.IGNORECASE | re.DOTALL)
        if desc_match:
            description = desc_match.group(1).strip()
        else:
            description = f"Skill for {goal_type} tasks"
        
        pattern_match = re.search(r'Goal Pattern:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if pattern_match:
            goal_pattern = pattern_match.group(1).strip()
        
        pre_match = re.search(r'Preconditions?:\s*(.+?)(?:\n\n|Postconditions|Steps)', response, re.IGNORECASE | re.DOTALL)
        if pre_match:
            pre_text = pre_match.group(1)
            preconditions = [p.strip().lstrip('- ') for p in pre_text.split('\n') if p.strip()]
        
        steps_match = re.search(r'Steps?:\s*(.+?)(?:\n\n|Postconditions|$)', response, re.IGNORECASE | re.DOTALL)
        if steps_match:
            steps_text = steps_match.group(1)
            steps = [s.strip().lstrip('0123456789.- ') for s in steps_text.split('\n') if s.strip()]
        
        post_match = re.search(r'Postconditions?:\s*(.+?)(?:\n\n|$)', response, re.IGNORECASE | re.DOTALL)
        if post_match:
            post_text = post_match.group(1)
            postconditions = [p.strip().lstrip('- ') for p in post_text.split('\n') if p.strip()]
        
        skill_id = f"skill_{goal_type}_{len(self.skills)}"
        
        skill_text = f"{name} {description} " + " ".join(steps)
        embedding = self.embedding_func.embed_query(skill_text)
        
        return Skill(
            skill_id=skill_id,
            name=name,
            description=description,
            goal_type=goal_type,
            goal_pattern=goal_pattern,
            preconditions=preconditions,
            postconditions=postconditions,
            steps=steps,
            supporting_trajectories=[t.task_id for t in trajectories],
            embedding=embedding
        )
    
    def retrieve_skills(
        self,
        goal: StructuredGoal,
        top_k: int = 3
    ) -> List[Tuple[Skill, float]]:
        """Retrieve relevant active skills for a goal."""
        if not self.skills:
            return []
        
        # Only consider active skills
        active_skills = [s for s in self.skills.values() if s.active]
        if not active_skills:
            return []
        
        # Fast path: goal_pattern match
        pattern_matches = [s for s in active_skills if s.matches_goal(goal.raw_task)]
        if pattern_matches:
            candidates = pattern_matches
        else:
            matching_type = [s for s in active_skills if s.goal_type == goal.verb]
            candidates = matching_type if matching_type else active_skills
        
        goal_text = goal.to_str()
        goal_embedding = self.embedding_func.embed_query(goal_text)
        
        scored = []
        for skill in candidates:
            if skill.embedding is not None:
                sim = np.dot(goal_embedding, skill.embedding) / (
                    np.linalg.norm(goal_embedding) * np.linalg.norm(skill.embedding) + 1e-8
                )
            else:
                sim = 1.0 if skill.goal_type == goal.verb else 0.5
            
            sr_boost = 0.5 + 0.5 * skill.success_rate
            score = sim * sr_boost
            scored.append((skill, float(score)))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
    
    def update_skill_stats(self, skill_id: str, success: bool, steps_saved: float = 0.0):
        """Update skill statistics after use, with auto-deactivation."""
        if skill_id in self.skills:
            self.skills[skill_id].record_use(success, steps_saved)
            self._save()
    
    def get_skill_by_id(self, skill_id: str) -> Optional[Skill]:
        return self.skills.get(skill_id)
    
    def get_active_skills(self) -> List[Skill]:
        """Get all active skills."""
        return [s for s in self.skills.values() if s.active]
    
    def reactivate_skill(self, skill_id: str):
        """Manually reactivate a deactivated skill."""
        if skill_id in self.skills:
            self.skills[skill_id].active = True
            self._save()
    
    def _prune_trajectories(self):
        max_pending = 100
        if len(self.pending_trajectories) > max_pending:
            self.pending_trajectories = self.pending_trajectories[-max_pending:]
            self.trajectory_embeddings = self.trajectory_embeddings[-max_pending:]
            self.trajectory_ids = self.trajectory_ids[-max_pending:]
    
    def _save(self):
        os.makedirs(self.working_dir, exist_ok=True)
        
        skills_data = {skill_id: skill.to_dict() for skill_id, skill in self.skills.items()}
        with open(self.skills_path, 'w') as f:
            json.dump(skills_data, f, indent=2)
        
        pending_data = []
        for t in self.pending_trajectories:
            pending_data.append({
                "task_id": t.task_id,
                "goal": t.goal.to_dict(),
                "trajectory": t.trajectory,
                "key_steps": t.key_steps,
                "success": t.success,
            })
        with open(self.pending_path, 'w') as f:
            json.dump(pending_data, f, indent=2)
    
    def _load(self):
        if os.path.exists(self.skills_path):
            try:
                with open(self.skills_path, 'r') as f:
                    skills_data = json.load(f)
                
                for skill_id, data in skills_data.items():
                    skill = Skill.from_dict(data)
                    skill_text = f"{skill.name} {skill.description} " + " ".join(skill.steps)
                    skill.embedding = self.embedding_func.embed_query(skill_text)
                    self.skills[skill_id] = skill
            except Exception as e:
                print(f"Failed to load skills: {e}")
        
        if os.path.exists(self.pending_path):
            try:
                with open(self.pending_path, 'r') as f:
                    pending_data = json.load(f)
                
                for data in pending_data:
                    goal = StructuredGoal.from_dict(data["goal"])
                    embedding = self.embedding_func.embed_query(data["trajectory"])
                    
                    record = TrajectoryRecord(
                        task_id=data["task_id"],
                        goal=goal,
                        trajectory=data["trajectory"],
                        key_steps=data["key_steps"],
                        success=data["success"],
                        embedding=embedding
                    )
                    self.pending_trajectories.append(record)
                    self.trajectory_embeddings.append(embedding)
                    self.trajectory_ids.append(data["task_id"])
            except Exception as e:
                print(f"Failed to load pending trajectories: {e}")
