"""
Math-specific Goal Module for G-Memory++
Parses math problems into structured goals and computes similarity.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import re
import json
import numpy as np

from mas.llm import LLMCallable, Message

# ================================ Goal Prompts ================================

MATH_GOAL_PARSE_SYSTEM_PROMPT = """You are an expert mathematician and problem parser.
Given a math problem description, extract the following structured information:
1. domain: Always "math"
2. problem_type: The sub-domain (e.g., "algebra", "geometry", "number_theory", "calculus", "probability")
3. core_question: What exactly are we asked to find? (e.g., "Find the area of the triangle", "Solve for x")
4. given_conditions: List of explicit conditions and known values given in the problem.
5. target_format: Expected output format (e.g., "integer", "fraction", "equation", "coordinate")

Output your response as a valid JSON object with EXACTLY these 5 keys.
"""

MATH_GOAL_PARSE_USER_PROMPT = """Parse the following math problem into a structured goal:

Problem: {task_description}

Context (if any): {context}

Output the structured goal as JSON:
"""

MATH_GOAL_SIMILARITY_SYSTEM_PROMPT = """You are an expert at comparing math problems.
Given two parsed math problems, rate their similarity on a scale of 0-10 where:
- 0: Completely different topics and methods
- 5: Similar topic (e.g., both are quadratic equations) but different specifics
- 10: Nearly identical problem structures requiring the exact same solution steps

Consider:
1. Same problem_type (algebra, geometry, etc.)
2. Similar core_questions (e.g., both asking for an area, or both solving for a variable)
3. Structural similarity in given_conditions
"""

MATH_GOAL_SIMILARITY_USER_PROMPT = """Rate the similarity between these two math problems:

Problem 1: {goal1}

Problem 2: {goal2}

Respond with ONLY a number from 0-10:
"""


# ================================ Data Classes ================================

@dataclass
class StructuredMathGoal:
    """Represents a parsed, structured goal from a math problem."""
    
    domain: str = "math"
    problem_type: str = "unknown"  # algebra, geometry, number_theory, etc.
    core_question: str = ""
    given_conditions: List[str] = field(default_factory=list)
    target_format: str = "unknown"
    difficulty: float = 0.5
    raw_task: str = ""
    
    def to_str(self) -> str:
        """Convert goal to a readable string."""
        parts = [
            f"Domain: {self.domain}",
            f"Type: {self.problem_type}",
            f"Question: {self.core_question}",
            f"Format expected: {self.target_format}"
        ]
        if self.given_conditions:
            parts.append(f"Given Conditions: {', '.join(self.given_conditions)}")
        return "\n".join(parts)
    
    def to_features(self) -> np.ndarray:
        """Convert goal to a feature vector for GCN."""
        # Simple one-hot encoding for common math types
        types = ["algebra", "geometry", "number_theory", "calculus", "probability", "counting", "unknown"]
        
        type_vec = [1.0 if t == self.problem_type else 0.0 for t in types]
        
        # Add difficulty and conditions count
        extra_features = [
            self.difficulty,
            len(self.given_conditions) / 10.0,  # Normalized condition count
        ]
        
        # Pad to match expected dimension if needed by existing generic algorithms 
        # (original goal feature was roughly 7 domains + 14 verbs + 4 = 25 dims)
        # Here we only have 7 types + 2 features = 9 dims. 
        # Let's pad it out to 25 to be safe with existing GCN hardcoded sizes.
        padded_features = type_vec + extra_features + [0.0] * (25 - len(type_vec) - len(extra_features))
        
        return np.array(padded_features, dtype=np.float32)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "domain": self.domain,
            "problem_type": self.problem_type,
            "core_question": self.core_question,
            "given_conditions": self.given_conditions,
            "target_format": self.target_format,
            "difficulty": self.difficulty,
            "raw_task": self.raw_task,
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "StructuredMathGoal":
        """Create from dictionary."""
        return StructuredMathGoal(
            domain=data.get("domain", "math"),
            problem_type=data.get("problem_type", "unknown"),
            core_question=data.get("core_question", ""),
            given_conditions=data.get("given_conditions", []),
            target_format=data.get("target_format", "unknown"),
            difficulty=data.get("difficulty", 0.5),
            raw_task=data.get("raw_task", ""),
        )


# ================================ Goal Parser ================================

class MathGoalParser:
    """Parses math problems into structured math goals."""
    
    def __init__(self, llm_model: Optional[LLMCallable] = None):
        self.llm_model = llm_model
    
    def parse(self, task_main: str, task_description: str = "", 
              context: str = "", use_llm: bool = False) -> StructuredMathGoal:
        """
        Parse a math problem into a structured math goal.
        
        Args:
            task_main: The main task string (usually the problem statement)
            task_description: Additional task description
            context: Optional context
            use_llm: Whether to use LLM for parsing. Highly recommended for math.
        
        Returns:
            StructuredMathGoal object
        """
        full_task = f"{task_main}\n{task_description}".strip()
        
        # Math is hard to parse with regex, so we heavily favor LLM parsing if available.
        if use_llm and self.llm_model:
            return self._parse_with_llm(full_task, context)
        else:
            return self._parse_with_rules(full_task, task_main)
    
    def _parse_with_rules(self, full_task: str, raw_task: str) -> StructuredMathGoal:
        """Fallback Rule-based parsing."""
        lower_task = full_task.lower()
        
        problem_type = "algebra"
        if "triangle" in lower_task or "circle" in lower_task or "area" in lower_task:
            problem_type = "geometry"
        elif "probability" in lower_task or "dice" in lower_task:
            problem_type = "probability"
            
        core_question = "Find the answer"
        match = re.search(r'(?:find|calculate|what is)\s+(.*?\?)', lower_task)
        if match:
            core_question = match.group(0).strip()
            
        return StructuredMathGoal(
            problem_type=problem_type,
            core_question=core_question,
            raw_task=raw_task,
            difficulty=0.5
        )
    
    def _parse_with_llm(self, full_task: str, context: str) -> StructuredMathGoal:
        """LLM-based parsing (highly recommended for math)."""
        if not self.llm_model:
            return self._parse_with_rules(full_task, full_task)
        
        prompt = MATH_GOAL_PARSE_USER_PROMPT.format(
            task_description=full_task,
            context=context or "None"
        )
        
        try:
            response = self.llm_model(
                messages=[
                    Message("system", MATH_GOAL_PARSE_SYSTEM_PROMPT),
                    Message("user", prompt)
                ],
                temperature=0.1
            )
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                return StructuredMathGoal(
                    domain=data.get("domain", "math"),
                    problem_type=data.get("problem_type", "unknown"),
                    core_question=data.get("core_question", ""),
                    given_conditions=data.get("given_conditions", []),
                    target_format=data.get("target_format", "unknown"),
                    raw_task=full_task,
                    difficulty=0.5
                )
        except Exception as e:
            print(f"LLM parsing failed: {e}, falling back to rules")
        
        return self._parse_with_rules(full_task, full_task)
    
    def compute_similarity(self, goal1: StructuredMathGoal, goal2: StructuredMathGoal,
                          use_llm: bool = False) -> float:
        if use_llm and self.llm_model:
            return self._compute_similarity_llm(goal1, goal2)
        else:
            return self._compute_similarity_rules(goal1, goal2)
            
    def _compute_similarity_rules(self, goal1: StructuredMathGoal, goal2: StructuredMathGoal) -> float:
        score = 0.0
        if goal1.problem_type == goal2.problem_type:
            score += 0.4
            
        # Very crude word overlap
        q1_words = set(goal1.core_question.lower().split())
        q2_words = set(goal2.core_question.lower().split())
        if q1_words and q2_words:
            overlap = len(q1_words & q2_words) / max(len(q1_words | q2_words), 1)
            score += 0.6 * overlap
        return score

    def _compute_similarity_llm(self, goal1: StructuredMathGoal, goal2: StructuredMathGoal) -> float:
        """LLM-based similarity computation for math problems."""
        if not self.llm_model:
            return self._compute_similarity_rules(goal1, goal2)
        
        prompt = MATH_GOAL_SIMILARITY_USER_PROMPT.format(
            goal1=goal1.to_str(),
            goal2=goal2.to_str()
        )
        
        try:
            response = self.llm_model(
                messages=[
                    Message("system", MATH_GOAL_SIMILARITY_SYSTEM_PROMPT),
                    Message("user", prompt)
                ],
                temperature=0.1
            )
            match = re.search(r'\d+', response)
            if match:
                return int(match.group()) / 10.0
        except Exception as e:
            print(f"LLM similarity failed: {e}")
        
        return self._compute_similarity_rules(goal1, goal2)
