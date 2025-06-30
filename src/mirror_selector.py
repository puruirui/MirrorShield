import openai
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
import time

from ..utils.constraint_utils import ConstraintValidator
from ..utils.text_processing import TextProcessor

logger = logging.getLogger(__name__)


class MirrorSelector:
    """Select optimal mirrors based on constraint satisfaction."""

    def __init__(self, config: Any):
        self.config = config
        self.constraint_validator = ConstraintValidator()
        self.text_processor = TextProcessor()

        if hasattr(config, 'openai_api_key') and config.openai_api_key:
            openai.api_key = config.openai_api_key
            self.use_llm_classifier = True
        else:
            logger.warning("OpenAI API key not provided. Using rule-based classification.")
            self.use_llm_classifier = False

        logger.info("MirrorSelector initialized")

    def select_mirrors(self,
                       input_prompt: str,
                       candidate_mirrors: List[str],
                       max_selected: int = 5) -> List[str]:
        logger.info(f"Selecting mirrors from {len(candidate_mirrors)} candidates")

        reference_constraints = self.constraint_validator.extract_constraints_from_text(input_prompt)

        # Evaluate each candidate
        evaluated_candidates = []

        for candidate in candidate_mirrors:
            evaluation = self._evaluate_candidate(candidate, reference_constraints)
            evaluated_candidates.append({
                "text": candidate,
                "evaluation": evaluation,
                "score": self._calculate_score(evaluation)
            })

        # Sort by score and select top candidates
        evaluated_candidates.sort(key=lambda x: x["score"], reverse=True)

        # Select candidates that pass all constraints
        selected_mirrors = []
        for candidate_info in evaluated_candidates:
            if (self._passes_all_constraints(candidate_info["evaluation"]) and
                    len(selected_mirrors) < max_selected):
                selected_mirrors.append(candidate_info["text"])

        logger.info(f"Selected {len(selected_mirrors)} mirrors")

        return selected_mirrors

    def _evaluate_candidate(self,
                            candidate: str,
                            reference_constraints: Dict[str, Any]) -> Dict[str, Any]:
        if self.use_llm_classifier:
            return self._llm_based_evaluation(candidate, reference_constraints)
        else:
            return self._rule_based_evaluation(candidate, reference_constraints)

    def _llm_based_evaluation(self,
                              candidate: str,
                              reference_constraints: Dict[str, Any]) -> Dict[str, Any]:

        # Construct evaluation prompt based on paper's template
        prompt = self._construct_evaluation_prompt(candidate, reference_constraints)

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a constraint evaluation assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )

            # Parse LLM response
            evaluation_text = response.choices[0].message.content
            evaluation = self._parse_llm_evaluation(evaluation_text)

        except Exception as e:
            logger.warning(f"LLM evaluation failed: {e}. Falling back to rule-based evaluation.")
            evaluation = self._rule_based_evaluation(candidate, reference_constraints)

        return evaluation

    def _rule_based_evaluation(self,
                               candidate: str,
                               reference_constraints: Dict[str, Any]) -> Dict[str, Any]:

        return self.constraint_validator.validate_all_constraints(
            candidate,
            reference_constraints
        )

    def _construct_evaluation_prompt(self,
                                     candidate: str,
                                     reference_constraints: Dict[str, Any]) -> str:

        prompt = """You will act as a classifier to evaluate a given candidate sentence against three specific criteria: Length Consistency, Syntax Consistency, and Sentiment Consistency. For each criterion, you must determine whether the sentence satisfies the requirement and respond with either "True" or "False".

The evaluation must be based on the following criteria:
• Length Consistency: The number of tokens in the candidate sentence must match the token count of the original input prompt (within reasonable tolerance).
• Syntax Consistency: The candidate sentence must adhere to a similar grammatical structure as the original prompt.
• Sentiment Consistency: The candidate sentence must exhibit positive or neutral sentiment without any harmful, offensive, or ambiguous language.

Input Information:
• Candidate Sentence: "{candidate}"
• Reference Length: {ref_length} words
• Reference Syntax: {ref_syntax}
• Required Sentiment: Non-negative (positive or neutral)

Output Format:
• Length Consistency: True/False
• Syntax Consistency: True/False
• Sentiment Consistency: True/False"""

        ref_length = reference_constraints.get("length", "unknown")
        ref_syntax = reference_constraints.get("syntax", "unknown")

        return prompt.format(
            candidate=candidate,
            ref_length=ref_length,
            ref_syntax=ref_syntax
        )

    def _parse_llm_evaluation(self, evaluation_text: str) -> Dict[str, bool]:
        evaluation = {
            "length": False,
            "syntax": False,
            "sentiment": False
        }

        # Simple parsing - can be enhanced with more robust parsing
        lines = evaluation_text.lower().split('\n')

        for line in lines:
            if "length consistency:" in line:
                evaluation["length"] = "true" in line
            elif "syntax consistency:" in line:
                evaluation["syntax"] = "true" in line
            elif "sentiment consistency:" in line:
                evaluation["sentiment"] = "true" in line

        return evaluation

    def _calculate_score(self, evaluation: Dict[str, bool]) -> float:
        # Weight constraints according to paper's findings
        weights = {
            "length": 0.5,  # Most important according to ablation study
            "syntax": 0.3,  # Important for structural similarity
            "sentiment": 0.2  # Least impact according to paper
        }

        score = 0.0
        for constraint, passed in evaluation.items():
            if constraint in weights and passed:
                score += weights[constraint]

        return score

    def _passes_all_constraints(self, evaluation: Dict[str, bool]) -> bool:
        return all(evaluation.values())
