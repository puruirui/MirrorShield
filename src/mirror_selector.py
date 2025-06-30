import openai
from typing import List, Dict, Any
import logging

from ..utils.constraint_utils import ConstraintValidator

logger = logging.getLogger(__name__)


class MirrorSelector:
    """Select optimal mirrors based on constraint satisfaction."""

    def __init__(self, config: Any):
        self.config = config
        self.constraint_validator = ConstraintValidator()

        if hasattr(config, 'openai_api_key') and config.openai_api_key:
            openai.api_key = config.openai_api_key
        else:
            raise ValueError("OpenAI API key required for GPT-4o mirror evaluation")

        logger.info("MirrorSelector initialized")

    def select_mirrors(self,
                       input_prompt: str,
                       candidate_mirrors: List[str],
                       max_selected: int = 5) -> List[str]:
        logger.info(f"Selecting mirrors from {len(candidate_mirrors)} candidates")

        reference_constraints = self.constraint_validator.extract_constraints_from_text(input_prompt)
        selected_mirrors = []

        for candidate in candidate_mirrors:
            if len(selected_mirrors) >= max_selected:
                break

            evaluation = self._evaluate_candidate_with_gpt4o(candidate, reference_constraints)

            if self._passes_all_constraints(evaluation):
                selected_mirrors.append(candidate)

        logger.info(f"Selected {len(selected_mirrors)} mirrors")
        return selected_mirrors

    def _evaluate_candidate_with_gpt4o(self,
                                       candidate: str,
                                       reference_constraints: Dict[str, Any]) -> Dict[str, bool]:
        """Evaluate candidate using GPT-4o classifier."""

        prompt = self._construct_evaluation_prompt(candidate, reference_constraints)

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a constraint evaluation assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=200
            )

            evaluation_text = response.choices[0].message.content
            return self._parse_llm_evaluation(evaluation_text)

        except Exception as e:
            logger.error(f"GPT-4o evaluation failed: {e}")
            return {"length": False, "syntax": False, "sentiment": False}

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
        """Parse GPT-4o evaluation response."""

        evaluation = {
            "length": False,
            "syntax": False,
            "sentiment": False
        }

        lines = evaluation_text.lower().split('\n')

        for line in lines:
            if "length consistency:" in line:
                evaluation["length"] = "true" in line
            elif "syntax consistency:" in line:
                evaluation["syntax"] = "true" in line
            elif "sentiment consistency:" in line:
                evaluation["sentiment"] = "true" in line

        return evaluation

    def _passes_all_constraints(self, evaluation: Dict[str, bool]) -> bool:
        """Check if candidate passes all three constraints."""
        return all(evaluation.values())
