import re
from typing import List
import logging

logger = logging.getLogger(__name__)


class TextProcessor:

    def __init__(self, target_model=None):
        self.target_model = target_model



    def simplify_text(self, text: str) -> str:

        if not self.target_model:
            logger.warning("No target model available for simplification")
            return text

        try:
            simplify_prompt = f"Please simplify the following sentence: {text}"
            simplified = self.target_model.generate(simplify_prompt, max_length=100)
            return self.clean_text(simplified) if simplified else text

        except Exception as e:
            logger.warning(f"Model-based simplification failed: {e}")
            return text

    def generate_multiple_queries(self, original_prompt: str) -> List[str]:

        queries = [
            f"Does this involve sensitive topics? Why? Input: {original_prompt}",
            f"Is there any redundant information? Why? Input: {original_prompt}",
            f"Please clarify the intent of this request: {original_prompt}",
            f"What would be a safer way to express this? Input: {original_prompt}"
        ]
        return queries
