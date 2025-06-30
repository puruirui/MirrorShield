import spacy
import nltk
from textblob import TextBlob
from typing import Dict, List, Tuple, Any, Optional
import re
import logging

logger = logging.getLogger(__name__)


class ConstraintValidator:

    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None

        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def validate_length_constraint(self,
                                   text: str,
                                   target_length: int,
                                   tolerance: float = 0.2) -> bool:
        tokens = self._tokenize(text)
        actual_length = len(tokens)

        min_length = int(target_length * (1 - tolerance))
        max_length = int(target_length * (1 + tolerance))

        return min_length <= actual_length <= max_length

    def validate_syntax_constraint(self,
                                   text: str,
                                   target_pos_sequence: List[str]) -> bool:
        if self.nlp is None:
            logger.warning("spaCy not available, skipping syntax validation")
            return True

        doc = self.nlp(text)
        actual_pos = [token.pos_ for token in doc]

        return actual_pos == target_pos_sequence

    def validate_sentiment_constraint(self,
                                      text: str,
                                      target_sentiment: str = "positive",
                                      threshold: float = 0.1) -> bool:

        blob = TextBlob(text)
        polarity = blob.sentiment.polarity

        if target_sentiment.lower() == "positive":
            return polarity >= threshold
        elif target_sentiment.lower() == "negative":
            return polarity <= -threshold
        elif target_sentiment.lower() == "neutral":
            return -threshold < polarity < threshold
        else:
            # For "non-negative" (positive or neutral)
            return polarity >= -threshold

    def extract_constraints_from_text(self, text: str) -> Dict[str, Any]:

        constraints = {}

        # Length constraint
        tokens = self._tokenize(text)
        constraints["length"] = len(tokens)

        # Syntax constraint (POS sequence)
        if self.nlp is not None:
            doc = self.nlp(text)
            constraints["syntax"] = [token.pos_ for token in doc]

        # Sentiment constraint
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            constraints["sentiment"] = "positive"
        elif polarity < -0.1:
            constraints["sentiment"] = "negative"
        else:
            constraints["sentiment"] = "neutral"

        return constraints

    def validate_all_constraints(self,
                                 candidate_text: str,
                                 reference_constraints: Dict[str, Any]) -> Dict[str, bool]:

        results = {}

        # Length validation
        if "length" in reference_constraints:
            results["length"] = self.validate_length_constraint(
                candidate_text,
                reference_constraints["length"]
            )

        # Syntax validation
        if "syntax" in reference_constraints:
            results["syntax"] = self.validate_syntax_constraint(
                candidate_text,
                reference_constraints["syntax"]
            )

        # Sentiment validation
        if "sentiment" in reference_constraints:
            results["sentiment"] = self.validate_sentiment_constraint(
                candidate_text,
                reference_constraints["sentiment"]
            )

        return results

    def _tokenize(self, text: str) -> List[str]:

        # Simple whitespace tokenization - can be enhanced
        return text.strip().split()

    def generate_constraint_ranges(self,
                                   reference_length: int,
                                   lambda_param: float = 1.0) -> Tuple[int, int]:

        n = int(reference_length / lambda_param)
        min_length = int(lambda_param * n)
        max_length = int(lambda_param * (n + 1))

        return min_length, max_length
