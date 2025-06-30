import spacy
import nltk
from textblob import TextBlob
from typing import Dict, List, Tuple, Any, Optional
import re
import logging

logger = logging.getLogger(__name__)


class ConstraintValidator:
    """Validator for checking mirror constraints."""

    def __init__(self):
        """Initialize constraint validator."""
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
        """Validate length constraint.

        Args:
            text: Text to validate
            target_length: Target length in tokens
            tolerance: Tolerance ratio (e.g., 0.2 for ±20%)

        Returns:
            True if length constraint is satisfied
        """
        tokens = self._tokenize(text)
        actual_length = len(tokens)

        min_length = int(target_length * (1 - tolerance))
        max_length = int(target_length * (1 + tolerance))

        return min_length <= actual_length <= max_length

    def validate_syntax_constraint(self,
                                   text: str,
                                   target_pos_sequence: List[str]) -> bool:
        """Validate syntax constraint using POS tagging.

        Args:
            text: Text to validate
            target_pos_sequence: Target POS sequence

        Returns:
            True if syntax constraint is satisfied
        """
        if self.nlp is None:
            logger.warning("spaCy not available, skipping syntax validation")
            return True

        doc = self.nlp(text)
        actual_pos = [token.pos_ for token in doc]

        # Simple matching - can be enhanced with fuzzy matching
        return actual_pos == target_pos_sequence

    def validate_sentiment_constraint(self,
                                      text: str,
                                      target_sentiment: str = "positive",
                                      threshold: float = 0.1) -> bool:
        """Validate sentiment constraint.

        Args:
            text: Text to validate
            target_sentiment: Target sentiment (positive, negative, neutral)
            threshold: Threshold for sentiment classification

        Returns:
            True if sentiment constraint is satisfied
        """
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
        """Extract constraints from input text.

        Args:
            text: Input text

        Returns:
            Dictionary containing extracted constraints
        """
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
        """Validate all constraints for a candidate text.

        Args:
            candidate_text: Candidate mirror text
            reference_constraints: Reference constraints

        Returns:
            Dictionary with validation results for each constraint
        """
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
        """Tokenize text into tokens.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Simple whitespace tokenization - can be enhanced
        return text.strip().split()

    def generate_constraint_ranges(self,
                                   reference_length: int,
                                   lambda_param: float = 1.0) -> Tuple[int, int]:
        """Generate length constraint ranges as described in paper.

        Args:
            reference_length: Reference text length
            lambda_param: Lambda parameter for range calculation

        Returns:
            Tuple of (min_length, max_length)
        """
        # As described in paper: λn to λ(n+1) words
        n = int(reference_length / lambda_param)
        min_length = int(lambda_param * n)
        max_length = int(lambda_param * (n + 1))

        return min_length, max_length