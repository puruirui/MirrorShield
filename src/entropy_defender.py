import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

from ..models.base_model import BaseModel
from ..utils.attention_utils import AttentionAnalyzer
from ..utils.text_processing import TextProcessor

logger = logging.getLogger(__name__)


class EntropyDefender:

    def __init__(self, config: Any, target_model: BaseModel):

        self.config = config
        self.target_model = target_model
        self.attention_analyzer = AttentionAnalyzer()
        self.text_processor = TextProcessor(target_model=target_model)

        logger.info("EntropyDefender initialized")

    def defend_input(self,
                     input_prompt: str,
                     mirrors: List[str]) -> Tuple[str, Dict[str, Any]]:

        logger.info("Starting entropy-based defense")

        if len(mirrors) < 2:
            logger.warning("Insufficient mirrors for comparison. Need at least 2 mirrors.")
            return "accept", {"warning": "insufficient_mirrors"}

        # Get first two mirrors for RIU calculation
        mirror1, mirror2 = mirrors[0], mirrors[1]

        # Calculate RIU
        riu_value = self._calculate_riu(input_prompt, mirror1, mirror2)

        defense_info = {
            "riu_value": riu_value,
            "threshold": self.config.riu_threshold,
            "mirrors_used": [mirror1, mirror2],
            "iterations": 0
        }

        # Make defense decision based on RIU threshold
        if riu_value >= self.config.riu_threshold:
            # Input is likely benign
            logger.info(f"Input accepted (RIU: {riu_value:.3f} >= {self.config.riu_threshold})")
            return "accept", defense_info
        else:
            # Input is potentially harmful - apply multiple-query guidance
            logger.info(f"Input flagged as risky (RIU: {riu_value:.3f} < {self.config.riu_threshold})")
            refined_prompt, iterations = self._apply_multiple_query_guidance(
                input_prompt, mirror1, mirror2
            )

            defense_info["iterations"] = iterations
            defense_info["refined_prompt"] = refined_prompt

            if refined_prompt != input_prompt:
                return "refined_prompt", defense_info
            else:
                return "reject", defense_info

    def _calculate_riu(self,
                       input_prompt: str,
                       mirror1: str,
                       mirror2: str) -> float:


        # Get attention weights for all inputs
        input_attention = self.target_model.get_attention_weights(input_prompt)
        mirror1_attention = self.target_model.get_attention_weights(mirror1)
        mirror2_attention = self.target_model.get_attention_weights(mirror2)

        # Extract encoder attention (primary attention for analysis)
        input_attn = self._extract_primary_attention(input_attention)
        mirror1_attn = self._extract_primary_attention(mirror1_attention)
        mirror2_attn = self._extract_primary_attention(mirror2_attention)

        # Compute attention entropy for each input
        input_entropy = self.attention_analyzer.compute_attention_entropy(input_attn)
        mirror1_entropy = self.attention_analyzer.compute_attention_entropy(mirror1_attn)
        mirror2_entropy = self.attention_analyzer.compute_attention_entropy(mirror2_attn)

        # Calculate RIU using the formula from the paper
        riu = self.attention_analyzer.compute_riu(
            input_entropy, mirror1_entropy, mirror2_entropy
        )

        logger.debug(f"Calculated RIU: {riu:.3f}")

        return riu

    def _extract_primary_attention(self, attention_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        if attention_output.get("encoder_attentions") is not None:
            attentions = attention_output["encoder_attentions"]
        elif attention_output.get("decoder_attentions") is not None:
            attentions = attention_output["decoder_attentions"]
        else:
            raise ValueError("No attention weights found in model output")

        # Use the last layer's attention
        return attentions[-1]

    def _apply_multiple_query_guidance(self,
                                       input_prompt: str,
                                       mirror1: str,
                                       mirror2: str) -> Tuple[str, int]:
        current_prompt = input_prompt
        iterations = 0

        logger.info("Applying multiple-query guidance")

        while iterations < self.config.max_iterations:
            iterations += 1

            # Generate multiple queries for refinement
            queries = self.text_processor.generate_multiple_queries(current_prompt)

            # Apply simplification using target model
            simplified_prompt = self.text_processor.simplify_text(current_prompt)

            # Recalculate RIU with simplified prompt
            new_riu = self._calculate_riu(simplified_prompt, mirror1, mirror2)

            logger.debug(f"Iteration {iterations}: RIU = {new_riu:.3f}")

            if new_riu >= self.config.riu_threshold:
                logger.info(f"Refinement successful after {iterations} iterations")
                return simplified_prompt, iterations

            current_prompt = simplified_prompt

        logger.warning(f"Refinement failed after {iterations} iterations")
        return current_prompt, iterations

    def analyze_input_risk(self,
                           input_prompt: str,
                           mirrors: List[str]) -> Dict[str, Any]:
        """Analyze risk level of input prompt based on RIU.

        Args:
            input_prompt: Input prompt to analyze
            mirrors: List of mirrors for comparison

        Returns:
            Risk analysis results
        """
        analysis = {
            "risk_level": "unknown",
            "confidence": 0.0,
            "recommendations": []
        }

        if len(mirrors) >= 2:
            # Calculate RIU
            riu = self._calculate_riu(input_prompt, mirrors[0], mirrors[1])
            analysis["riu_value"] = riu

            # Determine risk level based on RIU
            if riu >= self.config.riu_threshold:
                analysis["risk_level"] = "low"
                analysis["confidence"] = min(1.0, riu / self.config.riu_threshold)
            else:
                analysis["risk_level"] = "high"
                analysis["confidence"] = 1.0 - (riu / self.config.riu_threshold)

            # Generate recommendations based on RIU
            if analysis["risk_level"] == "high":
                analysis["recommendations"] = [
                    "Apply multiple-query guidance",
                    "Simplify the input prompt",
                    "Request clarification from user",
                    "Consider rejecting the input"
                ]

        return analysis
