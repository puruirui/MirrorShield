import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import math
import logging

logger = logging.getLogger(__name__)


class AttentionAnalyzer:
    """Analyzer for computing attention entropy and related metrics."""

    def __init__(self):
        """Initialize attention analyzer."""
        pass

    def compute_attention_entropy(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Compute attention entropy for each token.

        Args:
            attention_weights: Attention weights tensor [batch_size, num_heads, seq_len, seq_len]

        Returns:
            Attention entropy for each token [batch_size, seq_len]
        """
        # Average across attention heads (Equation 3 in paper)
        mean_attention = torch.mean(attention_weights, dim=1)  # [batch_size, seq_len, seq_len]

        # Compute entropy for each token (Equation 4 in paper)
        # H_i = -∑_j α̅_{i,j} log α̅_{i,j}
        epsilon = 1e-8  # Small constant to avoid log(0)
        log_attention = torch.log(mean_attention + epsilon)
        entropy = -torch.sum(mean_attention * log_attention, dim=-1)  # [batch_size, seq_len]

        return entropy

    def compute_information_gain(self,
                                 entropy1: torch.Tensor,
                                 entropy2: torch.Tensor) -> float:
        """Compute Information Gain between two entropy distributions.

        Args:
            entropy1: First entropy distribution
            entropy2: Second entropy distribution

        Returns:
            Information gain value
        """
        # |IG| = (1/ds) * ∑_i |H_i^1 - H_i^2| (Equation 5 in paper)
        diff = torch.abs(entropy1 - entropy2)
        ig = torch.mean(diff).item()

        return ig

    def compute_riu(self,
                    input_entropy: torch.Tensor,
                    mirror1_entropy: torch.Tensor,
                    mirror2_entropy: torch.Tensor) -> float:
        """Compute Relative Input Uncertainty (RIU).

        Args:
            input_entropy: Entropy of input prompt
            mirror1_entropy: Entropy of first mirror
            mirror2_entropy: Entropy of second mirror

        Returns:
            RIU value
        """
        # Compute IG_current = |H_input - H_mirror1|
        ig_current = self.compute_information_gain(input_entropy, mirror1_entropy)

        # Compute IG_reference = |H_mirror1 - H_mirror2|
        ig_reference = self.compute_information_gain(mirror1_entropy, mirror2_entropy)

        # RIU = |IG_reference| / |IG_current| (Equation 6 in paper)
        if ig_current == 0:
            return float('inf')  # Avoid division by zero

        riu = ig_reference / ig_current

        return riu

    def extract_attention_from_model_output(self,
                                            model_output: Dict[str, torch.Tensor],
                                            layer_idx: int = -1) -> torch.Tensor:
        """Extract attention weights from model output.

        Args:
            model_output: Model output containing attention weights
            layer_idx: Layer index to extract attention from (-1 for last layer)

        Returns:
            Attention weights tensor
        """
        if "encoder_attentions" in model_output and model_output["encoder_attentions"] is not None:
            attentions = model_output["encoder_attentions"]
        elif "decoder_attentions" in model_output and model_output["decoder_attentions"] is not None:
            attentions = model_output["decoder_attentions"]
        else:
            raise ValueError("No attention weights found in model output")

        # Select layer
        if layer_idx == -1:
            layer_idx = len(attentions) - 1

        return attentions[layer_idx]

    def analyze_attention_patterns(self,
                                   attention_weights: torch.Tensor,
                                   tokens: List[str]) -> Dict[str, float]:
        """Analyze attention patterns and provide insights.

        Args:
            attention_weights: Attention weights
            tokens: List of tokens

        Returns:
            Dictionary with attention analysis results
        """
        # Compute various attention statistics
        mean_attention = torch.mean(attention_weights, dim=1).squeeze()

        # Attention concentration (how focused is attention)
        attention_variance = torch.var(mean_attention, dim=-1)
        attention_max = torch.max(mean_attention, dim=-1)[0]

        # Token-level statistics
        token_importance = torch.sum(mean_attention, dim=0)

        results = {
            "attention_variance": attention_variance.mean().item(),
            "max_attention": attention_max.mean().item(),
            "token_importance": token_importance.tolist(),
            "entropy": self.compute_attention_entropy(attention_weights.unsqueeze(0)).squeeze().mean().item()
        }

        return results