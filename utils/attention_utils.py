import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import math
import logging

logger = logging.getLogger(__name__)


class AttentionAnalyzer:

    def __init__(self):
        pass

    def compute_attention_entropy(self, attention_weights: torch.Tensor) -> torch.Tensor:

        mean_attention = torch.mean(attention_weights, dim=1)  # [batch_size, seq_len, seq_len]

        epsilon = 1e-8
        log_attention = torch.log(mean_attention + epsilon)
        entropy = -torch.sum(mean_attention * log_attention, dim=-1)  # [batch_size, seq_len]

        return entropy

    def compute_information_gain(self,
                                 entropy1: torch.Tensor,
                                 entropy2: torch.Tensor) -> float:

        diff = torch.abs(entropy1 - entropy2)
        ig = torch.mean(diff).item()

        return ig

    def compute_riu(self,
                    input_entropy: torch.Tensor,
                    mirror1_entropy: torch.Tensor,
                    mirror2_entropy: torch.Tensor) -> float:


        ig_current = self.compute_information_gain(input_entropy, mirror1_entropy)

        ig_reference = self.compute_information_gain(mirror1_entropy, mirror2_entropy)

        if ig_current == 0:
            return float('inf')  

        riu = ig_reference / ig_current

        return riu

    def extract_attention_from_model_output(self,
                                            model_output: Dict[str, torch.Tensor],
                                            layer_idx: int = -1) -> torch.Tensor:

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

        mean_attention = torch.mean(attention_weights, dim=1).squeeze()

        attention_variance = torch.var(mean_attention, dim=-1)
        attention_max = torch.max(mean_attention, dim=-1)[0]

        token_importance = torch.sum(mean_attention, dim=0)

        results = {
            "attention_variance": attention_variance.mean().item(),
            "max_attention": attention_max.mean().item(),
            "token_importance": token_importance.tolist(),
            "entropy": self.compute_attention_entropy(attention_weights.unsqueeze(0)).squeeze().mean().item()
        }

        return results
