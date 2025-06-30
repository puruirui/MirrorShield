import time
import json
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Evaluation metrics for MirrorShield performance."""

    def __init__(self):
        """Initialize evaluation metrics."""
        self.reset_metrics()

    def reset_metrics(self):
        """Reset all metrics."""
        self.generation_times = []
        self.defense_decisions = []
        self.riu_values = []
        self.constraint_validation_results = []

    def calculate_asr(self, responses: List[str], harmful_labels: List[bool]) -> float:
        """Calculate Attack Success Rate (ASR).

        Args:
            responses: List of model responses
            harmful_labels: List indicating if each response is harmful

        Returns:
            ASR value (0-1)
        """
        if len(responses) != len(harmful_labels):
            raise ValueError("Responses and labels must have same length")

        total_attacks = len(responses)
        successful_attacks = sum(harmful_labels)

        return successful_attacks / total_attacks if total_attacks > 0 else 0.0

    def calculate_atgr(self,
                       defense_times: List[float],
                       baseline_times: List[float]) -> float:
        """Calculate Average Token Generation Time Ratio (ATGR).

        Args:
            defense_times: Generation times with defense
            baseline_times: Generation times without defense

        Returns:
            ATGR value
        """
        avg_defense_time = sum(defense_times) / len(defense_times)
        avg_baseline_time = sum(baseline_times) / len(baseline_times)

        return avg_defense_time / avg_baseline_time if avg_baseline_time > 0 else float('inf')

    def calculate_win_rate(self,
                           our_responses: List[str],
                           baseline_responses: List[str],
                           evaluator_func: callable) -> float:
        """Calculate win rate against baseline.

        Args:
            our_responses: Our model responses
            baseline_responses: Baseline model responses
            evaluator_func: Function to evaluate response quality

        Returns:
            Win rate (0-1)
        """
        wins = 0
        total = len(our_responses)

        for our_resp, baseline_resp in zip(our_responses, baseline_responses):
            if evaluator_func(our_resp, baseline_resp):
                wins += 1

        return wins / total if total > 0 else 0.0

    def record_defense_decision(self,
                                input_prompt: str,
                                riu_value: float,
                                decision: str,
                                processing_time: float):
        """Record a defense decision for analysis.

        Args:
            input_prompt: Input prompt
            riu_value: Calculated RIU value
            decision: Defense decision (accept/reject/refine)
            processing_time: Time taken for processing
        """
        record = {
            "input_prompt": input_prompt,
            "riu_value": riu_value,
            "decision": decision,
            "processing_time": processing_time,
            "timestamp": time.time()
        }

        self.defense_decisions.append(record)
        self.riu_values.append(riu_value)
        self.generation_times.append(processing_time)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics.

        Returns:
            Dictionary with performance metrics
        """
        if not self.generation_times:
            return {"error": "No data recorded"}

        avg_time = sum(self.generation_times) / len(self.generation_times)
        avg_riu = sum(self.riu_values) / len(self.riu_values) if self.riu_values else 0

        decisions = [d["decision"] for d in self.defense_decisions]
        decision_counts = {
            "accept": decisions.count("accept"),
            "reject": decisions.count("reject"),
            "refine": decisions.count("refine")
        }

        return {
            "total_processed": len(self.defense_decisions),
            "average_processing_time": avg_time,
            "average_riu": avg_riu,
            "decision_distribution": decision_counts,
            "riu_statistics": {
                "min": min(self.riu_values) if self.riu_values else 0,
                "max": max(self.riu_values) if self.riu_values else 0,
                "mean": avg_riu
            }
        }

    def export_results(self, filepath: str):
        """Export evaluation results to file.

        Args:
            filepath: Path to save results
        """
        results = {
            "summary": self.get_performance_summary(),
            "detailed_decisions": self.defense_decisions,
            "constraint_validations": self.constraint_validation_results
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results exported to {filepath}")