import time
from typing import List, Dict, Any, Optional, Tuple
import logging

from .mirror_generator import MirrorGenerator
from .mirror_selector import MirrorSelector
from .entropy_defender import EntropyDefender
from ..models.base_model import BaseModel
from ..utils.evaluation_utils import EvaluationMetrics

logger = logging.getLogger(__name__)


class MirrorShield:

    def __init__(self, config: Any, target_model: Optional[BaseModel] = None):
        self.config = config

        if target_model is None:
            self.target_model = BaseModel(config.target_model_name)
        else:
            self.target_model = target_model

        self.mirror_generator = MirrorGenerator(config)
        self.mirror_selector = MirrorSelector(config)
        self.entropy_defender = EntropyDefender(config, self.target_model)

        # Initialize evaluation metrics
        self.evaluation_metrics = EvaluationMetrics()

        logger.info("MirrorShield system initialized")

    def defend_and_generate(self, input_prompt: str) -> Dict[str, Any]:
        start_time = time.time()

        logger.info(f"Processing input prompt: {input_prompt[:100]}...")

        # Step 1: Generate mirror candidates
        logger.info("Step 1: Generating mirror candidates")
        mirror_candidates = self.mirror_generator.generate_mirrors(input_prompt)

        if not mirror_candidates:
            logger.warning("No mirror candidates generated")
            return {
                "decision": "reject",
                "reason": "failed_mirror_generation",
                "response": "I cannot process this request safely.",
                "processing_time": time.time() - start_time
            }

        # Step 2: Select optimal mirrors
        logger.info("Step 2: Selecting optimal mirrors")
        selected_mirrors = self.mirror_selector.select_mirrors(
            input_prompt,
            mirror_candidates
        )

        if len(selected_mirrors) < 2:
            logger.warning("Insufficient valid mirrors selected")
            return {
                "decision": "reject",
                "reason": "insufficient_mirrors",
                "response": "I cannot process this request safely.",
                "processing_time": time.time() - start_time
            }

        # Step 3: Apply entropy defense
        logger.info("Step 3: Applying entropy defense")
        decision, defense_info = self.entropy_defender.defend_input(
            input_prompt,
            selected_mirrors
        )

        # Step 4: Generate response based on decision
        response_info = self._generate_response(decision, defense_info, input_prompt)

        # Record metrics
        processing_time = time.time() - start_time
        self.evaluation_metrics.record_defense_decision(
            input_prompt,
            defense_info.get("riu_value", 0.0),
            decision,
            processing_time
        )

        # Compile final results
        results = {
            "decision": decision,
            "response": response_info["response"],
            "confidence": response_info.get("confidence", 0.0),
            "mirror_candidates": mirror_candidates,
            "selected_mirrors": selected_mirrors,
            "defense_info": defense_info,
            "processing_time": processing_time,
            "metadata": {
                "num_mirror_candidates": len(mirror_candidates),
                "num_selected_mirrors": len(selected_mirrors),
                "riu_value": defense_info.get("riu_value", 0.0),
                "iterations": defense_info.get("iterations", 0)
            }
        }

        logger.info(f"Processing completed. Decision: {decision}, Time: {processing_time:.2f}s")

        return results

    def _generate_response(self,
                           decision: str,
                           defense_info: Dict[str, Any],
                           original_prompt: str) -> Dict[str, Any]:
        if decision == "accept":
            # Generate normal response
            response = self.target_model.generate(
                original_prompt,
                max_length=self.config.max_mirror_length,
                temperature=self.config.temperature
            )[0]

            return {
                "response": response,
                "confidence": 0.9,
                "safe": True
            }

        elif decision == "refined_prompt":
            # Generate response for refined prompt
            refined_prompt = defense_info.get("refined_prompt", original_prompt)
            response = self.target_model.generate(
                refined_prompt,
                max_length=self.config.max_mirror_length,
                temperature=self.config.temperature
            )[0]

            return {
                "response": response,
                "confidence": 0.7,
                "safe": True,
                "note": "Response generated from refined prompt"
            }

        else:  # reject
            return {
                "response": "I cannot provide a response to this request as it may violate safety guidelines.",
                "confidence": 0.9,
                "safe": True,
                "rejected": True
            }

    def analyze_prompt_risk(self, input_prompt: str) -> Dict[str, Any]:
        
        logger.info("Analyzing prompt risk")

        # Generate minimal mirrors for analysis
        mirror_candidates = self.mirror_generator.generate_mirrors(
            input_prompt,
            num_candidates=3
        )

        if len(mirror_candidates) >= 2:
            selected_mirrors = self.mirror_selector.select_mirrors(
                input_prompt,
                mirror_candidates,
                max_selected=2
            )

            if len(selected_mirrors) >= 2:
                analysis = self.entropy_defender.analyze_input_risk(
                    input_prompt,
                    selected_mirrors
                )
            else:
                analysis = {"risk_level": "unknown", "reason": "insufficient_mirrors"}
        else:
            analysis = {"risk_level": "unknown", "reason": "failed_mirror_generation"}

        return analysis

    def get_performance_metrics(self) -> Dict[str, Any]:

        return self.evaluation_metrics.get_performance_summary()

    def export_metrics(self, filepath: str):

        self.evaluation_metrics.export_results(filepath)

    def reset_metrics(self):

        self.evaluation_metrics.reset_metrics()

    def fine_tune_mirror_generator(self, training_data: List[Dict[str, str]]):

        logger.info("Fine-tuning mirror generator")
        self.mirror_generator.fine_tune_model(training_data)
        logger.info("Mirror generator fine-tuning completed")
