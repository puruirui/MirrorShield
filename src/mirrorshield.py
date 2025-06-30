import time
from typing import List, Dict, Any, Optional, Tuple
import logging

from .mirror_generator import MirrorGenerator
from .mirror_selector import MirrorSelector
from .entropy_defender import EntropyDefender
from ..models.base_model import BaseModel

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

        logger.info("MirrorShield system initialized")

    def defend(self, input_prompt: str) -> Tuple[str, Dict[str, Any]]:
        start_time = time.time()
        logger.info(f"Processing input prompt: {input_prompt[:100]}...")

        # Step 1: Generate mirror candidates
        logger.info("Step 1: Generating mirror candidates")
        mirror_candidates = self.mirror_generator.generate_mirrors(input_prompt)

        if not mirror_candidates:
            logger.warning("No mirror candidates generated")
            defense_info = {
                "decision": "reject",
                "reason": "failed_mirror_generation",
                "processing_time": time.time() - start_time
            }
            return "I cannot process this request safely.", defense_info

        # Step 2: Select optimal mirrors
        logger.info("Step 2: Selecting optimal mirrors")
        selected_mirrors = self.mirror_selector.select_mirrors(
            input_prompt,
            mirror_candidates
        )

        if len(selected_mirrors) < 2:
            logger.warning("Insufficient valid mirrors selected")
            defense_info = {
                "decision": "reject",
                "reason": "insufficient_mirrors",
                "processing_time": time.time() - start_time
            }
            return "I cannot process this request safely.", defense_info

        # Step 3: Apply entropy defense
        logger.info("Step 3: Applying entropy defense")
        decision, defense_info = self.entropy_defender.defend_input(
            input_prompt,
            selected_mirrors
        )

        # Step 4: Generate response based on decision
        response = self._generate_response(decision, defense_info, input_prompt)

        # Add metadata to defense_info
        defense_info.update({
            "processing_time": time.time() - start_time,
            "mirror_candidates": mirror_candidates,
            "selected_mirrors": selected_mirrors,
            "num_mirror_candidates": len(mirror_candidates),
            "num_selected_mirrors": len(selected_mirrors)
        })

        logger.info(f"Processing completed. Decision: {decision}, Time: {defense_info['processing_time']:.2f}s")

        return response, defense_info

    def _generate_response(self,
                           decision: str,
                           defense_info: Dict[str, Any],
                           original_prompt: str) -> str:

        if decision == "accept":
            # Generate normal response
            response = self.target_model.generate(
                original_prompt,
                max_length=self.config.max_mirror_length,
                temperature=self.config.temperature
            )
            return response

        elif decision == "refined_prompt":
            # Generate response for refined prompt
            refined_prompt = defense_info.get("refined_prompt", original_prompt)
            response = self.target_model.generate(
                refined_prompt,
                max_length=self.config.max_mirror_length,
                temperature=self.config.temperature
            )
            return response

        else:  # reject
            return "I cannot provide a response to this request as it may violate safety guidelines."

    def analyze_prompt_risk(self, input_prompt: str) -> Dict[str, Any]:

        logger.info("Analyzing prompt risk")

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
