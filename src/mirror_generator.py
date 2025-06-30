import torch
from typing import List, Dict, Any, Optional, Tuple
import logging

from ..models.instruction_tuned_model import InstructionTunedModel
from ..utils.constraint_utils import ConstraintValidator
from ..utils.text_processing import TextProcessor

logger = logging.getLogger(__name__)

class MirrorGenerator:
    """Generate mirrors with syntactic similarity and semantic safety."""
    
    def __init__(self, config: Any):
        """Initialize mirror generator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.constraint_validator = ConstraintValidator()
        self.text_processor = TextProcessor()
        
        # Initialize instruction-tuned model
        self.instruction_model = InstructionTunedModel(
            config.base_model_name,
            device=config.device if hasattr(config, 'device') else None
        )
        
        logger.info("MirrorGenerator initialized")
    
    def generate_mirrors(self, 
                        input_prompt: str, 
                        num_candidates: int = None) -> List[str]:
        """Generate mirror candidates for input prompt.
        
        Args:
            input_prompt: Original input prompt
            num_candidates: Number of candidate mirrors to generate
            
        Returns:
            List of generated mirror candidates
        """
        if num_candidates is None:
            num_candidates = self.config.num_candidate_mirrors
        
        logger.info(f"Generating {num_candidates} mirror candidates for input prompt")
        
        # Extract constraints from input prompt
        constraints = self._extract_input_constraints(input_prompt)
        
        # Generate candidate mirrors using instruction-tuned model
        candidates = self.instruction_model.generate_with_constraints(
            input_prompt,
            constraints,
            num_candidates=num_candidates
        )
        
        # Clean and post-process candidates
        processed_candidates = []
        for candidate in candidates:
            processed = self.text_processor.clean_text(candidate)
            if processed and processed != input_prompt:  # Avoid duplicates
                processed_candidates.append(processed)
        
        logger.info(f"Generated {len(processed_candidates)} valid mirror candidates")
        
        return processed_candidates
    
    def _extract_input_constraints(self, input_prompt: str) -> Dict[str, Any]:
        """Extract constraints from input prompt.
        
        Args:
            input_prompt: Input prompt to analyze
            
        Returns:
            Dictionary of extracted constraints
        """
        constraints = self.constraint_validator.extract_constraints_from_text(input_prompt)
        
        # Ensure sentiment is non-negative for mirrors (as per paper)
        if constraints.get("sentiment") == "negative":
            constraints["sentiment"] = "neutral"
        
        # Adjust length constraint with tolerance
        if "length" in constraints:
            length = constraints["length"]
            min_len = max(1, int(length * (1 - self.config.length_tolerance)))
            max_len = int(length * (1 + self.config.length_tolerance))
            constraints["length"] = (min_len, max_len)
        
        logger.debug(f"Extracted constraints: {constraints}")
        
        return constraints
    
    def fine_tune_model(self, training_data: List[Dict[str, str]]) -> None:
        """Fine-tune the instruction model on constraint data.
        
        Args:
            training_data: Training data for fine-tuning
        """
        logger.info("Fine-tuning instruction model...")
        
        self.instruction_model.fine_tune(
            training_data,
            self.config,
            output_dir=self.config.instruction_tuned_model_path
        )
        
        logger.info("Model fine-tuning completed")
    
    def load_fine_tuned_model(self, model_path: str) -> None:
        """Load a pre-fine-tuned model.
        
        Args:
            model_path: Path to the fine-tuned model
        """
        logger.info(f"Loading fine-tuned model from {model_path}")
        
        # Reinitialize with fine-tuned model
        self.instruction_model = InstructionTunedModel(model_path)
        self.instruction_model.is_fine_tuned = True
        
        logger.info("Fine-tuned model loaded successfully")
