import torch
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
from typing import List, Dict, Any, Optional
import json
import logging

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class InstructionTunedModel(BaseModel):
    """Instruction-tuned model for generating mirrors with constraints."""

    def __init__(self, model_name: str, device: Optional[str] = None):
        """Initialize instruction-tuned model."""
        super().__init__(model_name, device)
        self.is_fine_tuned = False

    def fine_tune(self,
                  training_data: List[Dict[str, str]],
                  config: Any,
                  output_dir: str = "./models/instruction_tuned") -> None:
        """Fine-tune model on constraint-text pairs.

        Args:
            training_data: List of constraint-text pairs
            config: Training configuration
            output_dir: Output directory for fine-tuned model
        """
        logger.info(f"Fine-tuning model with {len(training_data)} examples")

        # Prepare dataset
        dataset = Dataset.from_list(training_data)

        def preprocess_function(examples):
            inputs = [ex["instruction"] for ex in examples]
            targets = [ex["output"] for ex in examples]

            model_inputs = self.tokenizer(
                inputs,
                max_length=512,
                truncation=True,
                padding=True
            )

            labels = self.tokenizer(
                targets,
                max_length=512,
                truncation=True,
                padding=True
            )

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_dataset = dataset.map(preprocess_function, batched=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            warmup_steps=config.warmup_steps,
            logging_steps=100,
            save_steps=500,
            evaluation_strategy="no",
            save_strategy="steps",
            load_best_model_at_end=False,
            report_to=None
        )

        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=-100
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        # Train
        trainer.train()
        trainer.save_model()
        self.is_fine_tuned = True

        logger.info(f"Model fine-tuned and saved to {output_dir}")

    def generate_with_constraints(self,
                                  input_text: str,
                                  constraints: Dict[str, Any],
                                  num_candidates: int = 5) -> List[str]:
        """Generate text with specified constraints.

        Args:
            input_text: Original input text
            constraints: Dictionary of constraints (length, syntax, sentiment)
            num_candidates: Number of candidate mirrors to generate

        Returns:
            List of generated mirror candidates
        """
        # Construct instruction prompt
        instruction = self._build_instruction_prompt(input_text, constraints)

        # Generate candidates
        candidates = self.generate(
            instruction,
            num_return_sequences=num_candidates,
            max_length=512,
            temperature=0.8,
            top_p=0.9
        )

        return candidates

    def _build_instruction_prompt(self, input_text: str, constraints: Dict[str, Any]) -> str:
        """Build instruction prompt for constraint-based generation.

        Args:
            input_text: Original input text
            constraints: Constraint dictionary

        Returns:
            Formatted instruction prompt
        """
        constraint_parts = []

        # Length constraint
        if "length" in constraints:
            min_len, max_len = constraints["length"]
            constraint_parts.append(f"has {min_len} to {max_len} words")

        # Syntax constraint
        if "syntax" in constraints:
            pos_sequence = constraints["syntax"]
            constraint_parts.append(f"follows the part-of-speech sequence {pos_sequence}")

        # Sentiment constraint
        if "sentiment" in constraints:
            sentiment = constraints["sentiment"]
            constraint_parts.append(f"has {sentiment} sentiment")

        constraint_text = " and ".join(constraint_parts)

        instruction = f"Write something that {constraint_text}.\n\nOriginal text: {input_text}\n\nGenerated text:"

        return instruction