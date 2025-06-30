import os
import json
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class MirrorShieldConfig:
    # Model configurations
    base_model_name: str = "bigscience/T0"
    instruction_tuned_model_path: str = "./models/instruction_tuned_t0"
    target_model_name: str = "meta-llama/Llama-2-7b-chat-hf"

    # Mirror generation parameters
    num_candidate_mirrors: int = 5
    max_mirror_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # Constraint parameters
    length_tolerance: float = 0.2  # ±20% tolerance for length constraint
    syntax_similarity_threshold: float = 0.8
    sentiment_threshold: float = 0.5  # Positive or neutral sentiment

    # Entropy defender parameters
    riu_threshold: float = 0.8  # σ in the paper
    max_iterations: int = 5
    attention_entropy_window: int = 10

    # Training parameters
    learning_rate: float = 5e-5
    batch_size: int = 8
    num_epochs: int = 3
    warmup_steps: int = 100

    # Data paths
    c4_dataset_path: str = "./data/c4_sample"
    training_data_path: str = "./data/training"
    constraint_templates_path: str = "./data/constraint_templates.json"

    # API keys
    openai_api_key: Optional[str] = None

    def __post_init__(self):
        """Post-initialization to load environment variables."""
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")

    @classmethod
    def from_file(cls, config_path: str) -> 'MirrorShieldConfig':
        """Load configuration from JSON file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        if config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                config_dict = json.load(f)

            # Filter only valid dataclass fields
            valid_fields = {field.name for field in cls.__dataclass_fields__.values()}
            filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}

            return cls(**filtered_config)
        else:
            # For Python files, just return default config
            return cls()


# Alias for backward compatibility
Config = MirrorShieldConfig

# Default configuration instance
default_config = MirrorShieldConfig()