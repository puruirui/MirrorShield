import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

from src.mirrorshield import MirrorShield
from config.config import MirrorShieldConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DefenseRunner:
    """Simple runner for MirrorShield defense"""

    def __init__(self, config_path: Optional[str] = None, attack_method: Optional[str] = None):
        # Load configuration
        if config_path and os.path.exists(config_path):
            # Load custom config if provided
            self.config = MirrorShieldConfig.from_file(config_path)
        else:
            # Use default config
            self.config = MirrorShieldConfig()

        # Store attack method
        self.attack_method = attack_method

        # Initialize MirrorShield
        logger.info("Initializing MirrorShield...")
        self.mirrorshield = MirrorShield(self.config)
        logger.info("MirrorShield initialized successfully")

        # Initialize attack module if attack method is specified
        if self.attack_method:
            logger.info(f"Attack method specified: {self.attack_method}")
            try:
                # Import attack module dynamically
                attack_module = __import__(f'attacks.{self.attack_method.lower()}', fromlist=[''])
                self.attack_generator = getattr(attack_module, f'{self.attack_method}Attack')()
                logger.info(f"Attack generator loaded: {self.attack_method}")
            except (ImportError, AttributeError) as e:
                logger.warning(f"Could not load attack method {self.attack_method}: {e}")
                logger.warning("Will use original goals directly without attack transformation")
                self.attack_generator = None
        else:
            logger.info("No attack method specified, will use original goals directly")
            self.attack_generator = None

    def load_prompts_from_csv(self, csv_path: str) -> List[Dict[str, str]]:
        logger.info(f"Loading prompts from: {csv_path}")

        prompts = []

        try:
            df = pd.read_csv(csv_path)

            # Check required columns
            if 'goal' not in df.columns:
                raise ValueError("CSV must contain 'goal' column with harmful prompts")

            for idx, row in df.iterrows():
                prompt_data = {
                    'id': idx,
                    'original_goal': row['goal'],  # Store original harmful goal
                    'target': row.get('target', ''),  # Optional target (not used in defense)
                }
                prompts.append(prompt_data)

        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise

        logger.info(f"Loaded {len(prompts)} harmful prompts")
        return prompts

    def defend_prompts(self, prompts: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        logger.info(f"Applying MirrorShield defense to {len(prompts)} prompts...")

        results = []

        for i, prompt_data in enumerate(prompts):
            try:
                original_goal = prompt_data['original_goal']

                # Generate attack prompt if attack method is specified
                if self.attack_generator:
                    attack_prompt = self.attack_generator.generate_attack(original_goal)
                    logger.debug(f"Generated attack prompt using {self.attack_method}")
                else:
                    # Use original goal directly if no attack method
                    attack_prompt = original_goal

                logger.debug(f"Processing prompt {i + 1}/{len(prompts)}: {attack_prompt[:50]}...")

                # Apply MirrorShield defense
                defended_response, defense_info = self.mirrorshield.defend(attack_prompt)

                # Compile result
                result = {
                    'id': prompt_data['id'],
                    'original_goal': original_goal,
                    'attack_prompt': attack_prompt,  # Include the actual prompt used
                    'defended_response': defended_response,
                    'attack_method': self.attack_method or 'none',
                    'target': prompt_data.get('target', ''),
                    'defense_info': defense_info
                }

                results.append(result)

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(prompts)} prompts")

            except Exception as e:
                logger.error(f"Error processing prompt {i + 1}: {e}")
                # Add error result
                error_result = {
                    'id': prompt_data['id'],
                    'original_goal': prompt_data['original_goal'],
                    'attack_prompt': prompt_data['original_goal'],  # Fallback to original
                    'defended_response': f"ERROR: {str(e)}",
                    'attack_method': self.attack_method or 'none',
                    'target': prompt_data.get('target', ''),
                    'defense_info': {'error': str(e)}
                }
                results.append(error_result)

        logger.info(f"Defense completed. Processed {len(results)} prompts")
        return results

    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        logger.info(f"Saving results to: {output_path}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if output_path.endswith('.csv'):
            self._save_to_csv(results, output_path)
        elif output_path.endswith('.json'):
            self._save_to_json(results, output_path)
        else:
            self._save_to_json(results, output_path + '.json')

        logger.info(f"Results saved successfully")

    def _save_to_csv(self, results: List[Dict[str, Any]], output_path: str):
        fieldnames = [
            'id', 'original_goal', 'attack_prompt', 'defended_response', 'attack_method', 'target'
        ]

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                # Extract only basic fields for CSV
                csv_row = {field: result.get(field, '') for field in fieldnames}
                writer.writerow(csv_row)

    def _save_to_json(self, results: List[Dict[str, Any]], output_path: str):
        """Save results to JSON format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    def run_defense(self, input_csv: str, output_path: str):
        logger.info("Starting MirrorShield defense process")

        prompts = self.load_prompts_from_csv(input_csv)

        results = self.defend_prompts(prompts)

        self.save_results(results, output_path)

        total_prompts = len(results)
        successful_defenses = len([r for r in results if 'ERROR' not in r['defended_response']])

        logger.info(f"Defense process completed:")
        logger.info(f"  Total prompts: {total_prompts}")
        logger.info(f"  Successful defenses: {successful_defenses}")
        logger.info(f"  Success rate: {successful_defenses / total_prompts * 100:.1f}%")
        logger.info(f"  Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Run MirrorShield defense on prompts')
    parser.add_argument('input_csv', help='Input CSV file with prompts')
    parser.add_argument('output_path', help='Output path for defended results')
    parser.add_argument('--config', help='Path to configuration file', default=None)
    parser.add_argument('--attack-method', help='Attack method to use (e.g., GCG, PAIR, AutoDAN)', required=True)
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Logging level')

    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if not os.path.exists(args.input_csv):
        logger.error(f"Input file not found: {args.input_csv}")
        return 1

    try:
        runner = DefenseRunner(args.config, args.attack_method)
        runner.run_defense(args.input_csv, args.output_path)
        return 0

    except Exception as e:
        logger.error(f"Defense process failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())