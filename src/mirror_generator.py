import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import Dataset
import json
from typing import List, Dict, Optional
from ..utils.constraint_utils import ConstraintProcessor, Constraint
from ..config.config import MirrorGeneratorConfig


class MirrorGenerator:
    """镜像生成器"""

    def __init__(self, config: MirrorGeneratorConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name)
        self.constraint_processor = ConstraintProcessor()

        # 添加特殊token
        special_tokens = ["<LENGTH>", "<SYNTAX>", "<SENTIMENT>", "</LENGTH>", "</SYNTAX>", "</SENTIMENT>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        self.model.resize_token_embeddings(len(self.tokenizer))

    def create_instruction_prompt(self, constraints: List[Constraint]) -> str:
        """创建指令提示"""
        instruction_parts = []

        for constraint in constraints:
            if constraint.constraint_type == "length":
                min_len, max_len = constraint.value
                instruction_parts.append(f"Write something that has {min_len} to {max_len} words")
            elif constraint.constraint_type == "syntax":
                pos_sequence = " ".join(constraint.value)
                instruction_parts.append(f"Write something with part-of-speech sequence {pos_sequence}")
            elif constraint.constraint_type == "sentiment":
                instruction_parts.append(f"Write something with {constraint.value} sentiment")

        if len(instruction_parts) > 1:
            instruction = " and ".join(instruction_parts) + "."
        else:
            instruction = instruction_parts[0] + "."

        return instruction

    def generate_mirrors(self, input_prompt: str, num_mirrors: Optional[int] = None) -> List[str]:
        """生成镜像"""
        if num_mirrors is None:
            num_mirrors = self.config.num_candidate_mirrors

        # 提取约束
        constraints = [
            self.constraint_processor.extract_length_constraint(input_prompt),
            self.constraint_processor.extract_syntax_constraint(input_prompt),
            self.constraint_processor.extract_sentiment_constraint(positive_sentiment=True)
        ]

        # 创建指令
        instruction = self.create_instruction_prompt(constraints)

        # 生成镜像
        mirrors = []
        for _ in range(num_mirrors):
            mirror = self._generate_single_mirror(instruction)
            if mirror and mirror not in mirrors:
                mirrors.append(mirror)

        return mirrors

    def _generate_single_mirror(self, instruction: str) -> str:
        """生成单个镜像"""
        inputs = self.tokenizer(
            instruction,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.config.max_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 移除指令部分，只保留生成的内容
        if instruction in generated_text:
            generated_text = generated_text.replace(instruction, "").strip()

        return generated_text

    def train(self, training_data: List[Dict], output_dir: str):
        """训练镜像生成器"""
        # 准备训练数据
        train_dataset = self._prepare_training_dataset(training_data)

        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            logging_steps=100,
            save_steps=1000,
            evaluation_strategy="steps",
            eval_steps=500,
            warmup_steps=100,
            load_best_model_at_end=True,
        )

        # 训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
        )

        # 开始训练
        trainer.train()

        # 保存模型
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

    def _prepare_training_dataset(self, training_data: List[Dict]) -> Dataset:
        """准备训练数据集"""
        processed_data = []

        for example in training_data:
            input_text = example["instruction"]
            target_text = example["output"]

            # 对输入和输出进行tokenize
            input_encoding = self.tokenizer(
                input_text,
                max_length=self.config.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            target_encoding = self.tokenizer(
                target_text,
                max_length=self.config.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            processed_data.append({
                "input_ids": input_encoding["input_ids"].squeeze(),
                "attention_mask": input_encoding["attention_mask"].squeeze(),
                "labels": target_encoding["input_ids"].squeeze()
            })

        return Dataset.from_list(processed_data)