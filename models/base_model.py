import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseModel:

    def __init__(self, model_name: str, device: Optional[str] = None):

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self,
                 input_text: str,
                 max_length: int = 512,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 num_return_sequences: int = 1,
                 **kwargs) -> List[str]:

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )

        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            # Remove input text from output for decoder-only models
            if input_text in text:
                text = text.replace(input_text, "").strip()
            generated_texts.append(text)

        return generated_texts

    def get_attention_weights(self, input_text: str) -> Dict[str, torch.Tensor]:

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        return {
            "encoder_attentions": outputs.encoder_attentions if hasattr(outputs, 'encoder_attentions') else None,
            "decoder_attentions": outputs.decoder_attentions if hasattr(outputs, 'decoder_attentions') else None,
            "cross_attentions": outputs.cross_attentions if hasattr(outputs, 'cross_attentions') else None
        }
