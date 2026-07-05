from __future__ import annotations

from typing import Any


class HFChatModel:
    """Minimal Hugging Face backend for Qwen/Mistral-style instruct models."""

    def __init__(self, model_name: str, max_new_tokens: int = 256, temperature: float = 0.0):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "HF backend requires torch and transformers. Install them first, e.g. "
                "pip install torch transformers accelerate sentencepiece safetensors"
            ) from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, prompt: str, **_: Any) -> str:
        messages = [
            {"role": "system", "content": "You are a careful grid-world planning agent. Output valid JSON only."},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        do_sample = self.temperature > 0
        with self.torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = generated_ids[:, inputs.input_ids.shape[1] :]
        return self.tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
