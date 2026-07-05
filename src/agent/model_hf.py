from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class HFModel:
    model_name: str
    temperature: float = 0.0
    max_new_tokens: int = 256

    def __post_init__(self) -> None:
        print(f"[HFModel] Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        print(f"[HFModel] Loading model: {self.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[HFModel] Model loaded.")

    def generate(
        self,
        prompt: str,
        *,
        env=None,
        belief_grid=None,
        last_feedback=None,
        current_pos=None,
        goal=None,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a grid-world planning agent. "
                    "Return valid JSON only. No markdown. No extra text."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        ).to(self.model.device)

        do_sample = self.temperature is not None and float(self.temperature) > 0.0

        gen_kwargs = {
            "max_new_tokens": int(self.max_new_tokens),
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if do_sample:
            gen_kwargs["temperature"] = float(self.temperature)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        response = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        return response
