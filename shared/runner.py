"""Model loader + batched generator (Transformers + bitsandbytes).

Single hardcoded model: Qwen3-4B-Thinking-2507 on GPU 0. We deliberately do not
use `lightning.fabric.Fabric` — for pure inference with `device_map="auto"`,
Fabric's `launch()` spawns worker processes that grab GPU memory and don't
release it on failure (saw 17 zombie workers holding 21 GB after a crash).

Model-load code is lifted from `starter_code_cse151b_comp.ipynb` cell `3d43b572`
and generation from cell `68bad2c0`.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from shared.schemas import SamplingCfg

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
MAX_MODEL_LEN = 16384
GPU_ID = "0"


@dataclass
class GenerationOutput:
    """Per-prompt generation result (for one or more samples)."""
    responses: list[str]  # length == n_samples
    n_response_tokens: list[int]


class ModelHandle(ABC):
    """Batched-generation interface."""

    tokenizer: Any

    @abstractmethod
    def generate_batch(
        self,
        chat_messages: list[list[dict]],
        sampling: SamplingCfg,
        max_tokens: int,
    ) -> list[GenerationOutput]:
        """Generate one or more samples per prompt. Returns a list parallel to
        `chat_messages`, each entry holding `n_samples` responses.
        """
        ...


class _HFHandle(ModelHandle):
    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    def generate_batch(
        self,
        chat_messages: list[list[dict]],
        sampling: SamplingCfg,
        max_tokens: int,
    ) -> list[GenerationOutput]:
        import torch

        prompts = [
            self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in chat_messages
        ]
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_MODEL_LEN,
        ).to(device)
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_tokens,
            repetition_penalty=sampling.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if sampling.temperature <= 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = sampling.temperature
            gen_kwargs["top_p"] = sampling.top_p
            if sampling.top_k > 0:
                gen_kwargs["top_k"] = sampling.top_k

        results: list[GenerationOutput] = []
        # Loop n_samples (HF doesn't accept `n=` like vLLM); usually n=1.
        per_sample_outputs: list[list[str]] = [[] for _ in prompts]
        per_sample_token_counts: list[list[int]] = [[] for _ in prompts]
        for _ in range(max(1, sampling.n_samples)):
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_kwargs)
            for i, out in enumerate(output_ids):
                new_tokens = out[prompt_len:]
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                per_sample_outputs[i].append(text)
                per_sample_token_counts[i].append(int(new_tokens.shape[0]))

        for i in range(len(prompts)):
            results.append(
                GenerationOutput(
                    responses=per_sample_outputs[i],
                    n_response_tokens=per_sample_token_counts[i],
                )
            )
        return results


def load_model() -> ModelHandle:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", GPU_ID)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    return _HFHandle(model=model, tokenizer=tokenizer)
