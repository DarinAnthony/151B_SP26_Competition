"""Model loader + batched generator.

Exposes a uniform `ModelHandle` so callers (`eval.py`, `multi_turn.py`) don't
branch on backend. The vLLM path is preferred for throughput; the Transformers
path is the DataHub fallback and is wrapped with `lightning.fabric.Fabric` so
device + precision is managed the same way SFT/RL will manage them later.

Loading code is lifted from `starter_code_cse151b_comp.ipynb` cells 3d43b572
(model load — both paths) and 68bad2c0 (generation).
"""

from __future__ import annotations

import logging
import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from shared.schemas import ModelCfg, SamplingCfg

logger = logging.getLogger(__name__)


@dataclass
class GenerationOutput:
    """Per-prompt generation result (for one or more samples)."""
    responses: list[str]  # length == n_samples
    n_response_tokens: list[int]


class ModelHandle(ABC):
    """Backend-agnostic interface for batched generation."""

    backend: str
    model_id: str
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


# ─── vLLM backend ─────────────────────────────────────────────────────────────


class _VLLMHandle(ModelHandle):
    backend = "vllm"

    def __init__(self, llm: Any, tokenizer: Any, model_id: str):
        from vllm import SamplingParams  # local import so import order doesn't matter

        self._SamplingParams = SamplingParams
        self.llm = llm
        self.tokenizer = tokenizer
        self.model_id = model_id

    def _params(self, sampling: SamplingCfg, max_tokens: int, n: int) -> Any:
        return self._SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k if sampling.top_k > 0 else -1,
            min_p=sampling.min_p,
            repetition_penalty=sampling.repetition_penalty,
            presence_penalty=0.0,
        )

    def generate_batch(
        self,
        chat_messages: list[list[dict]],
        sampling: SamplingCfg,
        max_tokens: int,
    ) -> list[GenerationOutput]:
        prompts = [
            self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in chat_messages
        ]
        params = self._params(sampling, max_tokens, sampling.n_samples)
        outputs = self.llm.generate(prompts, sampling_params=params)
        results: list[GenerationOutput] = []
        for out in outputs:
            responses = [o.text.strip() for o in out.outputs]
            n_tokens = [len(o.token_ids) for o in out.outputs]
            results.append(GenerationOutput(responses=responses, n_response_tokens=n_tokens))
        return results


def _load_vllm(cfg: ModelCfg) -> ModelHandle:
    from transformers import AutoTokenizer
    from vllm import LLM

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", cfg.gpu_id)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=cfg.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=cfg.model_id,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=False,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_model_len=cfg.max_model_len,
        trust_remote_code=cfg.trust_remote_code,
        max_num_seqs=cfg.max_num_seqs,
        max_num_batched_tokens=cfg.max_num_batched_tokens,
    )
    return _VLLMHandle(llm=llm, tokenizer=tokenizer, model_id=cfg.model_id)


# ─── Transformers + Lightning Fabric backend ─────────────────────────────────


class _HFHandle(ModelHandle):
    backend = "transformers"

    def __init__(self, model: Any, tokenizer: Any, model_id: str, fabric: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.fabric = fabric

    def generate_batch(
        self,
        chat_messages: list[list[dict]],
        sampling: SamplingCfg,
        max_tokens: int,
    ) -> list[GenerationOutput]:
        import torch

        if sampling.n_samples > 1:
            warnings.warn(
                f"n_samples={sampling.n_samples} on Transformers backend — "
                "this serializes N forward passes per prompt and will be slow. "
                "Use the vLLM backend for self-consistency in production.",
                stacklevel=2,
            )

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
            max_length=16384,
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


def _load_transformers(cfg: ModelCfg) -> ModelHandle:
    import torch
    from lightning.fabric import Fabric
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", cfg.gpu_id)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=cfg.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        trust_remote_code=cfg.trust_remote_code,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()

    # We do NOT call Fabric().launch() here. For pure inference with
    # `device_map="auto"`, Fabric's launch spawns worker processes that grab
    # GPU memory and don't release it on failure (saw 17 zombie workers
    # holding 21 GB after a crash). Construct a Fabric handle for API
    # consistency with future SFT/RL code paths, but skip the multi-process
    # launch — it's a no-op for single-process inference anyway.
    fabric = Fabric(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        precision="bf16-mixed",
        devices=1,
    )
    return _HFHandle(model=model, tokenizer=tokenizer, model_id=cfg.model_id, fabric=fabric)


# ─── Public entry point ──────────────────────────────────────────────────────


def _free_gpu_memory() -> None:
    """Release any cached GPU allocations between failed-load attempts.

    vLLM allocates a KV-cache pool eagerly during construction; if its `LLM(...)`
    raises after that, the allocation lingers and the Transformers fallback OOMs.
    Force gc + empty_cache so the fallback starts with a clean slate.
    """
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def load_model(cfg: ModelCfg) -> ModelHandle:
    """Auto-selects vLLM (preferred) → Transformers (fallback).

    Set `cfg.backend` to "vllm" or "transformers" to skip the auto-detection.
    DataHub users should pass `model.backend=transformers` to skip the vLLM
    attempt entirely (vLLM doesn't initialize cleanly on DataHub).
    """
    backend = cfg.backend.lower()
    if backend == "vllm":
        logger.info("Loading model via vLLM (forced).")
        return _load_vllm(cfg)
    if backend == "transformers":
        logger.info("Loading model via Transformers (forced).")
        return _load_transformers(cfg)

    # auto: try vLLM first
    try:
        import vllm  # noqa: F401
    except ImportError:
        logger.info("vLLM not importable; falling back to Transformers.")
        return _load_transformers(cfg)
    try:
        return _load_vllm(cfg)
    except Exception as e:  # vLLM imports but can't allocate / OOMs / DataHub-style failures
        logger.warning("vLLM load failed (%s); cleaning up GPU and falling back to Transformers.", e)
        _free_gpu_memory()
        return _load_transformers(cfg)
