"""Hydra structured-config dataclasses, shared across all experiments.

Registering these with `ConfigStore` means YAML files in `shared/configs/` and each
experiment's `configs/` are validated against the dataclass schema at load time —
typos and type mismatches fail fast instead of deep inside a run.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class EvalSliceCfg:
    slice: str = "default"
    slice_indices: Optional[list[int]] = None
    max_tokens: int = 4096
    confirm_long_runs: bool = False
    data_path: str = "data/public.jsonl"


@dataclass
class SamplingCfg:
    name: str = "greedy"
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    n_samples: int = 1
    repetition_penalty: float = 1.0
    min_p: float = 0.0


@dataclass
class ModelCfg:
    model_id: str = "Qwen/Qwen3-4B-Thinking-2507"
    backend: str = "auto"
    dtype: str = "bfloat16"
    max_model_len: int = 16384
    quantization: str = "auto"
    trust_remote_code: bool = True
    gpu_memory_utilization: float = 0.50
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 32768
    gpu_id: str = "0"


@dataclass
class PromptRunEntryCfg:
    prompt_id: str = MISSING
    regime: Optional[str] = None
    max_tokens: Optional[int] = None


@dataclass
class RunGroupCfg:
    runs: list[PromptRunEntryCfg] = field(default_factory=list)


@dataclass
class RunCfg:
    eval: EvalSliceCfg = field(default_factory=EvalSliceCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    regime: SamplingCfg = field(default_factory=SamplingCfg)
    run: RunGroupCfg = field(default_factory=RunGroupCfg)
    results_dir: str = "experiments/prompt_engineering/results"
    run_name: str = "default"
    seed: int = 0


def register_configs() -> None:
    """Register dataclass schemas with Hydra's ConfigStore.

    Called from `eval.py` before `@hydra.main` is invoked.
    """
    cs = ConfigStore.instance()
    cs.store(name="run_cfg_schema", node=RunCfg)
    cs.store(group="eval", name="base_eval_cfg", node=EvalSliceCfg)
    cs.store(group="model", name="base_model_cfg", node=ModelCfg)
    cs.store(group="regime", name="base_regime_cfg", node=SamplingCfg)
    cs.store(group="run", name="base_run_group_cfg", node=RunGroupCfg)
