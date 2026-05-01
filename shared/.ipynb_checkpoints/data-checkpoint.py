"""Eval-slice loader, shared across experiments.

The slice name is the cross-experiment contract — every track reports against
`slice=default` (first 100). `smoke` is for pre-flight, `full` is the entire
public set used to verify rankings hold beyond the 100-q slice.

Basically, how much data to load from the dataset
(always starting from the top, not randomized)
"""

from pathlib import Path

from shared.io import load_jsonl
from shared.schemas import EvalSliceCfg

_SLICE_RANGES: dict[str, tuple[int, int] | None] = {
    "smoke": (0, 5),
    "default": (0, 100),
    "full": None,
}


def load_eval_slice(cfg: EvalSliceCfg) -> list[dict]:
    """Load `data/public.jsonl` (or `cfg.data_path`) and apply the configured slice.

    Precedence: explicit `slice_indices` > named `slice`. Unknown names raise.
    """
    path = Path(cfg.data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Eval data not found at {path}. Run from the repo root or set "
            f"eval.data_path to an absolute path."
        )
    items = load_jsonl(path)

    if cfg.slice_indices is not None:
        return [items[i] for i in cfg.slice_indices if 0 <= i < len(items)]

    if cfg.slice not in _SLICE_RANGES:
        raise ValueError(
            f"Unknown slice '{cfg.slice}'. Valid: {sorted(_SLICE_RANGES.keys())} "
            f"or set eval.slice_indices explicitly."
        )
    rng = _SLICE_RANGES[cfg.slice]
    if rng is None:
        return items
    start, end = rng
    return items[start:end]
