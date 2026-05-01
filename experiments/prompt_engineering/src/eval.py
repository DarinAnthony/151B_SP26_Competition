"""Hydra-driven eval CLI for the prompt-engineering experiment.

Loads the model once and dispatches each entry in `cfg.run.runs` by prompt
`kind` (single / self_consistency / php / self_refine). Per-prompt JSONL lands
under `experiments/prompt_engineering/results/`; a leaderboard prints at the end.
"""

from __future__ import annotations

import logging
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# Make the repo root importable so `shared.*` resolves whether we're run as a
# module or as a script.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.data import load_eval_slice  # noqa: E402
from shared.io import ResultRow, save_jsonl  # noqa: E402
from shared.multi_turn import run_php, run_self_refine  # noqa: E402
from shared.prompt_format import build_chat_messages  # noqa: E402
from shared.prompts import Prompt  # noqa: E402
from shared.runner import MODEL_ID, ModelHandle, load_model  # noqa: E402
from shared.schemas import (  # noqa: E402
    EvalSliceCfg,
    PromptRunEntryCfg,
    SamplingCfg,
    register_configs,
)
from shared.scoring import score_one  # noqa: E402
from shared.voting import vote_index  # noqa: E402

from experiments.prompt_engineering.src.prompts import PROMPTS  # noqa: E402

logger = logging.getLogger(__name__)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _load_named_regime(name: str) -> SamplingCfg:
    """Load a regime YAML from shared/configs/regime/<name>.yaml as a SamplingCfg."""
    path = _REPO_ROOT / "shared" / "configs" / "regime" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Unknown regime '{name}' (looked for {path}).")
    raw = OmegaConf.load(path)
    schema = OmegaConf.structured(SamplingCfg)
    merged = OmegaConf.merge(schema, raw)
    return OmegaConf.to_object(merged)  # type: ignore[return-value]


def _resolve_entry(
    entry: PromptRunEntryCfg, default_regime: SamplingCfg, default_max_tokens: int
) -> tuple[Prompt, SamplingCfg, int]:
    if entry.prompt_id not in PROMPTS:
        raise KeyError(
            f"Unknown prompt_id '{entry.prompt_id}'. Known: {sorted(PROMPTS.keys())}"
        )
    prompt = PROMPTS[entry.prompt_id]
    sampling = _load_named_regime(entry.regime) if entry.regime else deepcopy(default_regime)
    max_tokens = entry.max_tokens if entry.max_tokens is not None else default_max_tokens
    return prompt, sampling, max_tokens


def _make_run_id(prompt_id: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"{prompt_id}__{ts}"


def _confirm_long_run(items: list[dict], sampling: SamplingCfg, max_tokens: int, prompt_id: str) -> None:
    # Rough estimate at ~50 tok/s for HF + bitsandbytes 4-bit.
    estimated_seconds = len(items) * sampling.n_samples * max_tokens / 50.0
    if estimated_seconds < 3600:
        return
    hours = estimated_seconds / 3600
    print(
        f"\n[long-run guard] {prompt_id}: estimated {hours:.1f}h "
        f"({len(items)} items × {sampling.n_samples} samples × {max_tokens} max_tokens). "
        f"Rerun with a smaller slice if that's too long."
    )


# ─── Per-kind dispatch ───────────────────────────────────────────────────────


def _run_single(
    prompt: Prompt,
    items: list[dict],
    handle: ModelHandle,
    sampling: SamplingCfg,
    max_tokens: int,
) -> list[dict]:
    """Returns per-item dicts: {response, all_responses?, n_iters, n_response_tokens}."""
    chat_messages = [build_chat_messages(item, prompt) for item in items]
    outputs = handle.generate_batch(chat_messages, sampling, max_tokens)
    rows: list[dict] = []
    for out in outputs:
        rows.append(
            {
                "response": out.responses[0],
                "all_responses": None,
                "n_iters": 1,
                "n_response_tokens": out.n_response_tokens[0] if out.n_response_tokens else 0,
            }
        )
    return rows


def _run_self_consistency(
    prompt: Prompt,
    items: list[dict],
    handle: ModelHandle,
    sampling: SamplingCfg,
    max_tokens: int,
) -> list[dict]:
    chat_messages = [build_chat_messages(item, prompt) for item in items]
    outputs = handle.generate_batch(chat_messages, sampling, max_tokens)
    rows: list[dict] = []
    for item, out in zip(items, outputs):
        # Score-extract every sample, then vote.
        per_sample_extracts = [score_one(item, r).extracted for r in out.responses]
        idx = vote_index(per_sample_extracts)
        rows.append(
            {
                "response": out.responses[idx],
                "all_responses": list(out.responses),
                "n_iters": 1,
                "n_response_tokens": out.n_response_tokens[idx] if out.n_response_tokens else 0,
            }
        )
    return rows


def _run_php_dispatch(
    prompt: Prompt,
    items: list[dict],
    handle: ModelHandle,
    sampling: SamplingCfg,
    max_tokens: int,
) -> list[dict]:
    php_results = run_php(prompt, items, handle, sampling, max_tokens, max_iters=3)
    return [
        {
            "response": r.response,
            "all_responses": r.all_responses,
            "n_iters": r.n_iters,
            "n_response_tokens": r.n_response_tokens,
        }
        for r in php_results
    ]


def _run_self_refine_dispatch(
    prompt: Prompt,
    items: list[dict],
    handle: ModelHandle,
    sampling: SamplingCfg,
    max_tokens: int,
) -> list[dict]:
    refine_results = run_self_refine(prompt, items, handle, sampling, max_tokens)
    return [
        {
            "response": r.response,
            "all_responses": r.all_responses,
            "n_iters": r.n_iters,
            "n_response_tokens": r.n_response_tokens,
        }
        for r in refine_results
    ]


_DISPATCH = {
    "single": _run_single,
    "self_consistency": _run_self_consistency,
    "php": _run_php_dispatch,
    "self_refine": _run_self_refine_dispatch,
}


# ─── Per-prompt orchestration ────────────────────────────────────────────────


def _score_and_save(
    prompt: Prompt,
    items: list[dict],
    gen_rows: list[dict],
    sampling: SamplingCfg,
    max_tokens: int,
    handle: ModelHandle,
    invocation_results_dir: Path,
) -> dict:
    run_id = _make_run_id(prompt.id)
    rows: list[ResultRow] = []
    correct_count = 0
    mcq_correct = mcq_total = free_correct = free_total = 0
    boxed_count = 0
    total_tokens = 0

    for item, gen in zip(items, gen_rows):
        scored = score_one(item, gen["response"])
        if scored.correct:
            correct_count += 1
        if scored.finished_with_box:
            boxed_count += 1
        total_tokens += gen["n_response_tokens"]
        is_mcq = bool(item.get("options"))
        if is_mcq:
            mcq_total += 1
            mcq_correct += int(scored.correct)
        else:
            free_total += 1
            free_correct += int(scored.correct)

        rows.append(
            ResultRow(
                id=int(item["id"]),
                run_id=run_id,
                prompt_id=prompt.id,
                is_mcq=is_mcq,
                gold=item["answer"],
                response=gen["response"],
                extracted=scored.extracted,
                correct=scored.correct,
                finished_with_box=scored.finished_with_box,
                n_response_tokens=int(gen["n_response_tokens"]),
                n_iters=int(gen["n_iters"]),
                sampling={
                    "temperature": sampling.temperature,
                    "top_p": sampling.top_p,
                    "top_k": sampling.top_k,
                    "n_samples": sampling.n_samples,
                },
                max_tokens=max_tokens,
                all_responses=gen.get("all_responses"),
            )
        )

    out_path = invocation_results_dir / f"{run_id}.jsonl"
    save_jsonl(rows, out_path)

    n = len(items)
    return {
        "prompt_id": prompt.id,
        "run_id": run_id,
        "n": n,
        "overall_acc": correct_count / n if n else 0.0,
        "mcq_acc": mcq_correct / mcq_total if mcq_total else float("nan"),
        "free_acc": free_correct / free_total if free_total else float("nan"),
        "avg_tokens": total_tokens / n if n else 0,
        "pct_boxed": boxed_count / n if n else 0.0,
        "regime": sampling.name,
        "max_tokens": max_tokens,
        "out_path": str(out_path),
    }


def _print_leaderboard(rows: list[dict], slice_label: str, max_tokens_default: int) -> None:
    print()
    print("=" * 110)
    print(
        f"PROMPT ENGINEERING LEADERBOARD  "
        f"(slice={slice_label}, default_max_tokens={max_tokens_default})"
    )
    print("=" * 110)
    print(
        f"{'prompt_id':<24} {'overall':>8} {'MCQ':>8} {'free':>8} "
        f"{'avg_tok':>8} {'pct_box':>8} {'regime':>16} {'max_tok':>8}"
    )
    print("-" * 110)
    for r in rows:
        mcq = "n/a" if r["mcq_acc"] != r["mcq_acc"] else f"{r['mcq_acc']*100:.1f}%"
        free = "n/a" if r["free_acc"] != r["free_acc"] else f"{r['free_acc']*100:.1f}%"
        print(
            f"{r['prompt_id']:<24} "
            f"{r['overall_acc']*100:>7.1f}% "
            f"{mcq:>8} "
            f"{free:>8} "
            f"{r['avg_tokens']:>8.0f} "
            f"{r['pct_boxed']*100:>7.1f}% "
            f"{r['regime']:>16} "
            f"{r['max_tokens']:>8}"
        )
    print("=" * 110)


# ─── Hydra entrypoint ────────────────────────────────────────────────────────


register_configs()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Convert the Hydra DictConfig into typed dataclass instances. We merge with
    # the structured schema first so OmegaConf.to_object yields the dataclass
    # type (not a plain dict).
    def _typed(node, schema):
        merged = OmegaConf.merge(OmegaConf.structured(schema), node)
        return OmegaConf.to_object(merged)

    eval_cfg: EvalSliceCfg = _typed(cfg.eval, EvalSliceCfg)  # type: ignore[assignment]
    default_regime: SamplingCfg = _typed(cfg.regime, SamplingCfg)  # type: ignore[assignment]
    runs: list[PromptRunEntryCfg] = [_typed(e, PromptRunEntryCfg) for e in cfg.run.runs]

    if cfg.get("run_name"):
        run_dir_name = str(cfg.run_name)
    else:
        invocation_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        run_group = HydraConfig.get().runtime.choices["run"]
        run_dir_name = f"{run_group}_{invocation_ts}"
    invocation_results_dir = Path(cfg.results_dir) / run_dir_name

    items = load_eval_slice(eval_cfg)
    print(
        f"Loaded {len(items)} items from {eval_cfg.data_path} "
        f"(slice={eval_cfg.slice}, max_tokens={eval_cfg.max_tokens})."
    )
    print(f"Writing results to {invocation_results_dir}")
    if not items:
        print("Empty eval slice; nothing to do.")
        return

    # Cost guard: if any run looks long, warn early
    for entry in runs:
        try:
            prompt, sampling, max_tokens = _resolve_entry(entry, default_regime, eval_cfg.max_tokens)
        except Exception:
            continue
        _confirm_long_run(items, sampling, max_tokens, prompt.id)

    print(f"Loading model: {MODEL_ID}")
    handle = load_model()
    print("Model loaded.")

    leaderboard_rows: list[dict] = []
    for entry in runs:
        prompt, sampling, max_tokens = _resolve_entry(entry, default_regime, eval_cfg.max_tokens)
        if prompt.kind not in _DISPATCH:
            raise ValueError(f"Unknown prompt.kind '{prompt.kind}' for prompt {prompt.id}.")
        runner_fn = _DISPATCH[prompt.kind]
        print(
            f"\n→ {prompt.id} ({prompt.kind}, regime={sampling.name}, "
            f"max_tokens={max_tokens}, n_samples={sampling.n_samples})"
        )
        t0 = time.time()
        gen_rows = runner_fn(prompt, items, handle, sampling, max_tokens)
        elapsed = time.time() - t0
        leaderboard = _score_and_save(
            prompt, items, gen_rows, sampling, max_tokens, handle, invocation_results_dir
        )
        leaderboard_rows.append(leaderboard)
        print(
            f"   acc={leaderboard['overall_acc']*100:.1f}% "
            f"(MCQ={leaderboard['mcq_acc']*100:.1f}% free={leaderboard['free_acc']*100:.1f}%) "
            f"pct_boxed={leaderboard['pct_boxed']*100:.1f}% "
            f"avg_tok={leaderboard['avg_tokens']:.0f} "
            f"wall={elapsed:.1f}s "
            f"→ {leaderboard['out_path']}"
        )

    _print_leaderboard(leaderboard_rows, eval_cfg.slice, eval_cfg.max_tokens)


if __name__ == "__main__":
    main()
