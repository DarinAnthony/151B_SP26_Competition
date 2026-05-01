# Prompt Engineering — Writeup

Living document for the prompt-engineering experiment. Update with every meaningful run.

## Hypothesis / thought process

The Qwen3-4B-Thinking model emits long `<think>...</think>` traces and frequently exhausts its token budget before reaching `\boxed{}`. The starter notebook documented 4 of 5 baseline answers as unparseable for this reason. We hypothesize:

1. **Format-forcing prompts** (`force_early_boxed`, `cot_structured`) are the largest single lever — they guarantee a boxed answer survives even truncated responses.
2. **Few-shot exemplars** (`fewshot_2_basic`, `fewshot_3_diverse`) improve formatting compliance and provide reasoning patterns; expected lift is moderate and largely orthogonal to (1).
3. **Self-consistency** (`sc_diverse_paths`, `sc_terse` at `temperature=0.8, n=8`) recovers when greedy decoding picks a wrong path; the cost is N× tokens, so it's only worth running on prompts that already do well at greedy.
4. **PHP / Self-Refine** (`php_basic`, `php_skeptical`, `self_refine_1pass`) add a verification pass at modest extra cost (2-3 turns), targeting answer-anchoring vs. early-error failures.
5. **Plan-and-Solve** (`plan_and_solve`) is the standard CoT-baseline-to-beat from Wang et al. 2023.

## Setup

- **Eval slice:** first 100 of `data/public.jsonl` (~38 MCQ, ~62 free-form).
- **Token cap:** `MAX_TOKENS=4096` (down from starter's 32768; pinned in `shared/configs/eval/standard.yaml`).
- **Decoding:** greedy by default; self-consistency prompts override to `regime: high_div_8` (T=0.8, top_p=0.95, top_k=40, n=8). All prompts share the same `MAX_TOKENS` total generation budget per item — multi-turn methods (PHP, Self-Refine) split it equally across turns rather than getting it per-turn — so accuracy / `pct_boxed` / `avg_tok` (now cumulative across turns for multi-turn) are apples-to-apples. (Self-consistency is still N×, since N parallel samples is the method's mechanism, not a hidden budget.)
- **Model:** `Qwen/Qwen3-4B-Thinking-2507`; vLLM (INT8 BnB) preferred, Transformers (INT4 BnB via `lightning.fabric.Fabric`) on DataHub.
- **Stack:** Hydra for config composition, PyTorch Lightning for device/precision (Fabric here; `Trainer` reserved for SFT/RL).

### Reproduce

```bash
# Smoke test (5 questions, 2 prompts) — should take <5 min on a local GPU
python -m experiments.prompt_engineering.src.eval run=smoke eval=smoke

# Full default sweep (11 prompts × 100 q)
python -m experiments.prompt_engineering.src.eval

# Single prompt on smoke slice
python -m experiments.prompt_engineering.src.eval run=smoke eval=smoke +run.runs=[{prompt_id:cot_structured}]

# Full 1,126-q public eval (top-3 prompts only)
python -m experiments.prompt_engineering.src.eval run=full_eval eval=full

# Sweep MAX_TOKENS for the baseline prompt
python -m experiments.prompt_engineering.src.eval --multirun \
    run=smoke eval=smoke eval.max_tokens=1024,2048,4096,8192
```

## Current progress / baseline numbers

| prompt_id          | overall | MCQ   | free  | avg_tok | pct_boxed | regime         | max_tok | run jsonl |
|---|---|---|---|---|---|---|---|---|
| baseline_starter   |  TBD%   | TBD%  | TBD%  |   TBD   |   TBD%    | greedy         |  4096   | TBD       |
| cot_explicit       |  TBD%   | TBD%  | TBD%  |   TBD   |   TBD%    | greedy         |  4096   | TBD       |
| cot_structured     |  TBD%   | TBD%  | TBD%  |   TBD   |   TBD%    | greedy         |  4096   | TBD       |
| fewshot_2_basic    |  TBD%   | TBD%  | TBD%  |   TBD   |   TBD%    | greedy         |  4096   | TBD       |
| fewshot_3_diverse  |  TBD%   | TBD%  | TBD%  |   TBD   |   TBD%    | greedy         |  4096   | TBD       |
| sc_diverse_paths   |  TBD%   | TBD%  | TBD%  |   TBD   |   TBD%    | high_div_8     |  4096   | TBD       |
| sc_terse           |  TBD%   | TBD%  | TBD%  |   TBD   |   TBD%    | high_div_8     |  4096   | TBD       |
| php_basic          |  TBD%   | TBD%  | TBD%  |   TBD   |   TBD%    | greedy         |  4096   | TBD       |
| php_skeptical      |  TBD%   | TBD%  | TBD%  |   TBD   |   TBD%    | greedy         |  4096   | TBD       |
| force_early_boxed  |  TBD%   | TBD%  | TBD%  |   TBD   |   TBD%    | greedy         |  4096   | TBD       |
| plan_and_solve     |  TBD%   | TBD%  | TBD%  |   TBD   |   TBD%    | greedy         |  4096   | TBD       |
| self_refine_1pass  |  TBD%   | TBD%  | TBD%  |   TBD   |   TBD%    | greedy         |  4096   | TBD       |

Update this table after each `run=default` run; link to the per-prompt JSONL under `experiments/prompt_engineering/results/`. Full-set numbers go under `results/full_eval/`.

