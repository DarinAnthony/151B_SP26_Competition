# GRPO RL — writeup

Living document. Update with every meaningful run.

## 1. Hypothesis / thought process

Static SFT teaches format and brevity but plateaus on **correctness** because the
target is a fixed reference string. GRPO optimizes the metric we're actually scored
on: it samples a *group* of completions per prompt, scores each with the answer
checker, and pushes the policy toward the above-average ones (group-relative
advantage — no learned critic, cheaper than PPO).

The key design choice: **the reward is the eval scorer.** Both call
`shared/scoring.py::score_one`, so reward `1.0` ⇔ a completion that would be marked
correct at eval. There is no proxy objective to drift from.

- **Reward:** `1.0 if score_one(item, completion).correct else 0.0`. Correctness
  only — explainable, directly aligned with scoring. (Shaping for early-`\boxed{}` /
  brevity is a deliberate *later* lever, not in v1.)
- **Init:** base `Qwen/Qwen3-4B-Thinking-2507` by default; `model.init_adapter_path`
  warm-starts from an SFT adapter once one exists (the README's recommended path).
- **No leakage:** train on `public.jsonl[100:]`, evaluate on `slice=default`
  (`[0:100]`). Disjoint. **RL eval must use `default`, not `full`** — `full` overlaps
  the training set and would report an inflated number.

## 2. Setup / reproduce

Install the RL extra (adds TRL; compatible with the pinned transformers 4.x / vllm 0.10):

```bash
pip install -e ".[rl]"        # or: pip install "trl>=0.17,<0.20"
```

Everything runs from the repo root with `REPO_ROOT=$PWD` (also makes the Hydra
searchpath resolve on Colab where cwd is `/content`). Target runtime: Colab **A100**.

```bash
# 0. plumbing smoke (2 steps, HF rollout, 8 rows) — just checks the loop + save path
REPO_ROOT=$PWD python -m experiments.reinforcement_learning.src.train_grpo grpo=smoke run_name=grpo_smoke

# 1. baseline eval (base model, no adapter)
REPO_ROOT=$PWD python -m experiments.reinforcement_learning.src.eval run_name=grpo_base_eval

# 2. demo train (50 steps, vLLM colocate; auto-falls back to HF if vLLM unavailable)
REPO_ROOT=$PWD python -m experiments.reinforcement_learning.src.train_grpo grpo=demo run_name=grpo_demo_v1

# 3. eval the trained adapter on the first-100 slice
REPO_ROOT=$PWD python -m experiments.reinforcement_learning.src.eval \
  adapter_path=experiments/reinforcement_learning/results/grpo_demo_v1/final_adapter \
  run_name=grpo_demo_v1_eval
```

Config groups: `grpo/{demo,smoke}` (rollout + optimization), `model/default`
(init checkpoint), `lora/r16a32` (adapter, matches SFT). Override anything on the
CLI, e.g. `grpo.use_vllm=false grpo.num_generations=8 grpo.max_steps=200`.

Adapters land in `results/<run_name>/final_adapter/`; eval JSONL (canonical
`ResultRow` schema) in `results/<run_name>.jsonl`.

## 3. Current progress / baseline numbers

_Eval slice: first 100 of `data/public.jsonl`, greedy._

| Run | Init | Steps | Overall | MCQ | Free-form |
|---|---|---|---|---|---|
| baseline (base model) | — | — | _TBD_ | _TBD_ | _TBD_ |
| grpo_demo_v1 | base | 50 | _TBD_ | _TBD_ | _TBD_ |

(Fill in after running steps 1–3 above on the A100.)

## 4. Open questions / next steps

- **Run the baseline + demo on A100** and populate the table.
- **`max_completion_length=8192`:** generous for the thinking trace; if the curve is
  starved by truncation (completions cut before `\boxed{}` → reward 0), raise it; if
  rollouts are too slow / OOM, lower it or drop `num_generations`.
- **Reward sparsity:** binary reward gives zero advantage on all-wrong (or all-right)
  groups. If signal is weak, raise `num_generations`, or curriculum the easier
  (MCQ-heavy) rows first — *without* changing the reward.
- **Reward-hacking watch:** `score_one`'s MCQ path accepts a bare trailing capital
  letter even without `\boxed{}`. Watch MCQ vs free-form reward separately via
  `log_completions`; tighten `beta` if the policy drifts to degenerate outputs.
- **Warm-start from SFT** (`model.init_adapter_path=<sft adapter>`) once the SFT track
  has a checkpoint that beats the prompt baseline — the README's gating condition.
- **Shaping reward** (early-`\boxed{}` / length) as a follow-up if correctness-only stalls.
