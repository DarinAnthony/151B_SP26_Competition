# CLAUDE.md

Guidance for working in this repo. Sourced entirely from `README.md` — read it for full context.

## What this repo is

CSE 151B Spring 2026 competition. Base model is **Qwen3-4B-Thinking-2507**, scored on a mix of MCQ + free-form math problems in `data/public.jsonl` (1,126 questions with ground truth).

## Repo layout

| Path | Purpose |
|---|---|
| `starter_code_cse151b_comp.ipynb` | End-to-end pipeline: install → load model → generate → score → save |
| `judger.py` | Scoring logic for free-form answers (symbolic + numeric equivalence) |
| `utils.py` | Helpers used by the judger |
| `data/public.jsonl` | 1,126 questions with ground-truth answers |
| `results/` | JSONL outputs from runs |
| `experiments/` | One subfolder per research direction |

## Experiment tracks

| Folder | Direction |
|---|---|
| `experiments/prompt_engineering/` | System prompts, few-shot examples, output formatting tweaks. Cheapest lever — try first. |
| `experiments/parameter_sampling/` | Decoding params (`temperature`, `top_p`, `top_k`), majority voting, self-consistency over N samples. |
| `experiments/supervised_fine_tuning/` | LoRA / QLoRA SFT on math-reasoning datasets to bias the model toward correct boxed-answer formatting and shorter chains-of-thought. |
| `experiments/reinforcement_learning/` | RLHF / GRPO-style RL with the judger as the reward signal. Most expensive — gate on SFT working first. |

Each experiment's `docs/README.md` has a one-page description of its goals, levers, and success criteria — read that first before opening the code.

## Required experiment file structure

Every experiment must follow this layout so runs and results are comparable across tracks:

```
experiments/<name>/
├── docs/
│   ├── README.md       # goals, levers, success criteria (one page)
│   └── writeup.md      # hypothesis, setup, current progress, baseline numbers, next steps
├── configs/            # YAML/JSON configs for each run, named by run_id
├── src/                # scripts, training/eval code specific to this experiment
├── results/            # JSONL outputs from runs, named <run_id>.jsonl
└── notebooks/          # optional exploratory notebooks (don't commit large outputs)
```

`docs/writeup.md` is the living document — update it with every meaningful run. It must cover:

1. **Hypothesis / thought process** — what we're trying and why we expect it to help.
2. **Setup** — how to reproduce the run (config path, data slice, command).
3. **Current progress / baseline numbers** — accuracy on the standard eval slice, broken down by MCQ vs. free-form.
4. **Open questions / next steps.**

Don't commit large outputs from notebooks.

## Default baseline (use across all experiments)

- **Eval slice:** first **100 questions** of `data/public.jsonl`.
- **Output token limit:** **TBD** — tune once and pin across experiments. Starter currently sets `MAX_TOKENS = 32768`, which is too generous; a smaller cap is likely the right anchor.
- **Metrics:** overall accuracy, MCQ accuracy, free-form accuracy.
- Save run output to `experiments/<name>/results/<run_id>.jsonl` and link it from the writeup.
