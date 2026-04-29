# CSE 151B SP26 Math Reasoning Competition

Working repo for the CSE 151B Spring 2026 competition. Base model is **Qwen3-4B-Thinking-2507**, scored on a mix of MCQ + free-form math problems in `data/public.jsonl`.

---

## Workflow

### Repo layout

| Path | Purpose |
|---|---|
| `starter_code_cse151b_comp.ipynb` | End-to-end pipeline: install → load model → generate → score → save |
| `judger.py` | Scoring logic for free-form answers (symbolic + numeric equivalence) |
| `utils.py` | Helpers used by the judger |
| `data/public.jsonl` | 1,126 questions with ground-truth answers |
| `results/` | JSONL outputs from runs |
| `experiments/` | One subfolder per research direction (see below) |

### Experiments

The four current tracks:

| Folder | Direction |
|---|---|
| `experiments/prompt_engineering/` | System prompts, few-shot examples, output formatting tweaks. Cheapest lever — try first. |
| `experiments/parameter_sampling/` | Decoding params (`temperature`, `top_p`, `top_k`), majority voting, self-consistency over N samples. |
| `experiments/supervised_fine_tuning/` | LoRA / QLoRA SFT on math-reasoning datasets to bias the model toward correct boxed-answer formatting and shorter chains-of-thought. |
| `experiments/reinforcement_learning/` | RLHF / GRPO-style RL with the judger as the reward signal. Most expensive — gate on SFT working first. |

Each experiment's `docs/README.md` has a one-page description of its goals, levers, and success criteria — read that first before opening the code.

#### File structure

Every experiment should follow the same layout so runs and results are comparable across tracks:

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

The `docs/writeup.md` is the living document — update it with every meaningful run. It must cover:

1. **Hypothesis / thought process** — what we're trying and why we expect it to help.
2. **Setup** — how to reproduce the run (config path, data slice, command).
3. **Current progress / baseline numbers** — accuracy on the standard eval slice (see "Default baseline" below), broken down by MCQ vs. free-form.
4. **Open questions / next steps.**

### Default baseline

Every experiment reports against the same eval slice so numbers are comparable:

- **Eval slice:** first **100 questions** of `data/public.jsonl`.
- **Output token limit:** **TBD** — tune once and pin it across experiments. The starter currently sets `MAX_TOKENS = 32768`, which is too generous (see "Known issues" below); a smaller cap is likely the right anchor.
- **Metrics:** overall accuracy, MCQ accuracy, free-form accuracy.
- Save the run output to `experiments/<name>/results/<run_id>.jsonl` and link it from the writeup.

---

## Running `starter_code_cse151b_comp.ipynb`

### First-time setup

1. **Confirm you're on a GPU node before doing anything.** `nvidia-smi` should list a device. Without a GPU the model won't load and you'll waste time debugging install paths instead of the real issue.
2. Open the notebook in Jupyter.
3. Run the **install cell** (section 1). It uses `uv` to create `.venv/` and install `vllm`, `transformers`, `bitsandbytes`, etc.
4. **Make sure the install pulls an updated `torch`.** The default starter packages can resolve to a torch version that doesn't match your CUDA / vLLM combo and you'll get cryptic load errors. If anything fails after this step, check `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` first.
5. **Restart the kernel** when the install finishes.
6. Top-right of the notebook → kernel picker → **"Python (cse151b)"**.
7. **Comment out the install cell** so it doesn't re-run on subsequent kernel restarts.
8. Run the activation cell (`!source ./.venv/bin/activate`) and continue from section 2.

### Two model-loading paths

Section 5 has **two** model-loading cells — pick one and comment out the other:

- **vLLM (INT8 BnB)** — faster batched inference. Use this on local / non-DataHub GPUs.
- **Transformers (INT4 BnB)** — slower, but the path that actually loads on UCSD DataHub. Use this if vLLM fails to import / OOMs / can't allocate KV cache.

The same split applies to section 6 (generation): there are vLLM and Transformers variants of the generation cell.

---

## Known issues / things that bit us

These are the rough edges hit so far. Add to this list as new ones come up.

- **Make sure you're on a GPU node first.** Easy to forget on shared infra (e.g. DataHub). Check `nvidia-smi` before running the install — without a GPU, downstream steps fail in confusing ways and the symptom doesn't point at the cause.
- **Torch version mismatch.** We hit this directly: the install resolved a torch that didn't match the surrounding CUDA / vLLM stack. After install, sanity-check with `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`; if `cuda.is_available()` is `False` on a GPU node, force-reinstall a torch build matching the node's CUDA version before continuing.
- **Kernel switch is manual.** After install, Jupyter does not automatically pick up the new `cse151b` kernel — select it from the kernel dropdown and re-run earlier cells.
- **`!source ./.venv/bin/activate` does not persist** across notebook cells the way it does in a shell. The kernel switch above is what actually pulls in the right packages; the source line is mostly cosmetic.
- **vLLM does not work on DataHub.** That's why the Transformers + BnB INT4 path was added in the latest upstream commit. On DataHub, default to Transformers and skip the vLLM cells entirely.
- **Thinking model burns the token budget before answering.** Most concrete pain point so far: on the first 5-question baseline, **4 of 5 responses never produced `\boxed{...}`** — the model "thought" through the whole budget and got cut off mid-reasoning.
  - Mitigations to try (these are the obvious starting points for `experiments/prompt_engineering/` and `experiments/parameter_sampling/`):
    - Tighter system prompt forcing an early `\boxed{}` commit.
    - Lower `temperature` / sampling determinism so the model stops branching.
    - Cap `MAX_TOKENS` lower and rely on a forced-answer fallback prompt for incomplete responses.
- **Quantization choice matters for accuracy.** INT4 (Transformers path) trades accuracy for fitting on smaller GPUs. If your numbers look worse than expected, check whether the run was on INT4 vs. INT8/BF16 before blaming your method.

---

## Submission

- **Public set** (with ground truth, for local eval): keep `SAVE_EVAL = True` — output includes `gold` and `correct`.
- **Private test set** (for leaderboard): set `SAVE_EVAL = False` — output is only `{id, is_mcq, response}`.
