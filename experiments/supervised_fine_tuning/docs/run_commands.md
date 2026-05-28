# SFT Run Commands

This file records QLoRA SFT training commands for `Qwen/Qwen3-4B-Thinking-2507`.
Run commands from the repo root on a GPU machine that can read and write
`/cephfs/qwen_math_comp`.

## Environment

```bash
cd /cephfs/qwen_math_comp/151B_SP26_Competition
source /cephfs/qwen_math_comp/.venv/bin/activate

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Completed Smoke Run

Known output:

```text
/cephfs/qwen_math_comp/outputs/qwen3_4b_numina_5k_smoketest/final_adapter
```

Recovered dataset/output settings from `/cephfs`:

- train: `/cephfs/qwen_math_comp/data_v2/sft_train_5k.jsonl` (5,000 rows)
- eval: `/cephfs/qwen_math_comp/data_v2/sft_val_300.jsonl` (300 rows)
- output: `/cephfs/qwen_math_comp/outputs/qwen3_4b_numina_5k_smoketest`
- completed: 1 epoch, 313 optimizer steps
- eval loss: 0.39937 at step 313

Command:

```bash
python train_qlora_sft.py \
  --train_path /cephfs/qwen_math_comp/data_v2/sft_train_5k.jsonl \
  --eval_path /cephfs/qwen_math_comp/data_v2/sft_val_300.jsonl \
  --output_dir /cephfs/qwen_math_comp/outputs/qwen3_4b_numina_5k_smoketest \
  --max_seq_len 4096 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --eval_steps 100 \
  --save_steps 100 \
  --logging_steps 10
```

## 10k Numina Harder Run

```bash
python train_qlora_sft.py \
  --train_path /cephfs/qwen_math_comp/data_numina_10k_harder/sft_train_10k.jsonl \
  --eval_path /cephfs/qwen_math_comp/data_numina_10k_harder/sft_val_500.jsonl \
  --output_dir /cephfs/qwen_math_comp/outputs/qwen3_4b_numina_10k_harder \
  --max_seq_len 4096 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --eval_steps 100 \
  --save_steps 100 \
  --logging_steps 10
```

Final adapter will be written to:

```text
/cephfs/qwen_math_comp/outputs/qwen3_4b_numina_10k_harder/final_adapter
```

## 20k Numina Harder Run

```bash
python train_qlora_sft.py \
  --train_path /cephfs/qwen_math_comp/data_numina_20k_harder/sft_train_20k.jsonl \
  --eval_path /cephfs/qwen_math_comp/data_numina_20k_harder/sft_val_500.jsonl \
  --output_dir /cephfs/qwen_math_comp/outputs/qwen3_4b_numina_20k_harder \
  --max_seq_len 4096 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --eval_steps 100 \
  --save_steps 100 \
  --logging_steps 10
```

Final adapter will be written to:

```text
/cephfs/qwen_math_comp/outputs/qwen3_4b_numina_20k_harder/final_adapter
```

## Eval After Training

```bash
export WANDB_NAME=sft_numina10k_harder_sc_terse_4096_vllm

python -m experiments.prompt_engineering.src.eval \
  run=sc_terse_only \
  eval.max_tokens=4096 \
  runner.engine=vllm \
  runner.quant=bf16 \
  runner.adapter_path=/cephfs/qwen_math_comp/outputs/qwen3_4b_numina_10k_harder/final_adapter \
  results_dir=/cephfs/qwen_math_comp/eval_results \
  run_name=sft_numina10k_harder_sc_terse_4096_vllm
```

For the 20k run, change `runner.adapter_path`, `WANDB_NAME`, and `run_name`
from `10k` to `20k`.

On DataHub or any machine where vLLM fails, keep using the HF/BnB fallback:

```bash
export ADAPTER_PATH=/cephfs/qwen_math_comp/outputs/qwen3_4b_numina_10k_harder/final_adapter
export WANDB_NAME=sft_numina10k_harder_sc_terse_4096_hf

python -m experiments.prompt_engineering.src.eval \
  run=sc_terse_only \
  eval.max_tokens=4096 \
  runner.engine=hf \
  runner.quant=bnb \
  results_dir=/cephfs/qwen_math_comp/eval_results \
  run_name=sft_numina10k_harder_sc_terse_4096_hf
```
