"""Train a QLoRA SFT adapter for Qwen3-4B-Thinking on chat JSONL data.

This script is intentionally standalone so SFT training stays separate from the
shared eval runner. It trains only on assistant completion tokens by masking the
system/user prompt tokens with -100.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
from pathlib import Path
from typing import Any


MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
IGNORE_INDEX = -100


class ChatSFTDataset:
    def __init__(self, path: str | Path, tokenizer: Any, max_seq_len: int):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = self._load_examples()

    def _load_examples(self) -> list[dict[str, list[int]]]:
        examples: list[dict[str, list[int]]] = []
        skipped_too_long = 0
        skipped_bad_rows = 0

        with self.path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    messages = row["messages"]
                    if len(messages) < 2 or messages[-1].get("role") != "assistant":
                        skipped_bad_rows += 1
                        continue

                    prompt_messages = messages[:-1]
                    assistant_text = str(messages[-1]["content"])
                    prompt_text = self.tokenizer.apply_chat_template(
                        prompt_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    full_text = prompt_text + assistant_text + self.tokenizer.eos_token

                    prompt_ids = self.tokenizer(
                        prompt_text,
                        add_special_tokens=False,
                    )["input_ids"]
                    full_ids = self.tokenizer(
                        full_text,
                        add_special_tokens=False,
                    )["input_ids"]
                except Exception as e:
                    raise ValueError(f"Failed to process {self.path}:{line_no}") from e

                if len(full_ids) > self.max_seq_len:
                    skipped_too_long += 1
                    continue

                labels = list(full_ids)
                prompt_len = min(len(prompt_ids), len(labels))
                labels[:prompt_len] = [IGNORE_INDEX] * prompt_len

                examples.append(
                    {
                        "input_ids": list(full_ids),
                        "attention_mask": [1] * len(full_ids),
                        "labels": labels,
                    }
                )

        if not examples:
            raise ValueError(
                f"No usable examples loaded from {self.path}. "
                f"Skipped too_long={skipped_too_long}, bad_rows={skipped_bad_rows}."
            )

        print(
            f"Loaded {len(examples)} examples from {self.path} "
            f"(skipped_too_long={skipped_too_long}, skipped_bad_rows={skipped_bad_rows})."
        )
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.examples[idx]


class DataCollatorForChatSFT:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        batch: dict[str, list[list[int]]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * pad_len)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [IGNORE_INDEX] * pad_len)

        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--eval_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--num_train_epochs", type=float, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument(
        "--resume_from_checkpoint",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Resume Trainer state from a checkpoint path. If passed without a "
            "value, uses the latest checkpoint-* under --output_dir."
        ),
    )
    return parser.parse_args()


def latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


def resolve_resume_checkpoint(args: argparse.Namespace) -> str | None:
    if args.resume_from_checkpoint is None:
        return None
    if args.resume_from_checkpoint != "auto":
        checkpoint = Path(args.resume_from_checkpoint).expanduser()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        return str(checkpoint)

    checkpoint = latest_checkpoint(Path(args.output_dir))
    if checkpoint is None:
        raise FileNotFoundError(
            f"No checkpoint-* directory found under {args.output_dir}; "
            "cannot resume automatically."
        )
    return str(checkpoint)


def make_training_args(args: argparse.Namespace, bf16: bool) -> Any:
    from transformers import TrainingArguments

    kwargs: dict[str, Any] = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=bf16,
        fp16=not bf16,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
    )

    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"

    return TrainingArguments(**kwargs)


def main() -> None:
    args = parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer

    output_dir = Path(args.output_dir)
    final_adapter_dir = output_dir / "final_adapter"

    bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    print(f"Using compute_dtype={compute_dtype}, bf16={bf16}, fp16={not bf16}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = ChatSFTDataset(args.train_path, tokenizer, args.max_seq_len)
    eval_dataset = ChatSFTDataset(args.eval_path, tokenizer, args.max_seq_len)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainer_kwargs: dict[str, Any] = dict(
        model=model,
        args=make_training_args(args, bf16=bf16),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForChatSFT(tokenizer.pad_token_id),
    )
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    resume_checkpoint = resolve_resume_checkpoint(args)
    if resume_checkpoint:
        print(f"Resuming training from {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    final_adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    print(f"Saved final adapter to {final_adapter_dir}")


if __name__ == "__main__":
    main()
