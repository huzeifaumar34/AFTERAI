"""Minimal, opt-in supervised LoRA fine-tuning for a compatible chat model.

Install optional dependencies with: pip install -e ".[training]"
This script does not upload model weights or datasets anywhere.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from afterai.data import validate_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Reviewed Hugging Face model ID or local path",
    )
    parser.add_argument("--dataset", required=True, help="Chat JSONL dataset path")
    parser.add_argument("--output", default="outputs/afterai-lora", help="Output directory")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--target-modules",
        default="q_proj,v_proj",
        help="Comma-separated LoRA target modules; check the selected model architecture",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)

    errors = validate_jsonl(dataset_path)
    if errors:
        raise SystemExit("Dataset validation failed:\n" + "\n".join(errors))

    if args.epochs <= 0 or args.batch_size <= 0 or args.grad_accum <= 0:
        raise SystemExit("epochs, batch-size, and grad-accum must be greater than zero")
    if args.max_length < 8 or args.learning_rate <= 0:
        raise SystemExit("max-length must be >= 8 and learning-rate must be positive")

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Install them with: "
            "pip install -e '.[training]'"
        ) from exc

    has_cuda = torch.cuda.is_available()
    device_name = "CUDA GPU" if has_cuda else "CPU (language-model training will be slow)"
    print(f"Device: {device_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise SystemExit("Tokenizer has neither a pad token nor an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token

    if not getattr(tokenizer, "chat_template", None):
        raise SystemExit(
            "This tokenizer has no chat template. Select a chat model with a template "
            "or add a reviewed formatting function before training."
        )

    dtype = torch.float16 if has_cuda else torch.float32
    model_kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False

    raw_dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    def tokenize_record(example: dict) -> dict:
        rendered = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return tokenizer(
            rendered,
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

    tokenized_dataset = raw_dataset.map(
        tokenize_record,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing AFTERAI dataset",
    )

    target_modules = [
        name.strip() for name in args.target_modules.split(",") if name.strip()
    ]
    if not target_modules:
        raise SystemExit("At least one --target-modules entry is required.")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=has_cuda,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print(f"LoRA adapter and tokenizer saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
