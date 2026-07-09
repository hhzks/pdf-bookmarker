"""QLoRA fine-tune of a small causal LM on the SFT dataset.

Trains on the raw prompt/completion pairs produced by build_dataset.py — no
chat template — because the planned local backend prompts the model with the
raw llm.PROMPT text too. Loss is computed on the completion only.

Heavy dependencies live in training/requirements.txt and are imported lazily,
so the rest of the training tooling (and the app) never needs them:

    pip install -r training/requirements.txt
    python training/finetune.py dataset/ -o checkpoints/outline-lora

4-bit QLoRA needs a CUDA GPU (~6 GB VRAM for the 1.5B default). On CPU or
Apple Silicon pass --no-4bit for a plain LoRA run (slow; use a small
--base-model). The output directory receives the LoRA adapter + tokenizer;
merging and GGUF export for llama.cpp are a later step (see README).
"""
import argparse
import json
import sys
from pathlib import Path

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load_split(path: Path) -> list[dict]:
    """Read one split written by build_dataset.py into TRL prompt-completion
    records. Missing file (e.g. an empty val split) -> empty list."""
    if not path.exists():
        return []
    return [
        {"prompt": record["prompt"], "completion": record["completion"]}
        for record in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset_dir", type=Path, help="directory with train/val.jsonl")
    parser.add_argument("-o", "--out", type=Path, required=True, help="adapter output dir")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=8192,
                        help="build_llm_context caps candidates at 400 lines; "
                        "8k tokens covers that with headroom")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--no-4bit", action="store_true",
                        help="skip bitsandbytes quantization (CPU / no CUDA)")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    train_records = load_split(args.dataset_dir / "train.jsonl")
    val_records = load_split(args.dataset_dir / "val.jsonl")
    if not train_records:
        print(f"no training records in {args.dataset_dir}", file=sys.stderr)
        return 1
    print(f"train: {len(train_records)}  val: {len(val_records)}", file=sys.stderr)

    try:
        # datasets must be imported BEFORE torch: the reverse order segfaults
        # (0xC0000005) on Windows with torch 2.7 + pyarrow 24.
        from datasets import Dataset

        import torch
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        print(
            f"missing training dependency ({exc.name}); "
            "run: pip install -r training/requirements.txt",
            file=sys.stderr,
        )
        return 1

    use_4bit = not args.no_4bit
    if use_4bit and not torch.cuda.is_available():
        print("no CUDA device; 4-bit QLoRA needs a GPU — rerun with --no-4bit "
              "or on a CUDA machine", file=sys.stderr)
        return 1

    model_kwargs: dict = {"torch_dtype": "auto"}
    if use_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["device_map"] = "auto"
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    train_config = SFTConfig(
        output_dir=str(args.out),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_seq_len,
        completion_only_loss=True,  # mask the prompt; train on the JSON outline
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        logging_steps=5,
        eval_strategy="epoch" if val_records else "no",
        save_strategy="epoch",
        seed=args.seed,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=train_config,
        train_dataset=Dataset.from_list(train_records),
        eval_dataset=Dataset.from_list(val_records) if val_records else None,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    print(f"saved LoRA adapter to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
