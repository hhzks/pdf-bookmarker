"""Generate outline predictions from a fine-tuned LoRA adapter.

Runs the adapter over harvest records' contexts using the exact serving
prompt (llm.PROMPT, raw text — matching how finetune.py trained) and writes
predictions consumable by evaluate.py --predictions.

Usage:
    python training/predict.py records.jsonl checkpoints/outline-lora -o preds.jsonl
    python training/predict.py records.jsonl checkpoints/outline-lora -o preds.jsonl \
        --split dataset/test.jsonl        # only predict docs in this SFT split

Output: JSONL lines of {"sha256": ..., "entries": [...], "parse_error": bool}.
Unparseable generations record empty entries so evaluate.py scores them as
total misses rather than silently skipping them.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pdf_bookmarker.llm import PROMPT, Outline

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def parse_generation(text: str) -> list[dict] | None:
    """Parse a generated completion into entry dicts; None if invalid.

    Tolerates markdown code fences and trailing junk after the JSON object.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].removeprefix("json").strip()
    start = text.find("{")
    if start == -1:
        return None
    decoder = json.JSONDecoder()
    try:
        payload, _ = decoder.raw_decode(text[start:])
        outline = Outline(**payload)
    except Exception:
        return None
    return [
        {"title": e.title, "level": e.level, "printed_page": e.printed_page}
        for e in outline.entries
    ]


def load_records(records_path: Path, split_path: Path | None) -> list[dict]:
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if split_path is not None:
        keep = {
            json.loads(line)["meta"]["sha256"]
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        records = [r for r in records if r["sha256"] in keep]
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records", type=Path, help="harvest records.jsonl")
    parser.add_argument("adapter", type=Path, help="LoRA adapter directory")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=None,
                        help="SFT split file; only predict its documents")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args(argv)

    records = load_records(args.records, args.split)
    if not records:
        print("no records to predict", file=sys.stderr)
        return 1
    print(f"predicting {len(records)} documents", file=sys.stderr)

    try:
        # datasets isn't needed here, but keep torch after any pyarrow users.
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(f"missing dependency ({exc.name}); "
              "run: pip install -r training/requirements.txt", file=sys.stderr)
        return 1

    model_kwargs: dict = {"torch_dtype": "auto", "device_map": "auto"}
    if not args.no_4bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter))

    parse_errors = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for i, record in enumerate(records, 1):
            prompt = PROMPT.format(context=record["context"])
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            completion = tokenizer.decode(
                generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            entries = parse_generation(completion)
            parse_error = entries is None
            if parse_error:
                parse_errors += 1
                entries = []
            out.write(json.dumps(
                {"sha256": record["sha256"], "entries": entries, "parse_error": parse_error},
                ensure_ascii=False) + "\n")
            print(f"  [{i}/{len(records)}] {len(entries)} entries", file=sys.stderr)

    print(f"done; {parse_errors} unparseable generations", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
