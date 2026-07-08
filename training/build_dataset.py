"""Turn harvested records into SFT-ready train/val/test splits.

Reads one or more records.jsonl files produced by harvest.py (or distill.py),
dedups by sha256, splits by *document* (so no PDF leaks across splits), and
formats each record exactly the way the serving path prompts the model:
llm._PROMPT with the record's context, completing with llm._Outline JSON.
Train == serve, by construction.

The split is deterministic — a document's bucket is derived from its sha256 —
so re-running with more data never moves an existing document between splits.

Usage:
    python training/build_dataset.py records.jsonl [more.jsonl ...] -o dataset/
    python training/build_dataset.py records.jsonl -o dataset/ --train 0.9 --val 0.05

Output: dataset/{train,val,test}.jsonl with lines of
    {"prompt": ..., "completion": ..., "meta": {"sha256", "file", "context_kind"}}
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pdf_bookmarker.llm import _PROMPT, _Outline

_BUCKETS = 1000


def split_of(sha256: str, train_frac: float, val_frac: float) -> str:
    """Deterministic document-level split keyed on content hash."""
    bucket = int(sha256[:8], 16) % _BUCKETS
    if bucket < train_frac * _BUCKETS:
        return "train"
    if bucket < (train_frac + val_frac) * _BUCKETS:
        return "val"
    return "test"


def to_sft(record: dict) -> dict:
    """Format one harvest record as a prompt/completion pair.

    Validates the gold entries through the same Pydantic schema the serving
    backends use, so a malformed record fails here rather than at train time.
    """
    outline = _Outline(entries=record["entries"])
    return {
        "prompt": _PROMPT.format(context=record["context"]),
        "completion": outline.model_dump_json(),
        "meta": {
            "sha256": record["sha256"],
            "file": record["file"],
            "context_kind": record["context_kind"],
        },
    }


def build(
    record_files: list[Path], out_dir: Path, train_frac: float, val_frac: float
) -> dict[str, int]:
    """Returns {split_name: record_count} (plus a "duplicates" count)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    handles = {
        name: open(out_dir / f"{name}.jsonl", "w", encoding="utf-8")
        for name in ("train", "val", "test")
    }
    try:
        for record_file in record_files:
            with open(record_file, encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    if record["sha256"] in seen:
                        counts["duplicates"] += 1
                        continue
                    seen.add(record["sha256"])
                    split = split_of(record["sha256"], train_frac, val_frac)
                    handles[split].write(
                        json.dumps(to_sft(record), ensure_ascii=False) + "\n"
                    )
                    counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    return dict(counts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records", type=Path, nargs="+", help="harvest records.jsonl file(s)")
    parser.add_argument("-o", "--out", type=Path, required=True, help="output directory")
    parser.add_argument("--train", type=float, default=0.8, dest="train_frac")
    parser.add_argument("--val", type=float, default=0.1, dest="val_frac")
    args = parser.parse_args(argv)
    if args.train_frac + args.val_frac >= 1.0:
        parser.error("--train + --val must leave room for a test split")

    counts = build(args.records, args.out, args.train_frac, args.val_frac)
    for name in ("train", "val", "test", "duplicates"):
        print(f"{name}: {counts.get(name, 0)}", file=sys.stderr)
    return 0 if counts.get("train") else 1


if __name__ == "__main__":
    raise SystemExit(main())
