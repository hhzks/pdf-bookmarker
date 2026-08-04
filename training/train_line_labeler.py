"""Train and score a layout-feature baseline for the line-labeling model.

This is the floor, not the destination: it uses only the geometry/typography
features in line_labeler.FEATURE_NAMES and never looks at what a line says.
Its job is to answer "how much of heading detection is layout?" before any
effort goes into a text encoder — if layout alone approaches the generative
model, the encoder only has to supply the rest.

Two stages, because headings are ~2% of lines and a single multiclass argmax
over that imbalance collapses to all-negative:

  1. detect  binary heading/not, with the decision threshold tuned on the
             validation split to maximise title F1 rather than left at 0.5
  2. level   nesting depth, trained on the positives only

Splits are build_dataset.split_of, the same deterministic sha256 buckets the
generative path uses, so no document leaks across splits and the test set is
the one every other experiment was scored on.

Usage:
    python training/train_line_labeler.py lines_all.jsonl -o preds_lines.jsonl
    python training/train_line_labeler.py lines_all.jsonl -o p.jsonl --test-shas records_test_eval.jsonl
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from build_dataset import split_of
from build_line_dataset import MASK
from line_labeler import FEATURE_NAMES, entries_from_labels, feature_vector


def load(path: Path, test_shas: set[str] | None, train_frac: float, val_frac: float):
    """Stream documents into per-split feature matrices, keeping test rows whole."""
    splits: dict[str, dict] = {
        name: {"X": [], "y": []} for name in ("train", "val", "test")
    }
    test_docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            split = split_of(doc["sha256"], train_frac, val_frac)
            if test_shas is not None:
                if doc["sha256"] in test_shas:
                    split = "test"
                elif split == "test":
                    # In the test bucket but not in the fixed set (harvest
                    # dropped it). Training on it would give this model data
                    # the generative model never saw, so drop it entirely.
                    continue
            rows = doc["lines"]
            if split == "test":
                # Test rows are never trained on and are scored through
                # evaluate.py, so keep the document whole and unfiltered.
                test_docs.append(doc)
            else:
                # Drop MASK rows: headings that could not be tied to a specific
                # bookmark. Keeping them would train the model to call a real
                # heading a negative.
                for row, vector in zip(rows, (feature_vector(r, doc["page_count"])
                                              for r in rows)):
                    if row["label"] == MASK:
                        continue
                    splits[split]["X"].append(vector)
                    splits[split]["y"].append(row["label"])
    for part in splits.values():
        part["X"] = np.asarray(part["X"], dtype=np.float32)
        part["y"] = np.asarray(part["y"], dtype=np.int8)
    return splits, test_docs


def tune_threshold(probabilities, truth, grid=None) -> tuple[float, float]:
    """Pick the probability cut that maximises F1 on held-out data."""
    # Reaches past 0.95: with headings at ~1.4% of lines and class_weight
    # "balanced", the useful cut sits far above 0.5 and a grid ending at 0.95
    # pins against its own edge.
    grid = grid if grid is not None else np.concatenate(
        [np.arange(0.05, 0.95, 0.05), np.arange(0.95, 0.999, 0.005)]
    )
    positive = truth > 0
    best = (0.5, -1.0)
    for threshold in grid:
        predicted = probabilities >= threshold
        true_positive = int(np.sum(predicted & positive))
        if not true_positive:
            continue
        precision = true_positive / int(np.sum(predicted))
        recall = true_positive / int(np.sum(positive))
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best[1]:
            best = (float(threshold), f1)
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("lines", type=Path, help="lines.jsonl from build_line_dataset.py")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="predictions JSONL, scoreable by evaluate.py")
    parser.add_argument("--test-shas", type=Path,
                        help="records.jsonl whose sha256s define the test set "
                        "(use the fixed 77-doc set for comparability)")
    parser.add_argument("--train", type=float, default=0.8, dest="train_frac")
    parser.add_argument("--val", type=float, default=0.1, dest="val_frac")
    parser.add_argument("--max-iter", type=int, default=300)
    args = parser.parse_args(argv)

    from sklearn.ensemble import HistGradientBoostingClassifier

    test_shas = None
    if args.test_shas:
        test_shas = {
            json.loads(l)["sha256"]
            for l in args.test_shas.read_text(encoding="utf-8").splitlines()
        }
        print(f"fixed test set: {len(test_shas)} documents", file=sys.stderr)

    splits, test_docs = load(args.lines, test_shas, args.train_frac, args.val_frac)
    for name, part in splits.items():
        positives = int(np.sum(part["y"] > 0))
        print(f"{name:<6} lines={len(part['y']):>9}  headings={positives:>7} "
              f"({positives / max(len(part['y']), 1):.2%})", file=sys.stderr)

    train, val = splits["train"], splits["val"]
    if not len(train["y"]):
        print("no training data", file=sys.stderr)
        return 1

    print("fitting detector...", file=sys.stderr)
    detector = HistGradientBoostingClassifier(
        max_iter=args.max_iter, class_weight="balanced", random_state=0
    )
    detector.fit(train["X"], (train["y"] > 0).astype(np.int8))

    threshold, val_f1 = tune_threshold(
        detector.predict_proba(val["X"])[:, 1], val["y"]
    ) if len(val["y"]) else (0.5, float("nan"))
    print(f"threshold={threshold:.2f} (val line-F1={val_f1:.4f})", file=sys.stderr)

    print("fitting level classifier...", file=sys.stderr)
    positive = train["y"] > 0
    leveler = HistGradientBoostingClassifier(
        max_iter=args.max_iter, class_weight="balanced", random_state=0
    )
    leveler.fit(train["X"][positive], train["y"][positive])

    print(f"predicting {len(test_docs)} test documents...", file=sys.stderr)
    counts: Counter[str] = Counter()
    with open(args.output, "w", encoding="utf-8") as out:
        for doc in test_docs:
            rows = doc["lines"]
            X = np.asarray(
                [feature_vector(r, doc["page_count"]) for r in rows], dtype=np.float32
            )
            is_heading = detector.predict_proba(X)[:, 1] >= threshold
            labels = np.zeros(len(rows), dtype=np.int8)
            if is_heading.any():
                labels[is_heading] = leveler.predict(X[is_heading])
            entries = entries_from_labels(rows, labels.tolist())
            counts["entries"] += len(entries)
            counts["docs"] += 1
            out.write(json.dumps({
                "sha256": doc["sha256"],
                "entries": [
                    # printed_page is unknown to a labeler, so evaluate.py's
                    # page_accuracy reads 0 for this model. It knows something
                    # strictly better: `page`, the exact 0-based physical page,
                    # which is what set_toc needs and what the generative path
                    # has to recover through the locator. Score that instead.
                    {
                        "title": e["title"],
                        "level": e["level"],
                        "printed_page": None,
                        "page": e["page"],
                    }
                    for e in entries
                ],
            }, ensure_ascii=False) + "\n")

    print(f"wrote {counts['docs']} documents, {counts['entries']} entries "
          f"to {args.output}", file=sys.stderr)
    importances = sorted(
        zip(FEATURE_NAMES, np.std(train["X"], axis=0)), key=lambda kv: -kv[1]
    )
    print("feature spread (sanity check, not importance): "
          + ", ".join(f"{n}={v:.2f}" for n, v in importances[:5]), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
