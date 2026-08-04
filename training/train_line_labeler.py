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

--text adds a lexical score over the line text as a 14th feature. **It is
off because it was measured and it changes nothing**: title F1 0.7671 -> 0.7675,
paired mean +0.0005 with a 95% interval of [-0.0092, +0.0096].

**Text is not the missing signal, and the reason is redundancy, not weakness.**
An earlier version of this note blamed the lexical model for being
uninformative -- "document-specific noun phrases with no lexical regularity".
That was wrong, and it was tested directly by replacing TF-IDF with a real
sentence encoder (all-MiniLM-L6-v2), frozen and then fine-tuned on these
labels:

    text model              text-only AP   corr. w/ layout   test title F1
      tfidf                       0.1243              0.26        +0.0005
      frozen encoder              0.4348             0.394        +0.0034
      frozen encoder, 32 dims     0.4348             0.394        +0.0034
      fine-tuned encoder          0.6668             0.644        +0.0022
      (layout alone)              0.7684                 -              -

Text quality improved **5.4x** and the end metric never moved -- every variant's
paired interval spans zero (the best was 24 wins against 25 losses). A
fine-tuned encoder reading only the words nearly matches the whole layout model
on its own, and contributes nothing on top of it, because **the better a text
model gets at this task the more it converges on what layout already knows**:
the correlation with the layout score rose monotonically, 0.26 -> 0.394 ->
0.644.

Two things learned along the way, both reusable:

  * collapsing an encoder to one stacked probability is a real bottleneck --
    32 SVD components beat the scalar 5:1 on validation (+0.0161 vs +0.0030);
  * but out-of-fold stacking only works for *scores*, not representations.
    Fold models do not share a coordinate space, so out-of-fold embeddings are
    incoherent across folds and collapse the booster (AP 0.7684 -> 0.1252).

Do not re-open this without a signal that is not "read the line's text".

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
from pdf_bookmarker.line_labeler import (
    FEATURE_NAMES,
    entries_from_labels,
    feature_matrix,
    text_for_model,
)


def load(path: Path, test_shas: set[str] | None, train_frac: float, val_frac: float):
    """Stream documents into per-split feature matrices, keeping test rows whole."""
    splits: dict[str, dict] = {
        name: {"X": [], "y": [], "text": []} for name in ("train", "val", "test")
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
                # Featurize the whole document, then drop MASK rows: the
                # window features read each line's neighbours, and filtering
                # first would compute them over a document production never
                # sees.
                for row, vector in zip(rows, feature_matrix(rows, doc["page_count"])):
                    if row["label"] == MASK:
                        continue
                    splits[split]["X"].append(vector)
                    splits[split]["y"].append(row["label"])
                    splits[split]["text"].append(text_for_model(row["text"]))
    for part in splits.values():
        part["X"] = np.asarray(part["X"], dtype=np.float32)
        part["y"] = np.asarray(part["y"], dtype=np.int8)
    return splits, test_docs


def fit_text_model(texts: list[str], truth, folds: int = 3, seed: int = 0):
    """Lexical heading model, plus leak-free scores for its own training rows.

    Returns (vectorizer, model, out_of_fold_scores). The scores handed back for
    the training rows come from models that never saw those rows: a stacked
    feature scored in-sample looks far more reliable than it is, and the
    gradient booster downstream would lean on it and generalise worse.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import StratifiedKFold

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=5, max_features=300_000, sublinear_tf=True
    )
    matrix = vectorizer.fit_transform(texts)
    positive = (truth > 0).astype(np.int8)

    def make():
        return SGDClassifier(
            loss="log_loss", class_weight="balanced", max_iter=15,
            tol=1e-4, random_state=seed,
        )

    out_of_fold = np.zeros(len(texts), dtype=np.float32)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (fit_idx, score_idx) in enumerate(splitter.split(matrix, positive), 1):
        model = make().fit(matrix[fit_idx], positive[fit_idx])
        out_of_fold[score_idx] = model.predict_proba(matrix[score_idx])[:, 1]
        print(f"  text fold {fold}/{folds}", file=sys.stderr)
    full = make().fit(matrix, positive)
    return vectorizer, full, out_of_fold


def with_text(X, scores):
    """Append the lexical score as one more column."""
    return np.hstack([X, np.asarray(scores, dtype=np.float32).reshape(-1, 1)])


# Shared by both stages. Two departures from sklearn's defaults, both measured
# on the validation documents:
#
#   early_stopping  sklearn turns it on above 10k samples and stops on the loss
#                   over a random 10% of rows. Headings are 1.4% of rows, so
#                   that split is almost all negatives and the detector quit at
#                   99 of 300 iterations -- badly under-fitted. Fitting to the
#                   cap instead: validation 0.6519 -> 0.6936, test 0.7631 ->
#                   0.7783 title F1.
#   max_leaf_nodes  63 against sklearn's 31, chosen on validation among a slow
#                   long fit and a patient stopping rule: 0.6979, the best of
#                   the four. Test 0.7797.
#
# Seed-averaging five detectors does *not* help (validation 0.6498), which is
# the tell that this was under-fitting rather than variance.
ESTIMATOR_PARAMS = {
    "class_weight": "balanced",   # 1.4% positives; without it, all-negative
    "early_stopping": False,
    "max_leaf_nodes": 63,
    "random_state": 0,
}


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
    parser.add_argument("--save-model", type=Path,
                        help="write a bundle loadable by "
                        "pdf_bookmarker.labeler.Labeler.load")
    parser.add_argument("--text", action="store_true",
                        help="add a lexical score over the line text as a "
                        "feature (default: layout only, the control)")
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

    vectorizer = text_model = None
    if args.text:
        print("fitting lexical model...", file=sys.stderr)
        vectorizer, text_model, out_of_fold = fit_text_model(
            train["text"], train["y"]
        )
        train["X"] = with_text(train["X"], out_of_fold)
        if len(val["y"]):
            val["X"] = with_text(
                val["X"],
                text_model.predict_proba(vectorizer.transform(val["text"]))[:, 1],
            )

    print("fitting detector...", file=sys.stderr)
    detector = HistGradientBoostingClassifier(
        max_iter=args.max_iter, **ESTIMATOR_PARAMS
    )
    detector.fit(train["X"], (train["y"] > 0).astype(np.int8))

    threshold, val_f1 = tune_threshold(
        detector.predict_proba(val["X"])[:, 1], val["y"]
    ) if len(val["y"]) else (0.5, float("nan"))
    print(f"threshold={threshold:.2f} (val line-F1={val_f1:.4f})", file=sys.stderr)

    print("fitting level classifier...", file=sys.stderr)
    positive = train["y"] > 0
    leveler = HistGradientBoostingClassifier(
        max_iter=args.max_iter, **ESTIMATOR_PARAMS
    )
    leveler.fit(train["X"][positive], train["y"][positive])

    if args.save_model:
        import joblib
        import sklearn

        if text_model is not None:
            # The serving path has no lexical stage, and shipping a model whose
            # 14th column would silently be absent is worse than refusing.
            print("--save-model cannot be combined with --text: the serving "
                  "path has no lexical stage", file=sys.stderr)
            return 1
        joblib.dump(
            {
                "detector": detector,
                "leveler": leveler,
                "threshold": threshold,
                "feature_names": list(FEATURE_NAMES),
                "sklearn_version": sklearn.__version__,
                "train_rows": int(len(train["y"])),
            },
            args.save_model,
            compress=3,
        )
        size = args.save_model.stat().st_size / 1e6
        print(f"saved model to {args.save_model} ({size:.1f} MB)", file=sys.stderr)

    print(f"predicting {len(test_docs)} test documents...", file=sys.stderr)
    counts: Counter[str] = Counter()
    with open(args.output, "w", encoding="utf-8") as out:
        for doc in test_docs:
            rows = doc["lines"]
            X = np.asarray(feature_matrix(rows, doc["page_count"]), dtype=np.float32)
            if text_model is not None:
                texts = [text_for_model(r["text"]) for r in rows]
                X = with_text(
                    X, text_model.predict_proba(vectorizer.transform(texts))[:, 1]
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
    names = FEATURE_NAMES + (["text_score"] if text_model is not None else [])
    spread = sorted(zip(names, np.std(train["X"], axis=0)), key=lambda kv: -kv[1])
    print("feature spread (sanity check, not importance): "
          + ", ".join(f"{n}={v:.2f}" for n, v in spread[:5]), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
