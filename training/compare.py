"""Compare two prediction sets on the same documents: paired mean, CI, W/L.

Every claim in this project is stated as a paired comparison with an interval
— "0.7797 -> 0.8009, 34 documents better against 14, CI [+0.0101, +0.0331]" —
because the macro averages alone do not settle anything at this corpus size:
the largest measured effect is about two F1 points over 76 documents, and
document-to-document variance dwarfs it. Two changes can move the average the
same amount while one wins on most documents and the other rides three
outliers.

This is the step that was being redone by hand for each experiment. Doing it
here instead fixes the methodology across experiments the same way
`locator.section_label` fixes the numbering rule across the locator, the
feature and the metric: the numbers are comparable because there is one
implementation of what they mean.

Three things it will not do, each of which is a way to overstate a result:

  * **Score a document only one side covered.** A prediction file missing a
    document is not a zero for that document, and the difference of two macro
    averages over different document sets is not a delta. Only documents
    scored by both sides are paired.
  * **Read a metric that is undefined.** `level_accuracy` is None when a
    document matched no titles; counting that as zero would conflate "found
    nothing" with "found things and mislevelled them". Those documents drop
    out of that metric alone, so each row carries its own n.
  * **Quote an arrow the delta disagrees with.** `baseline` and `candidate`
    are macro averages over the paired documents only, so
    candidate - baseline is exactly `mean`.

The interval is a percentile bootstrap over documents (resample the document
set with replacement, 10k times), which is the right unit because documents
are what vary. It is seeded, so a published number reproduces.

Scores come from `evaluate.score_outline` — the same function `evaluate.py`
macro-averages — so a comparison and a headline number cannot drift apart.
`--ignore-section-numbers` is passed straight through and is normally what you
want for model comparisons: gold comes from embedded bookmarks, which usually
strip the numbering the printed heading shows.

Usage:
    python training/compare.py records_test.jsonl baseline.jsonl candidate.jsonl
    python training/compare.py records_test.jsonl a.jsonl b.jsonl --ignore-section-numbers
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import score_outline

# score_outline's full output; page_accuracy is TOC-path only and will report a
# small n, which is why every row prints its own document count.
METRICS = ("f1", "precision", "recall", "level_accuracy", "page_accuracy")

# Scores are ratios of small counts, so anything this close is the same number
# arrived at by a different route, not a win.
TIE = 1e-12

RESAMPLES = 10000
SEED = 0
CONFIDENCE = 0.95


def score_pairs(records, baseline, candidate, ignore_section_numbers=False):
    """{sha: (baseline_score, candidate_score)} for documents both sides cover.

    Keyed and ordered by the gold records, so a prediction file carrying
    documents outside the evaluated split cannot widen the comparison.
    """
    pairs = {}
    for record in records:
        sha = record["sha256"]
        pred_a, pred_b = baseline.get(sha), candidate.get(sha)
        if pred_a is None or pred_b is None:
            continue
        gold = record["entries"]
        pairs[sha] = (
            score_outline(pred_a, gold, ignore_section_numbers),
            score_outline(pred_b, gold, ignore_section_numbers),
        )
    return pairs


def deltas(pairs, metric):
    """[(sha, candidate - baseline)] over documents where both define `metric`."""
    out = []
    for sha, (a, b) in pairs.items():
        if a[metric] is None or b[metric] is None:
            continue
        out.append((sha, b[metric] - a[metric]))
    return out


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def tally(values, tie=TIE):
    """(wins, losses, ties) — how many documents moved, and which way."""
    wins = sum(1 for v in values if v > tie)
    losses = sum(1 for v in values if v < -tie)
    return wins, losses, len(values) - wins - losses


def bootstrap_ci(values, confidence=CONFIDENCE, resamples=RESAMPLES, seed=SEED):
    """Percentile bootstrap over documents. (None, None) below two documents.

    One document cannot be resampled into evidence: every resample returns the
    same value and the interval would collapse to a point that looks certain.
    """
    values = list(values)
    if len(values) < 2:
        return None, None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(rng.choice(values) for _ in range(n)) / n for _ in range(resamples)
    )
    tail = (1.0 - confidence) / 2.0
    lo = means[int(tail * resamples)]
    hi = means[min(int((1.0 - tail) * resamples), resamples - 1)]
    return lo, hi


def compare(records, baseline, candidate, ignore_section_numbers=False, **ci_kwargs):
    """Paired comparison of two prediction sets over the gold records."""
    pairs = score_pairs(records, baseline, candidate, ignore_section_numbers)
    metrics = {}
    for metric in METRICS:
        scored = deltas(pairs, metric)
        values = [d for _, d in scored]
        lo, hi = bootstrap_ci(values, **ci_kwargs)
        wins, losses, ties = tally(values)
        metrics[metric] = {
            "documents": len(values),
            "baseline": mean(pairs[sha][0][metric] for sha, _ in scored),
            "candidate": mean(pairs[sha][1][metric] for sha, _ in scored),
            "mean": mean(values),
            "ci_low": lo,
            "ci_high": hi,
            "wins": wins,
            "losses": losses,
            "ties": ties,
        }
    return {"documents": len(pairs), "metrics": metrics}


def _load_predictions(path):
    return {
        p["sha256"]: p["entries"]
        for p in (
            json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l
        )
    }


def format_report(result):
    """One line per metric: the arrow, the paired mean with its interval, W/L/T."""
    header = f"{'metric':<15}{'baseline':>10}{'candidate':>11}{'delta':>10}" \
             f"{'95% CI':>22}{'W/L/T':>14}{'n':>5}"
    lines = [header, "-" * len(header)]
    for name, m in result["metrics"].items():
        if not m["documents"]:
            lines.append(f"{name:<15}{'--':>10}{'':>11}{'':>10}{'':>22}{'':>14}{0:>5}")
            continue
        ci = (
            f"[{m['ci_low']:+.4f}, {m['ci_high']:+.4f}]"
            if m["ci_low"] is not None
            else "--"
        )
        wlt = f"{m['wins']}/{m['losses']}/{m['ties']}"
        lines.append(
            f"{name:<15}{m['baseline']:>10.4f}{m['candidate']:>11.4f}"
            f"{m['mean']:>+10.4f}{ci:>22}{wlt:>14}{m['documents']:>5}"
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records", type=Path, help="harvest records.jsonl (the gold)")
    parser.add_argument("baseline", type=Path, help="JSONL of {sha256, entries}")
    parser.add_argument("candidate", type=Path, help="JSONL of {sha256, entries}")
    parser.add_argument("--ignore-section-numbers", action="store_true",
                        help="match titles ignoring a leading section label; "
                        "normally what you want for model comparisons "
                        "(see evaluate.py)")
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--seed", type=int, default=SEED,
                        help="fixed so a published interval reproduces")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    records = [
        json.loads(l)
        for l in args.records.read_text(encoding="utf-8").splitlines()
        if l
    ]
    result = compare(
        records,
        _load_predictions(args.baseline),
        _load_predictions(args.candidate),
        args.ignore_section_numbers,
        resamples=args.resamples,
        seed=args.seed,
    )

    if not result["documents"]:
        print("no documents scored by both sides", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2) if args.json else format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
