"""Tests for training/compare.py."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

import compare


def entry(title, level=1, printed_page=None):
    return {"title": title, "level": level, "printed_page": printed_page}


def gold(sha, titles):
    return {"sha256": sha, "entries": [entry(t) for t in titles]}


# --- pairing -----------------------------------------------------------------


def test_only_documents_scored_by_both_sides_are_paired():
    """A document one side never predicted cannot contribute a delta."""
    records = [gold("a", ["Intro"]), gold("b", ["Intro"])]
    pairs = compare.score_pairs(
        records, {"a": [entry("Intro")], "b": [entry("Intro")]}, {"a": [entry("Intro")]}
    )
    assert list(pairs) == ["a"]


def test_documents_missing_from_the_gold_are_ignored():
    records = [gold("a", ["Intro"])]
    pairs = compare.score_pairs(
        records,
        {"a": [entry("Intro")], "ghost": [entry("Intro")]},
        {"a": [entry("Intro")], "ghost": [entry("Intro")]},
    )
    assert list(pairs) == ["a"]


def test_a_metric_undefined_on_either_side_drops_that_document():
    """level_accuracy is None when a document matched no titles at all.

    Averaging it as zero would punish the side that found nothing differently
    from the side that found nothing *and* mislevelled it.
    """
    records = [gold("a", ["Intro"]), gold("b", ["Intro"])]
    pairs = compare.score_pairs(
        records,
        {"a": [entry("Intro")], "b": [entry("Nothing Like It")]},
        {"a": [entry("Intro")], "b": [entry("Intro")]},
    )
    assert [sha for sha, _ in compare.deltas(pairs, "f1")] == ["a", "b"]
    assert [sha for sha, _ in compare.deltas(pairs, "level_accuracy")] == ["a"]


# --- statistics --------------------------------------------------------------


def test_the_mean_is_paired_not_a_difference_of_averages():
    """Per-document deltas, so a document only one side covers cannot move it."""
    values = [0.2, -0.1, 0.5]
    assert compare.mean(values) == sum(values) / 3


def test_wins_and_losses_are_counted_separately_from_ties():
    wins, losses, ties = compare.tally([0.1, 0.1, -0.2, 0.0])
    assert (wins, losses, ties) == (2, 1, 1)


def test_a_delta_of_floating_point_dust_is_a_tie():
    """Scores are ratios of small integers; 1e-17 apart means identical."""
    assert compare.tally([1e-17, -1e-17]) == (0, 0, 2)


def test_the_interval_brackets_the_observed_mean():
    values = [0.05, 0.02, -0.01, 0.08, 0.03, 0.0, 0.04, -0.02]
    lo, hi = compare.bootstrap_ci(values)
    assert lo < compare.mean(values) < hi


def test_the_interval_is_deterministic():
    """Same input, same interval — a rerun must reproduce a published number."""
    values = [0.05, 0.02, -0.01, 0.08, 0.03, 0.0, 0.04, -0.02]
    assert compare.bootstrap_ci(values) == compare.bootstrap_ci(values)


def test_a_consistent_improvement_excludes_zero():
    lo, hi = compare.bootstrap_ci([0.04, 0.05, 0.03, 0.06, 0.04, 0.05, 0.03, 0.05])
    assert lo > 0


def test_noise_around_zero_spans_zero():
    lo, hi = compare.bootstrap_ci([0.4, -0.4, 0.3, -0.35, 0.02, -0.1, 0.25, -0.3])
    assert lo < 0 < hi


def test_a_single_document_is_not_an_interval():
    """One document cannot be resampled into evidence."""
    assert compare.bootstrap_ci([0.3]) == (None, None)


def test_no_documents_is_not_an_interval():
    assert compare.bootstrap_ci([]) == (None, None)


# --- end to end --------------------------------------------------------------


def test_finding_one_more_heading_reads_as_a_win():
    records = [gold("a", ["Intro", "Method"]), gold("b", ["Intro", "Method"])]
    baseline = {"a": [entry("Intro")], "b": [entry("Intro")]}
    candidate = {
        "a": [entry("Intro"), entry("Method")],
        "b": [entry("Intro"), entry("Method")],
    }
    result = compare.compare(records, baseline, candidate)
    f1 = result["metrics"]["f1"]
    assert f1["documents"] == 2
    assert f1["mean"] > 0
    assert (f1["wins"], f1["losses"]) == (2, 0)


def test_the_arrow_and_the_delta_agree():
    """"baseline -> candidate" is averaged over the paired documents only.

    Quoting a macro average taken over a different document set than the delta
    is how a writeup ends up with an arrow that does not equal its own mean.
    """
    records = [gold("a", ["Intro", "Method"]), gold("b", ["Intro"])]
    baseline = {"a": [entry("Intro")], "b": [entry("Intro")]}
    candidate = {"a": [entry("Intro"), entry("Method")], "b": [entry("Intro")]}
    f1 = compare.compare(records, baseline, candidate)["metrics"]["f1"]
    assert f1["candidate"] - f1["baseline"] == pytest.approx(f1["mean"])


def test_section_numbering_can_be_ignored_the_same_way_evaluate_does():
    """The metric is evaluate's, flag and all — the two cannot disagree."""
    records = [gold("a", ["Introduction"])]
    baseline = {"a": []}
    candidate = {"a": [entry("1. Introduction")]}
    strict = compare.compare(records, baseline, candidate)
    lenient = compare.compare(records, baseline, candidate, ignore_section_numbers=True)
    assert strict["metrics"]["f1"]["mean"] == 0.0
    assert lenient["metrics"]["f1"]["mean"] == 1.0
