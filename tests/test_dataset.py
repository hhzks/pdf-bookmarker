"""Tests for training/build_dataset.py, evaluate.py, and distill.py."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

import build_dataset
import distill
import evaluate
import harvest

from pdf_bookmarker.llm import PROMPT
from pdf_bookmarker.models import OutlineEntry


@pytest.fixture(scope="session")
def harvest_records(outlined_toc_pdf, bookmarked_pdf, tmp_path_factory):
    """A small records.jsonl built from the harvestable fixtures."""
    path = tmp_path_factory.mktemp("dataset") / "records.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for pdf, kwargs in ((outlined_toc_pdf, {}), (bookmarked_pdf, {"min_pages": 1})):
            record, reason = harvest.harvest_pdf(pdf, **kwargs)
            assert reason is None
            f.write(json.dumps(record) + "\n")
    return path


# --- build_dataset -----------------------------------------------------------

def test_to_sft_matches_serving_format():
    record = {
        "context": "Table of contents text:\nfoo",
        "sha256": "ab" * 32,
        "file": "x.pdf",
        "context_kind": "toc",
        "entries": [{"title": "Intro", "level": 1, "printed_page": 3}],
    }
    sft = build_dataset.to_sft(record)
    assert sft["prompt"] == PROMPT.format(context=record["context"])
    assert json.loads(sft["completion"]) == {
        "entries": [{"title": "Intro", "level": 1, "printed_page": 3}]
    }
    assert sft["meta"]["sha256"] == record["sha256"]


def test_split_is_deterministic_and_document_level():
    sha = "deadbeef" + "0" * 56
    first = build_dataset.split_of(sha, 0.8, 0.1)
    assert all(build_dataset.split_of(sha, 0.8, 0.1) == first for _ in range(5))
    assert first in ("train", "val", "test")


def test_build_dedups_and_splits(harvest_records, tmp_path):
    out = tmp_path / "dataset"
    # Pass the same records file twice: every doc is a duplicate second time.
    counts = build_dataset.build([harvest_records, harvest_records], out, 0.8, 0.1)
    assert counts["duplicates"] == 2
    total = sum(counts.get(k, 0) for k in ("train", "val", "test"))
    assert total == 2
    lines = []
    for name in ("train", "val", "test"):
        lines += (out / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all("prompt" in json.loads(l) and "completion" in json.loads(l) for l in lines)


# --- evaluate ----------------------------------------------------------------

GOLD = [
    {"title": "1 Introduction", "level": 1, "printed_page": 3},
    {"title": "1.1 Background", "level": 2, "printed_page": 3},
    {"title": "2 Methods", "level": 1, "printed_page": 4},
]


def test_score_outline_perfect():
    scores = evaluate.score_outline(GOLD, GOLD)
    assert scores["f1"] == 1.0
    assert scores["level_accuracy"] == 1.0
    assert scores["page_accuracy"] == 1.0


def test_score_outline_partial():
    pred = [
        {"title": "1 Introduction", "level": 1, "printed_page": 3},   # exact
        {"title": "1.1 Background", "level": 1, "printed_page": 9},   # wrong level+page
        {"title": "Hallucinated", "level": 1, "printed_page": None},  # extra
    ]
    scores = evaluate.score_outline(pred, GOLD)
    assert scores["precision"] == pytest.approx(2 / 3)
    assert scores["recall"] == pytest.approx(2 / 3)
    assert scores["level_accuracy"] == pytest.approx(1 / 2)
    assert scores["page_accuracy"] == pytest.approx(1 / 2)


def test_score_outline_empty_prediction():
    scores = evaluate.score_outline([], GOLD)
    assert scores["f1"] == 0.0
    assert scores["level_accuracy"] is None


def test_heuristic_baseline_on_fixture(harvest_records):
    records = [
        json.loads(l)
        for l in harvest_records.read_text(encoding="utf-8").splitlines()
    ]
    toc_record = next(r for r in records if r["context_kind"] == "toc")
    pred = evaluate.heuristic_predict(toc_record["file"])
    scores = evaluate.score_outline(pred, toc_record["entries"])
    # The gold outline mirrors the printed TOC, so the heuristic should ace it.
    assert scores["f1"] == 1.0
    assert scores["level_accuracy"] == 1.0


def test_evaluate_skips_missing_predictions():
    records = [
        {"sha256": "a", "entries": GOLD},
        {"sha256": "b", "entries": GOLD},
    ]
    result = evaluate.evaluate(records, {"a": GOLD})
    assert result["documents"] == 1
    assert result["skipped"] == 1
    assert result["f1"] == 1.0


# --- distill -----------------------------------------------------------------

class FakeBackend:
    def __init__(self, entries):
        self.entries = entries
        self.contexts = []

    def parse_outline(self, context):
        self.contexts.append(context)
        return self.entries


def test_distill_headings_pdf(headings_pdf):
    backend = FakeBackend([OutlineEntry(title="Chapter 1 Getting Started", level=1)])
    record, reason = distill.distill_pdf(Path(headings_pdf), backend, min_pages=1)
    assert reason is None
    assert record["silver"] is True
    assert record["context_kind"] == "headings"
    assert backend.contexts[0].startswith("Candidate heading lines")
    assert record["entries"] == [
        {"title": "Chapter 1 Getting Started", "level": 1, "printed_page": None}
    ]


def test_distill_skips_already_labeled(bookmarked_pdf):
    """PDFs with an embedded outline belong to harvest.py, not distillation."""
    backend = FakeBackend([])
    record, reason = distill.distill_pdf(Path(bookmarked_pdf), backend, min_pages=1)
    assert record is None
    assert reason == "has-embedded-outline"
    assert backend.contexts == []  # no LLM call was spent


def test_distill_skips_empty_teacher_output(headings_pdf):
    backend = FakeBackend([])
    record, reason = distill.distill_pdf(Path(headings_pdf), backend, min_pages=1)
    assert record is None
    assert reason == "teacher-empty"
