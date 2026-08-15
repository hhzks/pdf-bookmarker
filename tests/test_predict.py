"""Tests for training/predict.py's torch-free parts (parsing, record filtering)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

import predict


def test_parse_generation_clean_json():
    text = json.dumps(
        {"entries": [{"title": "Intro", "level": 1, "printed_page": 3}]}
    )
    assert predict.parse_generation(text) == [
        {"title": "Intro", "level": 1, "printed_page": 3}
    ]


def test_parse_generation_code_fence_and_trailing_junk():
    text = '```json\n{"entries": [{"title": "A", "level": 1}]}\n```\nextra text'
    assert predict.parse_generation(text) == [
        {"title": "A", "level": 1, "printed_page": None}
    ]


def test_parse_generation_invalid():
    assert predict.parse_generation("not json at all") is None
    assert predict.parse_generation('{"entries": [{"level": 1}]}') is None  # no title


def test_load_records_split_filter(tmp_path):
    records = tmp_path / "records.jsonl"
    records.write_text(
        "\n".join(
            json.dumps({"sha256": s, "context": "c"}) for s in ("aa", "bb", "cc")
        ),
        encoding="utf-8",
    )
    split = tmp_path / "test.jsonl"
    split.write_text(
        json.dumps({"prompt": "p", "completion": "c", "meta": {"sha256": "bb"}}),
        encoding="utf-8",
    )
    kept = predict.load_records(records, split)
    assert [r["sha256"] for r in kept] == ["bb"]
    assert len(predict.load_records(records, None)) == 3


def test_gguf_predictions_go_through_the_shipped_backend(tmp_path, monkeypatch):
    """A GGUF must be predicted through pdf_bookmarker's LocalBackend, not a
    hand-rolled loop. The one time predictions were produced outside it, the
    resulting file was a labeler union mislabelled as the model's output and
    was trusted for days (issue #17)."""
    from pdf_bookmarker.models import OutlineEntry

    seen = {}

    class FakeBackend:
        def __init__(self, model, **kwargs):
            seen["model"] = model
            seen["kwargs"] = kwargs

        def parse_outline(self, context):
            seen.setdefault("contexts", []).append(context)
            return [OutlineEntry(title="Introduction", level=1, printed_page=3)]

    monkeypatch.setattr(predict, "LocalBackend", FakeBackend)

    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps({"sha256": "a", "context": "TOC text", "entries": []}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "preds.jsonl"

    assert predict.main([
        str(records), str(tmp_path / "outline.gguf"), "-o", str(out), "--gguf",
    ]) == 0

    written = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert written[0]["entries"] == [
        {"title": "Introduction", "level": 1, "printed_page": 3}
    ]
    assert seen["contexts"] == ["TOC text"]  # raw context; the backend prompts


def test_gguf_parse_failure_is_recorded_not_crashed(tmp_path, monkeypatch):
    class Boom:
        def __init__(self, model, **kwargs):
            pass

        def parse_outline(self, context):
            raise ValueError("bad json")

    monkeypatch.setattr(predict, "LocalBackend", Boom)
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps({"sha256": "a", "context": "x", "entries": []}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "preds.jsonl"
    assert predict.main([
        str(records), str(tmp_path / "m.gguf"), "-o", str(out), "--gguf",
    ]) == 0
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert row["parse_error"] is True and row["entries"] == []
