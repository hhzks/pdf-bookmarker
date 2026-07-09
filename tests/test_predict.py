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
