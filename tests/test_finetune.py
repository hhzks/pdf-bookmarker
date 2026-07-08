"""Tests for training/finetune.py's torch-free parts (data loading, CLI)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

import finetune


def test_load_split(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = [
        {"prompt": "p1", "completion": "c1", "meta": {"sha256": "a"}},
        {"prompt": "p2", "completion": "c2", "meta": {"sha256": "b"}},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    records = finetune.load_split(path)
    # meta is dropped: TRL expects exactly prompt/completion columns.
    assert records == [
        {"prompt": "p1", "completion": "c1"},
        {"prompt": "p2", "completion": "c2"},
    ]


def test_load_split_missing_file_is_empty(tmp_path):
    assert finetune.load_split(tmp_path / "val.jsonl") == []


def test_parser_defaults(tmp_path):
    args = finetune.build_parser().parse_args([str(tmp_path), "-o", str(tmp_path / "out")])
    assert args.base_model == finetune.DEFAULT_BASE_MODEL
    assert args.lora_r == 16
    assert not args.no_4bit


def test_main_fails_cleanly_without_data(tmp_path, capsys):
    assert finetune.main([str(tmp_path), "-o", str(tmp_path / "out")]) == 1
    assert "no training records" in capsys.readouterr().err
