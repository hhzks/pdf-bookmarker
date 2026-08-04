"""Tests for training/export_gguf.py."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

import export_gguf


def write_config(directory, **fields):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(fields), encoding="utf-8")
    return directory


def test_a_declared_mtp_layer_is_skipped(tmp_path):
    """merge_and_unload keeps the text layers only, so a config still claiming
    a multi-token-prediction layer makes the converter write a block_count one
    higher than the tensors it emits, and llama.cpp refuses the file:
    "missing tensor 'blk.24.attn_norm.weight'"."""
    merged = write_config(tmp_path, num_hidden_layers=24, mtp_num_hidden_layers=1)
    assert export_gguf.converter_flags(merged) == ["--no-mtp"]


def test_a_model_without_mtp_layers_needs_no_flags(tmp_path):
    merged = write_config(tmp_path, num_hidden_layers=28)
    assert export_gguf.converter_flags(merged) == []


def test_a_missing_config_needs_no_flags(tmp_path):
    """Only Qwen3.5-shaped models have the field; nothing else should care."""
    assert export_gguf.converter_flags(tmp_path) == []


def _stub_converter(tmp_path):
    """A converter that records its argv and writes an empty output file."""
    llama_cpp = tmp_path / "llama.cpp"
    llama_cpp.mkdir(parents=True, exist_ok=True)
    (llama_cpp / "convert_hf_to_gguf.py").write_text(
        "import sys, json, pathlib\n"
        "pathlib.Path(sys.argv[sys.argv.index('--outfile') + 1]).write_bytes(b'GGUF')\n"
        "pathlib.Path(r'{argv}').write_text(json.dumps(sys.argv))\n".format(
            argv=tmp_path / "argv.json"
        ),
        encoding="utf-8",
    )
    return llama_cpp


def test_the_flag_reaches_the_converter(tmp_path):
    write_config(tmp_path / "merged", num_hidden_layers=24, mtp_num_hidden_layers=1)
    export_gguf.main([
        str(tmp_path / "adapter"), "-o", str(tmp_path / "out.gguf"),
        "--workdir", str(tmp_path), "--llama-cpp", str(_stub_converter(tmp_path)),
    ])
    argv = json.loads((tmp_path / "argv.json").read_text())
    assert "--no-mtp" in argv


def test_an_ordinary_model_is_converted_without_it(tmp_path):
    """The flag is new; passing it always would break older converters."""
    write_config(tmp_path / "merged", num_hidden_layers=28)
    export_gguf.main([
        str(tmp_path / "adapter"), "-o", str(tmp_path / "out.gguf"),
        "--workdir", str(tmp_path), "--llama-cpp", str(_stub_converter(tmp_path)),
    ])
    argv = json.loads((tmp_path / "argv.json").read_text())
    assert "--no-mtp" not in argv
