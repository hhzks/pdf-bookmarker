"""Merge a LoRA adapter into its base model and export a GGUF for llama.cpp.

Two steps:
  1. peft merge_and_unload — bakes the adapter into full fp16 weights.
  2. llama.cpp's convert_hf_to_gguf.py — converts + quantizes in one pass
     (q8_0 by default: ~1.7 GB for the 1.5B model, no llama-quantize binary
     needed). The converter is fetched by shallow-cloning llama.cpp if no
     --llama-cpp checkout is given; it needs the `gguf` pip package
     (in training/requirements.txt).

Usage:
    python training/export_gguf.py checkpoints/outline-lora-v2 -o models/outline.gguf

The result is what pdf_bookmarker's local backend consumes:
    pdf-bookmarker input.pdf --llm --model "local:models/outline.gguf"
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"


def merge_adapter(adapter_dir: Path, base_model: str, merged_dir: Path) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading base model {base_model} (cpu, fp16)...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype="float16", device_map="cpu"
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    print("merging adapter...", file=sys.stderr)
    model = model.merge_and_unload()
    model.save_pretrained(str(merged_dir))
    AutoTokenizer.from_pretrained(str(adapter_dir)).save_pretrained(str(merged_dir))


def find_converter(llama_cpp_dir: Path | None, workdir: Path) -> Path:
    if llama_cpp_dir is not None:
        script = llama_cpp_dir / "convert_hf_to_gguf.py"
        if not script.exists():
            raise FileNotFoundError(f"{script} not found")
        return script
    clone = workdir / "llama.cpp"
    if not (clone / "convert_hf_to_gguf.py").exists():
        print(f"cloning {LLAMA_CPP_REPO} (shallow)...", file=sys.stderr)
        subprocess.run(
            # core.longpaths: the repo's deep tools/ui paths exceed Windows'
            # 260-char limit when cloned under an already-long workdir.
            ["git", "clone", "--depth", "1", "-c", "core.longpaths=true",
             LLAMA_CPP_REPO, str(clone)],
            check=True,
        )
    return clone / "convert_hf_to_gguf.py"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("adapter", type=Path, help="LoRA adapter directory")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output .gguf path")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--outtype", default="q8_0",
                        help="GGUF quantization: q8_0 (default), f16, bf16, f32")
    parser.add_argument("--llama-cpp", type=Path, default=None,
                        help="existing llama.cpp checkout (skips the shallow clone)")
    parser.add_argument("--workdir", type=Path, default=None,
                        help="scratch dir for the merged model and llama.cpp clone "
                        "(default: a temp dir, deleted on success)")
    args = parser.parse_args(argv)

    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="gguf-export-"))
    workdir.mkdir(parents=True, exist_ok=True)
    merged_dir = workdir / "merged"
    try:
        merge_adapter(args.adapter, args.base_model, merged_dir)
        converter = find_converter(args.llama_cpp, workdir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        print(f"converting to GGUF ({args.outtype})...", file=sys.stderr)
        subprocess.run(
            [sys.executable, str(converter), str(merged_dir),
             "--outfile", str(args.output), "--outtype", args.outtype],
            check=True,
        )
    finally:
        if args.workdir is None:
            shutil.rmtree(workdir, ignore_errors=True)

    size_mb = args.output.stat().st_size / 1e6
    print(f"wrote {args.output} ({size_mb:.0f} MB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
