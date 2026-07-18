# Remote retrain runbook (8k context, 3B base)

Goal: retrain the outline adapter with `--max-seq-len 8192` and the
`Qwen/Qwen2.5-3B-Instruct` base. The 4k/1.5B local runs drop ~47% of training
records as over-long (including most of the synthetic headings-path
augmentation); 8k reclaims nearly all of them but needs more VRAM than the
local 12 GB card.

The GPU box only needs this repo and the dataset bundle — no PDFs, no PyMuPDF.
`finetune.py` reads `dataset/{train,val}.jsonl` and imports nothing from the
app package.

## 1. Setup

```bash
git clone https://github.com/hhzks/pdf-bookmarker.git
cd pdf-bookmarker
python -m venv .venv && source .venv/bin/activate
pip install -r training/requirements.txt
```

Needs: CUDA GPU (see VRAM notes below), internet access to pull the base
model from Hugging Face on first run (~6 GB download, cached after).

## 2. Dataset

Copy `dataset-bundle.zip` (produced by the local machine; contains
`train/val/test.jsonl` from `build_dataset.py`) into the repo root and unzip:

```bash
unzip dataset-bundle.zip   # creates dataset/{train,val,test}.jsonl
```

`test.jsonl` is included for completeness but is not used remotely —
prediction and evaluation happen back on the local machine, which has the
corpus PDFs.

## 3. Train

```bash
python training/finetune.py dataset/ -o checkpoints/outline-lora-8k \
    --base-model Qwen/Qwen2.5-3B-Instruct --max-seq-len 8192
```

Sanity-check the first stderr line: `train: N (dropped M over-long)` — at 8192
the drop count should be near zero (at 4096 it was 413 of 873).

VRAM guidance (4-bit QLoRA, batch 1, grad checkpointing on):

- 24 GB (3090/4090/A5000): should fit as-is.
- 16 GB: likely fits; if you hit CUDA OOM, first lower `--max-seq-len` to
  6144, then consider the 1.5B base.
- If throughput is inexplicably terrible on a Windows box, check for silent
  spill into shared GPU memory — on Linux you get a clean OOM instead.

Optional knobs: `--epochs 2` (default) matched best locally; `--grad-accum 8`
(default) gives an effective batch of 8.

## 4. Send back the adapter

Only the top-level adapter files are needed — skip the `checkpoint-*`
subdirectories:

```bash
zip -r outline-lora-8k.zip checkpoints/outline-lora-8k -x "*/checkpoint-*"
```

The zip is a few hundred MB at most (LoRA r=16 on 3B plus tokenizer files).
Back on the local machine it goes through the usual `predict.py` →
`evaluate.py` → `export_gguf.py` flow (training/README.md steps 5–7).
