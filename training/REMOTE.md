# Remote retrain runbook (8k context, 3B base)

Goal: retrain the outline adapter with `--max-seq-len 8192` and the
`Qwen/Qwen2.5-3B-Instruct` base. The 4k/1.5B local runs drop ~47% of training
records as over-long (including most of the synthetic headings-path
augmentation); 8k reclaims nearly all of them but needs more VRAM than the
local 12 GB card.

The GPU box only needs this repo and the dataset bundle — no PDFs, no PyMuPDF.
`finetune.py` reads `dataset/{train,val}.jsonl` and imports nothing from the
app package.

For the Azure version of this — scripted provisioning, dataset transfer, and
teardown, plus the quota and region constraints of the student subscription —
see [AZURE.md](AZURE.md). This file stays the generic runbook.

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

## Low-disk remote (small home quota)

A 10 GB home quota does not fit a naive install. Where it goes:

| item | size | default location |
|---|---|---|
| pip download cache | ~3.5 GB | `~/.cache/pip` |
| installed torch + CUDA libs | ~5.5–6 GB | the venv |
| `Qwen/Qwen2.5-3B-Instruct` weights | 6.2 GB | `~/.cache/huggingface` |
| dataset + adapter + checkpoints | ~0.3 GB | wherever you run |

~15 GB, all under `$HOME`. Two of the three are caches that need not be in the
quota at all, so:

```bash
bash training/remote/probe_env.sh            # where can we put things?
PREFIX=/scratch/$USER training/remote/setup_lowdisk.sh
. /scratch/$USER/env.sh                      # in every shell + job script
```

`probe_env.sh` reports quota, writable scratch, whether the cluster already
provides torch, and what is currently reclaimable. `setup_lowdisk.sh` puts the
venv, the pip cache, `HF_HOME`, and `TMPDIR` under `PREFIX` and installs with
`--no-cache-dir`. Sourcing `env.sh` is not optional — without it the caches
default back to `$HOME` and you are over quota again mid-download.

Three levers, in order of how much they save:

1. **Scratch space.** If `probe_env.sh` finds writable space outside the quota,
   point `PREFIX` at it and the problem disappears. Note node-local dirs
   (`$SLURM_TMPDIR`) are fine for `TMPDIR` but vanish when the job ends — do not
   put the venv or model cache there.
2. **A cluster-provided torch.** `module load pytorch` then
   `SYSTEM_TORCH=1 training/remote/setup_lowdisk.sh` builds the venv with
   `--system-site-packages` and installs only the light deps (~0.5 GB instead of
   ~6 GB). Biggest win if scratch is unavailable.
3. **A pre-quantized base.** `--base-model unsloth/Qwen2.5-3B-Instruct-bnb-4bit`
   is 2.1 GB on disk instead of 6.2 GB, and QLoRA quantizes to 4-bit anyway.
   `finetune.py` detects an already-quantized base and skips its own
   `BitsAndBytesConfig` rather than erroring.

With all three: ~0.5 GB deps + 2.1 GB model ≈ 3 GB, comfortable in 10 GB. With
none of them there is no way to fit, and the run needs a different host.

Caveat on lever 3: the adapter is trained against unsloth's 4-bit copy of the
weights, then `export_gguf.py` merges it into the *full-precision*
`Qwen/Qwen2.5-3B-Instruct`. That is standard QLoRA practice and the
architectures match, but do the merge and export on a machine with disk (the
local one), not the quota-bound box — and evaluate the result before shipping
it, since the numerics are not bit-identical to training against the fp16 base.

## 4. Send back the adapter

Only the top-level adapter files are needed — skip the `checkpoint-*`
subdirectories:

```bash
zip -r outline-lora-8k.zip checkpoints/outline-lora-8k -x "*/checkpoint-*"
```

The zip is a few hundred MB at most (LoRA r=16 on 3B plus tokenizer files).
Back on the local machine it goes through the usual `predict.py` →
`evaluate.py` → `export_gguf.py` flow (training/README.md steps 5–7).
