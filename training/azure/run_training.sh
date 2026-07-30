#!/usr/bin/env bash
# Runs ON the Azure GPU box: fetch dataset -> train -> push adapter -> stop the
# meter. See training/AZURE.md step 4.
#
# Run it inside tmux (`tmux new -s train`) — a dropped SSH connection kills the
# run, and a killed run never reaches the self-deallocate at the bottom.
set -euo pipefail

REPO=${REPO:-https://github.com/hhzks/pdf-bookmarker.git}
WORKDIR=${WORKDIR:-$HOME/pdf-bookmarker}
CONTAINER=${CONTAINER:-artifacts}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-8192}
RUN=${RUN:-outline-lora-8k}
DEALLOCATE=${DEALLOCATE:-1}   # set 0 to keep the box up after the run

say() { printf '\n== %s\n' "$1"; }

say "gpu"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

say "azure login (managed identity)"
az login --identity -o none
SUB=$(az account show --query id -o tsv)
STORAGE=${STORAGE:-pdfbm$(printf '%s' "$SUB" | shasum | cut -c1-12)}

say "repo"
if [ -d "$WORKDIR/.git" ]; then
    git -C "$WORKDIR" pull --ff-only
else
    git clone --depth 1 "$REPO" "$WORKDIR"
fi
cd "$WORKDIR"

say "python env"
[ -d .venv ] || python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r training/requirements.txt

say "dataset"
az storage blob download --account-name "$STORAGE" --auth-mode login \
    -c "$CONTAINER" -n dataset-bundle.zip -f dataset-bundle.zip --overwrite -o none
unzip -oq dataset-bundle.zip
wc -l dataset/train.jsonl dataset/val.jsonl

say "train ($BASE_MODEL, max-seq-len $MAX_SEQ_LEN)"
# Watch the first stderr line: `train: N (dropped M over-long)` — at 8192 the
# drop count should be near zero (it was 413 of 873 at 4096).
python training/finetune.py dataset/ -o "checkpoints/$RUN" \
    --base-model "$BASE_MODEL" --max-seq-len "$MAX_SEQ_LEN"

say "upload adapter"
# Top-level adapter files only; the checkpoint-* dirs are per-epoch copies.
zip -rq "$RUN.zip" "checkpoints/$RUN" -x "*/checkpoint-*"
az storage blob upload --account-name "$STORAGE" --auth-mode login \
    -c "$CONTAINER" -n "$RUN.zip" -f "$RUN.zip" --overwrite -o none
printf 'uploaded %s (%s)\n' "$RUN.zip" "$(du -h "$RUN.zip" | cut -f1)"

if [ "$DEALLOCATE" = "1" ]; then
    say "deallocating this vm (stops compute billing)"
    VM_ID=$(curl -fsS -H Metadata:true \
        "http://169.254.169.254/metadata/instance/compute/resourceId?api-version=2021-02-01&format=text")
    # --no-wait: the CLI would otherwise be killed by the shutdown it triggers.
    az vm deallocate --ids "$VM_ID" --no-wait
    echo "deallocate requested for $VM_ID"
else
    say "DEALLOCATE=0 — vm left running, it is billing at \$0.558/hr"
fi
