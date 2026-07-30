#!/usr/bin/env bash
# Download the trained adapter from blob storage to the local machine, where
# predict.py / evaluate.py / export_gguf.py run. See training/AZURE.md step 5.
set -euo pipefail

RUN=${RUN:-outline-lora-8k}
RG=${RG:-pdfbm-train}
CONTAINER=${CONTAINER:-artifacts}
SUB=$(az account show --query id -o tsv)
STORAGE=${STORAGE:-pdfbm$(printf '%s' "$SUB" | shasum | cut -c1-12)}

KEY=$(az storage account keys list -g "$RG" -n "$STORAGE" --query "[0].value" -o tsv)
az storage blob download \
    --account-name "$STORAGE" --account-key "$KEY" \
    -c "$CONTAINER" -n "$RUN.zip" -f "$RUN.zip" --overwrite -o none

unzip -oq "$RUN.zip"
printf 'unpacked checkpoints/%s\n' "$RUN"
cat <<EOF

next (training/README.md steps 4-7) — --base-model must match what the
adapter was trained on, or the merge/predict silently mismatches:
  python training/predict.py records.jsonl checkpoints/$RUN -o preds.jsonl \\
      --split dataset/test.jsonl --base-model Qwen/Qwen2.5-3B-Instruct
  python training/evaluate.py records.jsonl --predictions preds.jsonl
  python training/export_gguf.py checkpoints/$RUN -o models/outline.gguf
EOF
