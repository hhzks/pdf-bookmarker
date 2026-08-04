#!/usr/bin/env bash
# Run this ON the remote machine before installing anything. It answers the
# three questions that decide whether a 10 GB quota is workable:
#
#   1. is there writable space outside the quota (scratch / node-local)?
#   2. does the cluster already provide torch (module / conda), so we skip 6 GB?
#   3. what is already eating the quota that can be deleted right now?
#
# Read-only apart from a tiny write test in candidate scratch dirs. Paste the
# output back and the install can be aimed correctly.
set -u

hr() { printf '\n=== %s\n' "$1"; }

hr "who / where"
echo "host:   $(hostname)"
echo "user:   $(whoami)"
echo "home:   $HOME"
echo "shell:  ${SHELL:-?}"
[ -r /etc/os-release ] && . /etc/os-release && echo "os:     ${PRETTY_NAME:-?}"

hr "quota"
if command -v quota >/dev/null 2>&1; then
    quota -s 2>/dev/null || echo "(quota command present but reported nothing)"
else
    echo "(no quota command)"
fi
# Lustre and GPFS report quota through their own tools, not `quota`.
command -v lfs   >/dev/null 2>&1 && lfs quota -h "$HOME" 2>/dev/null
command -v mmlsquota >/dev/null 2>&1 && mmlsquota 2>/dev/null | head -5
echo "--- df for \$HOME ---"
df -h "$HOME" 2>/dev/null

hr "candidate space outside \$HOME"
# The usual suspects on HPC boxes. A writable one with tens of GB free is the
# whole solution: caches and venv go there, quota stays untouched.
for d in "${SCRATCH:-}" "${TMPDIR:-}" "${SLURM_TMPDIR:-}" "${LOCAL_SCRATCH:-}" \
         "/scratch/$USER" "/scratch" "/localscratch/$USER" "/local/$USER" \
         "/local" "/data/$USER" "/work/$USER" "/tmp/$USER" "/tmp" "/var/tmp"; do
    [ -z "$d" ] && continue
    [ -d "$d" ] || continue
    avail=$(df -h "$d" 2>/dev/null | awk 'NR==2{print $4}')
    if probe=$(mktemp "$d/.wtest.XXXXXX" 2>/dev/null); then
        rm -f "$probe"
        printf '  %-26s avail %-8s WRITABLE\n' "$d" "${avail:-?}"
    else
        printf '  %-26s avail %-8s not writable\n' "$d" "${avail:-?}"
    fi
done
echo "(node-local dirs like \$SLURM_TMPDIR usually vanish when the job ends —"
echo " fine for TMPDIR, not for the venv or the model cache)"

hr "is torch already provided?"
if command -v module >/dev/null 2>&1 || [ -n "${MODULESHOME:-}" ]; then
    echo "module system: yes"
    # module writes to stderr; grep needs it merged.
    module avail 2>&1 | grep -iE "pytorch|torch|cuda|python/3" | head -20 \
        || echo "  (nothing matched pytorch/cuda/python)"
else
    echo "module system: no"
fi
for c in conda mamba micromamba; do
    command -v $c >/dev/null 2>&1 && echo "$c: $(command -v $c)"
done
python3 -c "import torch,sys; print('torch ALREADY IMPORTABLE:', torch.__version__, 'cuda', torch.version.cuda)" 2>/dev/null \
    || echo "torch: not importable in the default python3"
echo "python3: $(python3 --version 2>&1) at $(command -v python3)"

hr "gpu"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv 2>/dev/null \
        || nvidia-smi 2>&1 | head -12
else
    echo "no nvidia-smi on this node (normal on a login node — check a compute node)"
fi
for s in sinfo qstat bsub; do
    command -v $s >/dev/null 2>&1 && echo "scheduler tool present: $s"
done

hr "what is eating the quota now (reclaimable)"
for d in ~/.cache/pip ~/.cache/huggingface ~/.cache/torch ~/.cache/uv \
         ~/.conda ~/miniconda3 ~/anaconda3 ~/.local/lib ~/.venv ~/venv; do
    [ -e "$d" ] && du -sh "$d" 2>/dev/null
done
echo "--- biggest things in \$HOME ---"
du -sh "$HOME"/* 2>/dev/null | sort -rh | head -12
