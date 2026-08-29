#!/usr/bin/env bash
# Stage 1 orchestrator — train S1/S2/S3 sequentially, then evaluate + summarize.
#
# Usage:
#   bash experiments/stage1/run.sh                     # run all three
#   bash experiments/stage1/run.sh s1_linear_probe     # run one
#   bash experiments/stage1/run.sh s1_linear_probe s3_unfreeze4
#
# Env vars:
#   STAGE1_DATA   dataset root (default: data/datasets/CIFAKE_images)
#   STAGE1_RUNS   runs root    (default: runs/stage1)
#   SKIP_EVAL     if set,   skip evaluate.py
#   SKIP_TRAIN    if set,   skip train.py (useful when re-evaluating)

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="${STAGE1_DATA:-data/datasets/CIFAKE_images}"
RUNS_ROOT="${STAGE1_RUNS:-runs/stage1}"
CONFIG_DIR="experiments/stage1/configs"

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "ERROR: dataset root $DATA_ROOT not found."
    echo "Prepare it first:  python data/prepare_cifake.py --dest $DATA_ROOT"
    exit 1
fi

if [[ $# -gt 0 ]]; then
    EXPERIMENTS=("$@")
else
    EXPERIMENTS=(s1_linear_probe s2_unfreeze2 s3_unfreeze4)
fi

for name in "${EXPERIMENTS[@]}"; do
    cfg="$CONFIG_DIR/$name.yaml"
    out="$RUNS_ROOT/$name"
    echo
    echo "============================================================"
    echo " Stage 1 — $name"
    echo "============================================================"

    if [[ -z "${SKIP_TRAIN:-}" ]]; then
        python experiments/stage1/train.py \
            --config "$cfg" \
            --data   "$DATA_ROOT" \
            --output "$out"
    else
        echo "  [SKIP_TRAIN] skipping training."
    fi

    if [[ -z "${SKIP_EVAL:-}" ]]; then
        if [[ -f "$out/best.pt" ]]; then
            python evaluate.py \
                --backbone clip_h \
                --ckpt        "$out/best.pt" \
                --data        "$DATA_ROOT/test" \
                --calibrator  "$out/calibrator.pkl" \
                --output      "$out/eval_results.json"
        else
            echo "  no best.pt at $out, skipping evaluate."
        fi
    else
        echo "  [SKIP_EVAL] skipping evaluation."
    fi
done

echo
echo "============================================================"
echo " Summarizing"
echo "============================================================"
python experiments/stage1/summarize.py \
    --runs   "$RUNS_ROOT" \
    --output "experiments/stage1/stage1_results.md"

echo
echo "Done. See experiments/stage1/stage1_results.md"
