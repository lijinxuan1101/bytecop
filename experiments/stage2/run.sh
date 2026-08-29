#!/usr/bin/env bash
# Stage 2 orchestrator — train and evaluate RGPA.
#
# Usage:
#   bash experiments/stage2/run.sh                     # rgpa_p50
#   bash experiments/stage2/run.sh rgpa_p50 rgpa_p30
#
# Env vars:
#   STAGE2_DATA   dataset root (default: data/datasets/SID_Set_images)
#   STAGE2_RUNS   runs root    (default: runs/stage2)
#   SKIP_EVAL     if set, skip evaluate.py
#   SKIP_TRAIN    if set, skip train.py

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="${STAGE2_DATA:-data/datasets/SID_Set_images}"
RUNS_ROOT="${STAGE2_RUNS:-runs/stage2}"
CONFIG_DIR="experiments/stage2/configs"

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "ERROR: dataset root $DATA_ROOT not found."
    echo "CIFAKE pipeline:  STAGE2_DATA=data/datasets/CIFAKE_images bash experiments/stage2/run.sh"
    exit 1
fi

if [[ -d "$DATA_ROOT/test" ]]; then
    EVAL_DATA="$DATA_ROOT/test"
else
    EVAL_DATA="$DATA_ROOT/val"
    echo "No test/ split under $DATA_ROOT; evaluating on val/."
fi

if [[ $# -gt 0 ]]; then
    EXPERIMENTS=("$@")
else
    EXPERIMENTS=(rgpa_p50)
fi

for name in "${EXPERIMENTS[@]}"; do
    cfg="$CONFIG_DIR/$name.yaml"
    out="$RUNS_ROOT/$name"

    echo
    echo "============================================================"
    echo " Stage 2 — $name"
    echo "============================================================"

    if [[ -z "${SKIP_TRAIN:-}" ]]; then
        python experiments/stage2/train.py \
            --config "$cfg" \
            --data   "$DATA_ROOT" \
            --output "$out"
    else
        echo "  [SKIP_TRAIN] skipping training."
    fi

    if [[ -z "${SKIP_EVAL:-}" ]]; then
        if [[ -f "$out/best.pt" ]]; then
            cal_args=()
            if [[ -f "$out/calibrator.pkl" ]]; then
                cal_args=(--calibrator "$out/calibrator.pkl")
            fi
            python evaluate.py \
                --backbone rgpa \
                --ckpt        "$out/best.pt" \
                --data        "$EVAL_DATA" \
                "${cal_args[@]}" \
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
python experiments/stage2/summarize.py \
    --runs   "$RUNS_ROOT" \
    --output "experiments/stage2/stage2_results.md"

echo
echo "Done. See experiments/stage2/stage2_results.md"
