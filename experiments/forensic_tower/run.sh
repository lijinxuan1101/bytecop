#!/usr/bin/env bash
# Train the RGPA forensic tower.
#
# Usage:
#   bash experiments/forensic_tower/run.sh
#
# Env vars:
#   FORENSIC_DATA   dataset root (default: data/datasets/SID_Set_images)
#   FORENSIC_RUNS   runs root    (default: runs/forensic_tower)
#   SKIP_EVAL / SKIP_TRAIN

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="${FORENSIC_DATA:-data/datasets/SID_Set_images}"
RUNS_ROOT="${FORENSIC_RUNS:-runs/forensic_tower}"
CONFIG_DIR="experiments/forensic_tower/configs"

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "ERROR: dataset root $DATA_ROOT not found."
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
    EXPERIMENTS=(forensic_tower)
fi

for name in "${EXPERIMENTS[@]}"; do
    cfg="$CONFIG_DIR/$name.yaml"
    out="$RUNS_ROOT/$name"
    echo
    echo "============================================================"
    echo " Forensic tower — $name"
    echo "============================================================"

    if [[ -z "${SKIP_TRAIN:-}" ]]; then
        NUM_GPUS="${NUM_GPUS:-$(python -c "import yaml; print(yaml.safe_load(open('$cfg')).get('num_gpus', 1))")}"
        if [[ "$NUM_GPUS" -gt 1 ]]; then
            torchrun --standalone --nproc_per_node="$NUM_GPUS" \
                experiments/forensic_tower/train.py \
                --config "$cfg" \
                --data   "$DATA_ROOT" \
                --output "$out"
        else
            python experiments/forensic_tower/train.py \
                --config "$cfg" \
                --data   "$DATA_ROOT" \
                --output "$out"
        fi
    fi

    if [[ -z "${SKIP_EVAL:-}" && -f "$out/best.pt" ]]; then
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
    fi
done
