#!/usr/bin/env bash
# Train the OpenCLIP-H spatial tower.
#
# Usage:
#   bash experiments/spatial_tower/run.sh
#   bash experiments/spatial_tower/run.sh --gpus 2,3,4,5 \
#       --resume runs/spatial_tower/spatial_tower \
#       --skip-eval --nccl-timeout 1800
#
# Env vars (still work; CLI flags override):
#   SPATIAL_DATA   dataset root (default: data/datasets/SID_Set_images)
#   SPATIAL_RUNS   runs root    (default: runs/spatial_tower)
#   SKIP_EVAL / SKIP_TRAIN
#   RESUME          last.pt / best.pt / run dir
#   EXTRA_EPOCHS    extra epochs after resume

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

_need_value() {
    if [[ $# -lt 2 || "$2" == -* ]]; then
        echo "ERROR: $1 needs a value"
        exit 1
    fi
}

GPUS=""
EXPERIMENTS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            _need_value "$@"
            GPUS="$2"
            shift 2
            ;;
        --gpus=*)
            GPUS="${1#*=}"
            shift
            ;;
        --resume)
            _need_value "$@"
            RESUME="$2"
            shift 2
            ;;
        --resume=*)
            RESUME="${1#*=}"
            shift
            ;;
        --extra-epochs)
            _need_value "$@"
            EXTRA_EPOCHS="$2"
            shift 2
            ;;
        --extra-epochs=*)
            EXTRA_EPOCHS="${1#*=}"
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=1
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=1
            shift
            ;;
        --nccl-timeout)
            _need_value "$@"
            TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="$2"
            shift 2
            ;;
        --nccl-timeout=*)
            TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${1#*=}"
            shift
            ;;
        -*)
            echo "ERROR: unknown option $1"
            exit 1
            ;;
        *)
            EXPERIMENTS+=("$1")
            shift
            ;;
    esac
done
if [[ ${#EXPERIMENTS[@]} -eq 0 ]]; then
    EXPERIMENTS=(spatial_tower)
fi
if [[ -n "$GPUS" ]]; then
    GPUS="${GPUS// /}"
    export CUDA_VISIBLE_DEVICES="$GPUS"
    IFS=',' read -r -a _gpu_ids <<< "$GPUS"
    NUM_GPUS="${#_gpu_ids[@]}"
fi
if [[ -n "${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-}" ]]; then
    export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC
fi

# This box's GPU peer-to-peer path deadlocks NCCL's first collective whenever
# more than 2 ranks take part (verified: 4 ranks 0/6 runs pass, 6/6 with P2P
# off). Route collectives through host memory instead. Override by exporting
# NCCL_P2P_DISABLE=0 if the hardware is ever fixed.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

DATA_ROOT="${SPATIAL_DATA:-data/datasets/SID_Set_images}"
RUNS_ROOT="${SPATIAL_RUNS:-runs/spatial_tower}"
CONFIG_DIR="experiments/spatial_tower/configs"

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

for name in "${EXPERIMENTS[@]}"; do
    cfg="$CONFIG_DIR/$name.yaml"
    out="$RUNS_ROOT/$name"
    echo
    echo "============================================================"
    echo " Spatial tower — $name"
    echo "============================================================"

    if [[ -z "${SKIP_TRAIN:-}" ]]; then
        NUM_GPUS="${NUM_GPUS:-$(python -c "import yaml; print(yaml.safe_load(open('$cfg')).get('num_gpus', 1))")}"
        extra_args=()
        if [[ -n "${RESUME:-}" ]]; then
            extra_args+=(--resume "$RESUME")
        fi
        if [[ -n "${EXTRA_EPOCHS:-}" ]]; then
            extra_args+=(--extra-epochs "$EXTRA_EPOCHS")
        fi
        if [[ "$NUM_GPUS" -gt 1 ]]; then
            torchrun --standalone --nproc_per_node="$NUM_GPUS" \
                experiments/spatial_tower/train.py \
                --config "$cfg" \
                --data   "$DATA_ROOT" \
                --output "$out" \
                "${extra_args[@]}"
        else
            python experiments/spatial_tower/train.py \
                --config "$cfg" \
                --data   "$DATA_ROOT" \
                --output "$out" \
                "${extra_args[@]}"
        fi
    fi

    if [[ -z "${SKIP_EVAL:-}" && -f "$out/best.pt" ]]; then
        python evaluate.py \
            --backbone clip_h \
            --ckpt        "$out/best.pt" \
            --data        "$EVAL_DATA" \
            --calibrator  "$out/calibrator.pkl" \
            --output      "$out/eval_results.json"
    fi
done
