#!/usr/bin/env bash
# Fit GatedFusion on frozen CLIP-H + RGPA logits.
# 1) sample 50k subset  2) extract logits once  3) train MLP on rank 0.
#
#   source ~/techjam/venv/bin/activate
#   FUSION_DATA=data/datasets/WildFake_images \
#   bash experiments/fusion/run.sh fusion_wildfake --gpus 1,2,3,4

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
    EXPERIMENTS=(fusion_wildfake)
fi
if [[ -n "$GPUS" ]]; then
    GPUS="${GPUS// /}"
    export CUDA_VISIBLE_DEVICES="$GPUS"
    IFS=',' read -r -a _gpu_ids <<< "$GPUS"
    NUM_GPUS="${#_gpu_ids[@]}"
fi

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"

DATA_ROOT="${FUSION_DATA:-data/datasets/WildFake_images}"
RUNS_ROOT="${FUSION_RUNS:-runs/fusion}"
CONFIG_DIR="experiments/fusion/configs"

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "ERROR: dataset root $DATA_ROOT not found."
    exit 1
fi

for name in "${EXPERIMENTS[@]}"; do
    cfg="$CONFIG_DIR/$name.yaml"
    out="$RUNS_ROOT/$name"
    echo
    echo "============================================================"
    echo " Fusion (gated) — $name"
    echo "============================================================"
    NUM_GPUS="${NUM_GPUS:-$(python -c "import yaml; print(yaml.safe_load(open('$cfg')).get('num_gpus', 1))")}"
    if [[ "$NUM_GPUS" -gt 1 ]]; then
        torchrun --standalone --nproc_per_node="$NUM_GPUS" \
            experiments/fusion/train.py \
            --config "$cfg" \
            --data   "$DATA_ROOT" \
            --output "$out"
    else
        python experiments/fusion/train.py \
            --config "$cfg" \
            --data   "$DATA_ROOT" \
            --output "$out"
    fi
done
