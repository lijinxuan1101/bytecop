# Stage 2 — RGPA Forensic Branch

Stage 2 trains the low-level forensic branch: **RGPA only**. It fuses with
OpenCLIP-H in Stage 3. See `取证分支方案_RGPA.md`.

CIFAKE (32×32) only checks that the training loop runs. Formal evaluation
uses SID-Set.

```
RGB (pixel-scale, no CLIP normalize)
  → probabilistic single degradation (P50 by default)
  → resize/crop to 224×224
  → frozen whole-image SRM-inspired high-pass residual
  → 49×32×32 patches, shared CNN, high/low soft aggregation
  → forensic logit
```

## 1. Experiments

| Name | Mix | Role |
|---|---|---|
| `rgpa_p50` | P50 (`clean_prob=0.5`) | Main candidate |
| `rgpa_p30` | P30 | Optional clean-leaning mix |
| `rgpa_p70` | P70 | Optional robust-leaning mix |

## 2. Metric

```
Final Score = 0.5 × AUC_clean + 0.5 × AUC_robust
```

Exit: send RGPA into Stage 3 if it has independent value and the worst
degradation condition does not clearly collapse. Otherwise drop the forensic
branch and keep OpenCLIP-H.

## 3. How to Run

```bash
# Formal set (SID-Set). Default: rgpa_p50.
bash experiments/stage2/run.sh

# Optional mix ablations
bash experiments/stage2/run.sh rgpa_p50 rgpa_p30 rgpa_p70

# CIFAKE pipeline check (not a performance claim)
STAGE2_DATA=data/datasets/CIFAKE_images bash experiments/stage2/run.sh
```

Or one config at a time:

```bash
python experiments/stage2/train.py \
    --config experiments/stage2/configs/rgpa_p50.yaml \
    --data data/datasets/SID_Set_images

python evaluate.py \
    --backbone rgpa \
    --ckpt runs/stage2/rgpa_p50/best.pt \
    --data data/datasets/SID_Set_images/val \
    --calibrator runs/stage2/rgpa_p50/calibrator.pkl \
    --output runs/stage2/rgpa_p50/eval_results.json
```

Outputs under `runs/stage2/<name>/`:

| File | Description |
|---|---|
| `best.pt` | Best checkpoint (by val AUC) |
| `calibrator.pkl` | Temperature scaler on the calibration split |
| `val_predictions.json` | Per-sample forensic logits for Stage 3 fusion |
| `aggregation_stats.json` | High/low weight divergence |
| `eval_results.json` | Official 15-condition matrix |
| `history.json` | Per-epoch train/val metrics |

## 4. Files in this Directory

| File | Purpose |
|---|---|
| `configs/rgpa_p50.yaml` | Main RGPA config |
| `configs/rgpa_p{30,70}.yaml` | Optional mix ablations |
| `train.py` | Stage 2 training script |
| `run.sh` | Train, evaluate, summarize |
| `summarize.py` | Merge `eval_results.json` → `stage2_results.md` |
