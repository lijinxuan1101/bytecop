# Stage 1 — OpenCLIP Single-Tower Baseline on CIFAKE

Stage 1 is a **pipeline validation stage**, not a final-model stage.
CIFAKE (32×32 native resolution) is not representative of the competition's
real-world image distribution, so results here do **not** speak to final
generalization ability.

## 1. Experimental Setup

### 1.1 Dataset

**CIFAKE**, sampled to 10,000 images.

| Split       | Size  | Purpose                                   |
|-------------|-------|-------------------------------------------|
| train       | 7,000 | Model training                            |
| val         | 1,000 | Best-checkpoint selection                 |
| test        | 1,000 | Final evaluation (15-condition matrix)    |
| calibration | 1,000 | Temperature scaling (independent split)   |

Real / fake ratio is 1:1 within every split.

### 1.2 Metric

Uses the official robustness function in `evaluate.py`:

```
Final Score = 0.5 × AUC_clean + 0.5 × AUC_robust
```

`AUC_robust` is the mean AUC across the 14 degraded conditions
(JPEG q90/70/50/30, Gaussian blur σ0.5/1.0/2.0, resize 0.5/0.25,
Gaussian noise σ0.02/0.05/0.10, color jitter 0.2, center crop 0.8).

### 1.3 Result Recording

Each experiment writes:
- `runs/stage1/<name>/best.pt`
- `runs/stage1/<name>/calibrator.pkl`
- `runs/stage1/<name>/history.json`
- `runs/stage1/<name>/eval_results.json`
- `runs/stage1/<name>/tensorboard/`

`experiments/stage1/summarize.py` merges the three `eval_results.json` files
into a single Markdown table (written to `stage1_results.md`).

## 2. OpenCLIP Single-Tower Baseline

### 2.1 S1 — Linear Probe

- Freeze the entire OpenCLIP backbone.
- Train only the linear classification head.
- Clean images only (no degradation augmentation).
- Select best checkpoint by validation AUC.

**Purpose**: verify that raw OpenCLIP features already contain
information separating real photos from AI-generated images.

Config: `configs/s1_linear_probe.yaml`.

### 2.2 S2 — Partial Fine-tune (last 2 blocks)

- Unfreeze the last **2** transformer blocks; keep the rest frozen.
- Train these blocks together with the classification head.
- Identical data split to S1.
- Clean images only.

**Purpose**: measure whether minimal task adaptation improves detection
while preserving generalization.

Config: `configs/s2_unfreeze2.yaml`.

### 2.3 S3 — Extended Partial Fine-tune (last 4 blocks) — Optional

Run **only** if S2 shows a stable improvement over S1.

- Unfreeze the last **4** transformer blocks.
- Everything else identical to S2.

Config: `configs/s3_unfreeze4.yaml`.

## 3. How to Run

```bash
# 1. Prepare data (once)
python data/prepare_cifake.py --dest data/datasets/CIFAKE_images

# 2. Run all three experiments (train + evaluate + summarize)
bash experiments/stage1/run.sh

# Or run one at a time
python experiments/stage1/train.py --config experiments/stage1/configs/s1_linear_probe.yaml
python evaluate.py --backbone clip_h \
    --ckpt runs/stage1/s1_linear_probe/best.pt \
    --data data/datasets/CIFAKE_images/test \
    --calibrator runs/stage1/s1_linear_probe/calibrator.pkl \
    --output runs/stage1/s1_linear_probe/eval_results.json
```

## 4. Files in this Directory

| File | Purpose |
|---|---|
| `configs/s{1,2,3}_*.yaml` | Per-experiment hyperparameters |
| `train.py`                | Stage 1 training script (reads a single config) |
| `run.sh`                  | One-command run of all three experiments |
| `summarize.py`            | Merge three `eval_results.json` → `stage1_results.md` |
| `README.md`               | This file |
