# AIGC Image Detection — TikTok TechJam Track 5

Distinguish AI-generated images from real photographs under realistic post-processing conditions (JPEG compression, blur, resize, noise, color jitter, cropping).

**Final Score** = 0.50 × AUC_clean + 0.50 × AUC_robust

Architecture: **OpenCLIP-H spatial tower + optional RGPA forensic branch**, fused by standardized weighted logits. See `AI图像检测_技术方案.md` and `取证分支方案_RGPA.md`.

---

## Project Structure

```
tiktok_bytecop/
├── data/
│   ├── datasets/                    # Raw datasets (git-ignored)
│   │   ├── SID_Set/                 # Parquet shards from HuggingFace
│   │   ├── SID_Set_images/          # Extracted image folders (train/val/calibration)
│   │   └── WildFake/                # Cross-generator generalization test
│   ├── dataset.py                   # AIGCDataset — directory & manifest loader
│   ├── transforms.py                # Official 6 transforms + training augmentation policy
│   └── prepare_sid_set.py           # Convert SID_Set parquet → image folders
├── models/
│   ├── clip_tower.py                # OpenCLIP ViT-H/14 spatial classifier
│   ├── rgpa.py                      # RGPA forensic branch (SRM-inspired residual)
│   ├── dual_tower.py                # Standardized weighted CLIP + RGPA logit fusion
│   └── open_clip/                   # OpenCLIP source (git-ignored, installed via pip)
├── weights/                         # Pretrained weights (git-ignored)
│   └── clip_h/open_clip_pytorch_model.bin
├── calibration/
│   └── temperature_scaling.py       # Temperature scaling + ECE / Brier Score
├── configs/
│   └── smoke.yaml                   # Minimal config for smoke test
├── experiments/
│   ├── spatial_tower/               # OpenCLIP-H spatial tower
│   └── forensic_tower/              # RGPA forensic tower
├── scripts/
│   ├── download_clip.py             # Download CLIP ViT-H/14 (DFN-5B) weights
│   └── smoke_test.py                # End-to-end pipeline connectivity test
├── evaluate.py                      # Official 15-condition robustness matrix
├── infer.py                         # Batch inference → JSON output
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt

# Install OpenCLIP from local clone (editable mode)
pip install -e models/open_clip
```

If `models/open_clip/` doesn't exist yet:

```bash
git clone https://github.com/mlfoundations/open_clip.git models/open_clip
pip install -e models/open_clip
```

Verify:

```bash
python -c "import open_clip; print(open_clip.__version__)"
```

---

## Pretrained Weights

### CLIP ViT-H/14 (DFN-5B, ~3.9 GB)

```bash
python scripts/download_clip.py
```

RGPA is trained from scratch (tens of thousands of parameters). It does not need a pretrained checkpoint.

---

## Smoke Test

Verify end-to-end pipeline (model loading, forward/backward pass, calibration) with synthetic data. No real dataset required.

```bash
python scripts/smoke_test.py --backbone clip_h
python scripts/smoke_test.py --backbone rgpa
```

Expected: 5-stage checklist, ends with `PASSED`. CLIP takes ~1 min; RGPA is much faster.

---

## Datasets

### SID_Set (main training data)

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 hf download saberzl/SID_Set \
    --repo-type dataset \
    --local-dir data/datasets/SID_Set

python data/prepare_sid_set.py \
    --src  data/datasets/SID_Set \
    --dest data/datasets/SID_Set_images \
    --delete-parquet
```

Output layout:

```
data/datasets/SID_Set_images/
├── train/
│   ├── real/
│   └── fake/
├── val/
│   ├── real/
│   └── fake/
└── calibration/
    ├── real/
    └── fake/
```

### WildFake (cross-generator generalization test)

Hosted on ModelScope. Download after SID_Set training is complete.

```bash
pip install modelscope
python -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('hy2628982280/WildFake', cache_dir='data/datasets/WildFake')
"
```

---

## Training

Two towers train independently. `batch_size` in each YAML is **per GPU**.

```bash
# Spatial tower — 4 GPUs
torchrun --standalone --nproc_per_node=4 experiments/spatial_tower/train.py \
    --config experiments/spatial_tower/configs/spatial_tower.yaml

# Forensic tower — 2 GPUs
torchrun --standalone --nproc_per_node=2 experiments/forensic_tower/train.py \
    --config experiments/forensic_tower/configs/forensic_tower.yaml

# Single GPU (still works)
python experiments/spatial_tower/train.py \
    --config experiments/spatial_tower/configs/spatial_tower.yaml
```

Outputs saved under `runs/spatial_tower/<name>/` or `runs/forensic_tower/<name>/`:

| File | Description |
|---|---|
| `best.pt` | Best checkpoint (by val AUC) |
| `calibrator.pkl` | Temperature scaler fitted on calibration split |
| `history.json` | Per-epoch train/val metrics |
| `calibration_metrics.json` | ECE / Brier Score |
| `tensorboard/` | TensorBoard event files |

### TensorBoard

```bash
tensorboard --logdir runs/
```

Then open http://localhost:6006 in your browser.

---

## Evaluation

Runs all 15 official conditions and computes the final score.

```bash
python evaluate.py \
    --backbone clip_h \
    --ckpt runs/clip_h/best.pt \
    --data data/datasets/SID_Set_images/val \
    --calibrator runs/clip_h/calibrator.pkl \
    --output runs/clip_h/eval_results.json
```

`--backbone` is one of `clip_h`, `rgpa`.

Example output:

```
AUC_clean  = 0.9830
AUC_robust = 0.9612  (mean over 14 conditions)
Final Score = 0.9721
Worst condition AUC = 0.9201
```

---

## Inference

```bash
# Spatial tower
python infer.py \
    --backbone clip_h \
    --ckpt runs/clip_h/best.pt \
    --calibrator runs/clip_h/calibrator.pkl \
    --input /path/to/images \
    --output predictions.json

# CLIP + RGPA (standardized weighted logit fusion)
python infer.py \
    --backbone dual \
    --clip-ckpt runs/clip_h/best.pt \
    --rgpa-ckpt runs/rgpa/best.pt \
    --calibrator runs/dual/calibrator.pkl \
    --input /path/to/images \
    --output predictions.json
```

Output format (one entry per image):

```json
[
  {"image_path": "/abs/path/to/img.jpg", "pred": 0.923},
  ...
]
```

`pred` is the calibrated probability that the image is AI-generated (1.0 = AI, 0.0 = real).

---

## Experiment Plan

| # | Experiment | Status |
|---|---|---|
| ① | OpenCLIP-H spatial single tower | P0 Stage 1 |
| ② | RGPA (patch encoding + bidirectional aggregation) | P1 Stage 2 |
| ③ | Standardized weighted logit fusion | P2 Stage 3A |
| ④ | Feature concat | P2 Stage 3B — only if ③ has headroom |

OpenCLIP-H is the minimum deliverable. RGPA / fusion enter the final model only if they improve Final Score without clearly hurting the worst degradation condition.
