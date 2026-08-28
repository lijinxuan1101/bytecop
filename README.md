# AIGC Image Detection — TikTok TechJam Track 5

Distinguish AI-generated images from real photographs under realistic post-processing conditions (JPEG compression, blur, resize, noise, color jitter, cropping).

**Final Score** = 0.50 × AUC_clean + 0.50 × AUC_robust

---

## Project Structure

```
tiktok_bytecop/
├── data/
│   ├── datasets/               # Raw datasets (not committed to git)
│   │   ├── SID_Set/            # Main training data (HuggingFace parquet)
│   │   └── WildFake/           # Cross-generator generalization test
│   ├── dataset.py              # AIGCDataset — directory & manifest loader
│   └── transforms.py           # Official transforms + training augmentation policy
├── models/
│   ├── clip_tower.py           # CLIP ViT-H/14 fine-tuned classifier
│   ├── dino_tower.py           # DINOv3 ViT-H+ fine-tuned classifier
│   └── dual_tower.py           # Logit-average fusion of both towers
├── calibration/
│   └── temperature_scaling.py  # Temperature scaling + ECE / Brier Score
├── tests/
│   └── test_real_world_transforms.py
├── configs/
│   ├── clip_h.yaml             # CLIP-H training hyperparameters
│   └── dino_h.yaml             # DINO-H training hyperparameters
├── train.py                    # Train a single tower
├── evaluate.py                 # Official 15-condition robustness matrix
├── infer.py                    # Batch inference → JSON output
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Datasets

### SID_Set (main training data)

Hosted on HuggingFace as 249 parquet shards (~30 万 images).
Labels: `Real` (0) / `Full Synthetic` (1) / `Tampered` (ignored during training).

```bash
HF_ENDPOINT=https://hf-mirror.com hf download saberzl/SID_Set \
    --repo-type dataset \
    --local-dir data/datasets/SID_Set
```

After downloading, convert parquet shards to an image directory tree:

```bash
python data/prepare_sid_set.py \
    --src  data/datasets/SID_Set \
    --dest data/datasets/SID_Set_images
```

Expected output layout:

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

```bash
# Train CLIP ViT-H/14 tower
python train.py \
    --backbone clip_h \
    --data data/datasets/SID_Set_images \
    --output runs/clip_h \
    --epochs 10 \
    --batch-size 32

# Train DINOv3 ViT-H+ tower
python train.py \
    --backbone dino_h \
    --data data/datasets/SID_Set_images \
    --output runs/dino_h \
    --epochs 10 \
    --batch-size 32
```

Checkpoints and calibrators are saved under `runs/<backbone>/`:

| File | Description |
|---|---|
| `best.pt` | Best checkpoint by val AUC |
| `calibrator.pkl` | Temperature scaler fitted on calibration split |
| `history.json` | Per-epoch training metrics |
| `calibration_metrics.json` | ECE / Brier Score |

---

## Evaluation

```bash
# Single tower
python evaluate.py \
    --backbone clip_h \
    --ckpt runs/clip_h/best.pt \
    --data data/datasets/SID_Set_images/test \
    --calibrator runs/clip_h/calibrator.pkl \
    --output runs/clip_h/eval_results.json
```

Output includes AUC for all 15 official conditions and the final score:

```
AUC_clean  = 0.9830
AUC_robust = 0.9612  (mean over 14 conditions)
Final Score = 0.9721
Worst condition AUC = 0.9201
```

---

## Inference

```bash
# Single tower
python infer.py \
    --backbone clip_h \
    --ckpt runs/clip_h/best.pt \
    --calibrator runs/clip_h/calibrator.pkl \
    --input /path/to/images \
    --output predictions.json

# Dual tower (logit average)
python infer.py \
    --backbone dual \
    --clip-ckpt runs/clip_h/best.pt \
    --dino-ckpt runs/dino_h/best.pt \
    --calibrator runs/dual/calibrator.pkl \
    --input /path/to/images \
    --output predictions.json
```

Output format:

```json
[
  {"image_path": "/abs/path/to/img.jpg", "pred": 0.923},
  ...
]
```

`pred` is the calibrated probability that the image is AI-generated (1 = AI, 0 = real).

---

## Ablation Plan

| Experiment | Purpose |
|---|---|
| ① CLIP-H single tower | Language-supervised H-level baseline |
| ② DINOv3-H+ single tower | Self-supervised H-level baseline |
| ③ H+H logit average | Main fusion; re-calibrated after fusion |
| ④ H+H feature concat | Only if ③ still has headroom and budget allows |
| ⑤ FFT frequency branch | Optional; verify whether frequency signal adds value |

The final submission uses the model with the best official Final Score.
If dual-tower fusion does not stably outperform the best single tower, the
single tower is submitted instead.
