# AIGC Image Detection — TikTok TechJam Track 5

Distinguish AI-generated images from real photographs under realistic post-processing conditions (JPEG compression, blur, resize, noise, color jitter, cropping).

**Final Score** = 0.50 × AUC_clean + 0.50 × AUC_robust

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
│   ├── clip_tower.py                # CLIP ViT-H/14 fine-tuned classifier
│   ├── dino_tower.py                # DINOv3 ViT-H+ fine-tuned classifier
│   ├── dual_tower.py                # Logit-average fusion of both towers
│   └── open_clip/                   # OpenCLIP source (git-ignored, installed via pip install -e)
├── weights/                         # Pretrained weights (git-ignored)
│   ├── clip_h/open_clip_pytorch_model.bin
│   └── dino_h/model.safetensors
├── calibration/
│   └── temperature_scaling.py       # Temperature scaling + ECE / Brier Score
├── configs/
│   ├── clip_h.yaml                  # CLIP-H training hyperparameters
│   ├── dino_h.yaml                  # DINO-H training hyperparameters
│   └── smoke.yaml                   # Minimal config for smoke test
├── scripts/
│   ├── download_clip.py             # Download CLIP ViT-H/14 (DFN-5B) weights
│   ├── download_dino.py             # Download DINOv3 ViT-H+ weights
│   └── smoke_test.py                # End-to-end pipeline connectivity test
├── tests/
│   └── test_real_world_transforms.py
├── runs/                            # Training outputs, TensorBoard logs (git-ignored)
├── train.py                         # Train a single tower
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

### DINOv3 ViT-H+/16 (~3.2 GB)

```bash
python scripts/download_dino.py
```

Both scripts download from `hf-mirror.com` (no HuggingFace auth required) and save to `weights/`.

---

## Smoke Test

Verify end-to-end pipeline (model loading, forward/backward pass, calibration) with synthetic data. No real dataset required.

```bash
python scripts/smoke_test.py
```

Expected: 5-stage checklist, ends with `PASSED`. Takes ~1 min on CPU.

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

```bash
# Train CLIP ViT-H/14 single tower
python train.py \
    --backbone clip_h \
    --data data/datasets/SID_Set_images \
    --output runs/clip_h \
    --epochs 10 \
    --batch-size 32

# Train DINOv3 ViT-H+ single tower
python train.py \
    --backbone dino_h \
    --data data/datasets/SID_Set_images \
    --output runs/dino_h \
    --epochs 10 \
    --batch-size 32
```

Outputs saved under `runs/<backbone>/`:

| File | Description |
|---|---|
| `best.pt` | Best checkpoint (by val AUC) |
| `calibrator.pkl` | Temperature scaler fitted on calibration split |
| `history.json` | Per-epoch train/val metrics |
| `calibration_metrics.json` | ECE / Brier Score |
| `tensorboard/` | TensorBoard event files |

### TensorBoard

Live training curves (step loss, val AUC, learning rate, hparam comparison):

```bash
tensorboard --logdir runs/
```

Then open http://localhost:6006 in your browser. Multiple runs (`clip_h`, `dino_h`) appear side-by-side for comparison.

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

Output format (one entry per image):

```json
[
  {"image_path": "/abs/path/to/img.jpg", "pred": 0.923},
  ...
]
```

`pred` is the calibrated probability that the image is AI-generated (1.0 = AI, 0.0 = real).

---

## Ablation Plan

| # | Experiment | Status |
|---|---|---|
| ① | CLIP-H single tower | P0 |
| ② | DINOv3-H+ single tower | P0 |
| ③ | H+H logit average (re-calibrated) | P1 |
| ④ | H+H feature concat | P2 — only if ③ has headroom |
| ⑤ | FFT frequency branch | P3 — optional |

Final submission uses whichever model achieves the best official Final Score.
If dual-tower fusion does not stably outperform the best single tower, the single tower is submitted.
