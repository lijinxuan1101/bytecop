# ByteCop — AIGC Image Detection

TikTok TechJam Track 5: tell AI-generated images from real photographs under realistic resharing (JPEG compression, blur, resize, noise, color jitter, cropping).

**Final Score** = 0.50 × AUC_clean + 0.50 × AUC_robust

The submitted model is the **OpenCLIP-H spatial tower** (ViT-H/14). The only weight file in this repo is `runs/spatial_tower/spatial_tower_wildfake/best.pt`. RGPA and gated fusion stay as ablations and are not in the demo.

Interactive inference is Streamlit: upload a photo, stack official degradations, click Detect. See [Run inference](#run-inference).

---

## Project structure

```
tiktok_bytecop/
├── viz/
│   └── app.py                       # Streamlit UI (Detect + Ablation)
├── serve/
│   ├── spatial_backend.py           # SpatialDetector — shared by UI and CLI
│   └── app.py                       # Optional FastAPI (/v1/score, /v1/score-dir)
├── models/
│   ├── clip_tower.py                # OpenCLIP ViT-H/14 spatial classifier
│   ├── rgpa.py                      # RGPA forensic branch (ablation)
│   ├── dual_tower.py                # Weighted CLIP + RGPA fusion (ablation)
│   └── gated_fusion.py              # Gated logit fusion (ablation)
├── data/
│   ├── dataset.py                   # AIGCDataset
│   ├── transforms.py                # Official 6 degradations + training aug
│   ├── mbe.py                       # Mean Bias Error
│   ├── prepare_sid_set.py           # SID_Set parquet → image folders
│   └── datasets/                    # Raw data (git-ignored)
├── experiments/
│   ├── spatial_tower/               # Submitted model: OpenCLIP-H
│   ├── forensic_tower/              # RGPA (ablation)
│   └── fusion/                      # Frozen towers + GatedFusion (ablation)
├── calibration/
│   └── temperature_scaling.py
├── ablation/
│   └── ablation.html                # Ablation report (embedded in Streamlit)
├── scripts/
│   ├── download_clip.py
│   └── smoke_test.py
├── configs/
│   └── smoke.yaml
├── .streamlit/
│   └── config.toml                  # Theme, upload cap, runOnSave
├── runs/spatial_tower/spatial_tower_wildfake/best.pt   # Submitted ckpt (Git LFS)
├── infer.py                         # Directory batch scoring → contest JSON
├── evaluate.py                      # Official 15-condition robustness matrix
└── requirements.txt
```

More detail: data [`data/README.md`](data/README.md), training [`train.md`](train.md), experiments [`experiments/README.md`](experiments/README.md), UI [`viz/README.md`](viz/README.md), backend [`serve/README.md`](serve/README.md).

---

## Installation

```bash
source ~/techjam/venv/bin/activate
pip install -r requirements.txt

# Install OpenCLIP from the local clone (editable)
pip install -e models/open_clip
```

If `models/open_clip/` is missing:

```bash
git clone https://github.com/mlfoundations/open_clip.git models/open_clip
pip install -e models/open_clip
```

Verify:

```bash
python -c "import open_clip; print(open_clip.__version__)"
```

---

## Run inference

Start from the **repo root** so Streamlit picks up `.streamlit/config.toml` (theme, 50 MB upload cap, rerun on save).

```bash
source ~/techjam/venv/bin/activate
cd /home/xuting/tiktok_bytecop
streamlit run viz/app.py --server.port 8508
```

Open http://localhost:8508. Stop with `Ctrl+C`.

### Using the UI

1. **Detect**: drop or pick a photo (jpg / png / webp / bmp / tiff).
2. Left **Adjustments** stack official degradations (crop, resample, blur, noise, color, JPEG) in resharing order.
3. Click **Detect** to score the original; if anything is adjusted, the degraded copy is scored too.
4. **Ablation** embeds `ablation/ablation.html`.

Without a GPU the Detect button is disabled; upload and adjustments still work. With a GPU, the first Detect loads `runs/spatial_tower/spatial_tower_wildfake/best.pt` onto `cuda:1` (~2.4 GB; falls back to `cuda:0` if that card is missing) and reuses it afterwards.

`runOnSave = true`: save `viz/app.py` and the browser reruns; no process restart.

### Batch / CLI (optional)

Directory → contest JSON (`[{image_path, pred}, ...]`, `pred` is P(AI)):

```bash
python infer.py --input /path/to/images --output predictions.json
```

`--full` also writes `logit` / `label`. HTTP API and Python import: [`serve/README.md`](serve/README.md).

---

## Datasets

Training scripts expect an image tree: `train|val|calibration/{real,fake}/`. Raw parquet / zip cannot be used as-is. Paths and extraction: [`data/README.md`](data/README.md).

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

### WildFake (cross-generator generalization)

```bash
pip install modelscope
python -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('hy2628982280/WildFake', cache_dir='data/datasets/WildFake')
"
```

---

## Training

Full procedure (machine, data, commands, outputs): [`train.md`](train.md). Towers train independently; `batch_size` is **per GPU**. Submitted model is the spatial tower:

```bash
# 4 GPUs
torchrun --standalone --nproc_per_node=4 experiments/spatial_tower/train.py \
    --config experiments/spatial_tower/configs/spatial_tower.yaml

# Single GPU
python experiments/spatial_tower/train.py \
    --config experiments/spatial_tower/configs/spatial_tower.yaml
```

Outputs land in `runs/spatial_tower/<name>/`: `best.pt`, `calibrator.pkl`, `history.json`, `tensorboard/`.

```bash
tensorboard --logdir runs/
```

---

## Evaluation

Runs all 15 official conditions and computes the final score:

```bash
python evaluate.py \
    --backbone clip_h \
    --ckpt runs/spatial_tower/spatial_tower_wildfake/best.pt \
    --data data/datasets/SID_Set_images/val \
    --output runs/spatial_tower/eval_results.json
```

`--backbone` is `clip_h` or `rgpa`.

Smoke test (synthetic data, no real dataset):

```bash
python scripts/smoke_test.py --backbone clip_h
python scripts/smoke_test.py --backbone rgpa
```

---

## Experiment plan

| # | Experiment | Status |
|---|---|---|
| 1 | OpenCLIP-H spatial single tower | **Submitted model** (WildFake `best.pt`) |
| 2 | RGPA forensic tower | Ablation — weaker, not shipped |
| 3 | Gated logit fusion | Ablation — below spatial on official val slice |

The UI and contest inference both go through the spatial backend in `serve/`.
