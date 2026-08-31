# TraceLens: Robust AI-Generated Image Detection Under Real-World Transformations

TikTok TechJam Track 5: tell AI-generated images from real photographs under realistic resharing (JPEG compression, blur, resize, noise, color jitter, cropping).

**Final Score** = 0.50 × AUC_clean + 0.50 × AUC_robust

The submitted model is the **OpenCLIP ViT-H/14 spatial tower**, fine-tuned on WildFake. Checkpoint: `runs/spatial_tower/spatial_tower_wildfake/best.pt`. We designed a dual-tower system (CLIP spatial + RGPA forensic + gated fusion) and then removed every extra piece that did not raise official score. RGPA, fusion, and MBE stay as ablations; they are not in the demo.

Interactive inference is Streamlit: upload a photo, stack official degradations, click Detect. See [Run inference](#run-inference).

---

## Approach

Two difficulties dominate AIGC detection:

- **Generalization.** A new generator leaves a fingerprint that need not match the training set.
- **Robustness.** Images on a real sharing path are compressed, cropped, blurred, and recolored. Those operations destroy part of the signal a detector relies on.

We started from a two-branch idea, inspired by AIDE-style global semantics plus local forensics. The **spatial branch** reads content and structure from RGB. The **forensic branch** was meant to read pixel-level generation traces from a high-pass residual. The two branches train independently; Stage 3 tests whether their logits are complementary.

```text
shared RGB image
    |-> official OpenCLIP normalize -> OpenCLIP-H -> spatial logit
    |-> full-image SRM-inspired residual -> RGPA -> forensic logit
                              |
         standardized / gated logit fusion -> calibrated P(AI)
```

**Rule from day one:** OpenCLIP-H is the starting point and the fallback. RGPA is an optional local-forensics increment. Fusion enters the submitted model only if it stably raises Final Score. Feature concat was never a minimum deliverable.

### Spatial branch (OpenCLIP-H)

OpenCLIP ViT-H/14 (`laion2b_s32b_b79k`). Vision tower ~632M; do not count an unused text tower against the <2B cap.

```text
RGB image
  -> keep clean, or apply one official degradation
  -> Resize 224 + CenterCrop 224
  -> official OpenCLIP channel normalize
  -> ViT-H/14 vision tower
  -> classification head
  -> spatial logit
```

Training recipe:

1. Linear probe (frozen backbone) to check that pretrained CLIP already has a real vs AI margin.
2. Unfreeze the last **2** of 32 transformer blocks plus `ln_post`, CLIP projection, and the head. This is the submitted setting.
3. Optional last-4 unfreeze was not needed; last-2 already saturates train-loop val.

Stage 1 trains on SID-Set (learn "is this AI-generated?"). Stage 2 fine-tunes on WildFake (open-source generators and in-the-wild images). CIFAKE is 32x32 pipeline smoke only; upsampling does not restore missing high-frequency detail.

The checkpoint is taken at **minimum val_loss**, not at a later AUC peak.

### Forensic branch (RGPA) — built, then dropped

Residual-Guided Patch Aggregation answers: after a **fixed full-image high-pass residual**, can shared patch encoding plus bidirectional high/low aggregation give a gain independent of OpenCLIP-H?

- Input: 224x224 RGB (pixel-scale; **not** CLIP normalize).
- Patch size 32x32 → 7x7 = 49 patches.
- SRM-inspired high-pass runs on the **full image before the split** (per-patch padding would invent fake borders). The kernel stays frozen.
- Residual energy is standardized **inside the image**; softmax(+a/τ) and softmax(-a/τ) softly weight high- vs low-residual patches. No patch is hard-dropped.
- RGPA trains from scratch on the same split and the same single-degradation distribution as Spatial.

On SID, RGPA looks interchangeable with Spatial (train-loop clean/robust ≈ 1.0). On WildFake it does not transfer: train-loop robust 0.936 vs Spatial 0.997; official 15-grid **0.9064**. JPEG, blur, resize, and noise attack a high-pass residual directly. That is why Clean AUC alone was never a keep criterion.

### Shared training

Both branches used the same official single-degradation family. The contest applies **one** transform per image, never a stack, so training does the same: 30% clean + 70% one random official transform (real and fake share the sampler, after DDA, so the model cannot cheat on "was this processed?").

| Transform | Parameters | Real-world analogue |
|---|---|---|
| JPEG | quality 90 / 70 / 50 / 30 | Social re-encode, messaging |
| Gaussian blur | σ 0.5 / 1.0 / 2.0 | Defocus, screenshot smoothing |
| Resize (down then up) | 0.5x / 0.25x, then back | Thumbnails, CDN resize |
| Gaussian noise | σ 0.02 / 0.05 / 0.10 | Sensor grain |
| Color jitter | ±20% | Filter apps, auto-enhance |
| Center crop | 80% | Avatar crop, reframing |

Eval covers every cell. Official score is 0.5 × clean + 0.5 × mean of the 14 transformed cells.

### Fusion (Stage 3) — built, then dropped

Towers stay frozen. A small gated MLP (`models/gated_fusion.py`) mixes the two logits. Logits are not probabilities; scales differ, so a naive average is not a fair fusion. Keep fusion only if Final Score rises and the worst cell does not collapse.

What we measured instead: CLIP is far stronger than RGPA on WildFake, but the gate still mixes in ~25% RGPA. Under noise the residual is gone and that mix **hurts**. Official fusion 0.9900 vs Spatial 0.9924.

---

## Ablation: from dual tower to spatial only

Keep a module only if it raises official score on the **same** 5,000-image WildFake val slice (50/50 real / AI). MBE is an eval-only preprocess in front of Spatial, not a second tower.

| System | Official AUC | vs Spatial |
|---|---:|---:|
| **Spatial (submitted)** | **0.9924** | — |
| Spatial + MBE | 0.9901 | −0.0023 |
| Gated fusion (CLIP + RGPA) | 0.9900 | −0.0024 |
| RGPA alone | 0.9064 | −0.0860 |

Submitted Spatial on that slice (threshold 0.5):

| | AUC | AP | Acc | F1 | EER |
|---|---:|---:|---:|---:|---:|
| Clean | 0.9966 | 0.9968 | 0.9392 | 0.9354 | 0.0298 |
| Robust (14 transforms) | 0.9882 | 0.9893 | 0.9238 | 0.9183 | 0.0579 |
| **Official** | **0.9924** | **0.9931** | **0.9315** | **0.9269** | **0.0438** |

Worst cell: **Noise 10%** (AUC 0.9707). No cell collapses. At 0.5 the model is conservative (clean precision 0.997, recall 0.881).

Family-level AUC vs Spatial (negative means the add-on did not help):

| Condition | Spatial | Fusion | Δ Fusion | Spatial+MBE | Δ MBE | RGPA |
|---|---:|---:|---:|---:|---:|---:|
| Clean | 0.9966 | 0.9956 | −0.0010 | 0.9950 | −0.0016 | 0.9260 |
| JPEG 30 | 0.9902 | 0.9880 | −0.0022 | 0.9896 | −0.0006 | 0.9226 |
| Blur (mean) | 0.9865 | 0.9861 | −0.0004 | 0.9782 | −0.0083 | 0.8803 |
| Resize (mean) | 0.9905 | 0.9903 | −0.0002 | 0.9876 | −0.0029 | 0.8809 |
| **Noise (mean)** | **0.9778** | **0.9627** | **−0.0151** | 0.9783 | +0.0005 | 0.8433 |
| Crop 80% | 0.9914 | 0.9915 | +0.0001 | 0.9845 | −0.0069 | 0.9167 |

Fusion fails most on noise. MBE fails most on blur and crop. SID train-loop numbers hid this: both towers look saturated there; WildFake is the domain that decides.

What we kept from the proposal:

- Official 6 degradations at **train** time (the robustness recipe).
- CLIP-H, last-2 unfreeze, min val_loss, WildFake fine-tune.

What we did not submit:

- RGPA second tower
- Gated / weighted logit fusion
- MBE at eval
- Feature concat, native-resolution patches, stacked degradations as the selection metric

Interactive write-up: [`ablation/ablation.html`](ablation/ablation.html) (also the Streamlit Ablation tab). Grid and errors: [`ablation/robustness_evaluation.md`](ablation/robustness_evaluation.md), [`ablation/error_analysis.md`](ablation/error_analysis.md). Serve cost of the spatial tower: [`throughput/README.md`](throughput/README.md).

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
│   ├── mbe.py                       # Mean Bias Error (ablation preprocess)
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

Training scripts expect an image tree: `train|val|calibration/{real,fake}/`. Raw parquet / zip cannot be used as-is. Paths, counts, and how data couples to throughput: [`data/README.md`](data/README.md).

| Dataset | Train | Val | Role |
|---|---:|---:|---|
| SID-Set | 119,590 | 23,943 | Spatial stage 1 |
| WildFake | 3,238,953 | 66,069 | Spatial stage 2 (submitted weights) |
| CIFAKE | 100,000 | 10,000 | Pipeline smoke only |

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

RGPA and gated fusion remain runnable as ablations (`experiments/forensic_tower/`, `experiments/fusion/`). They are not loaded by Streamlit or `infer.py`.

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
