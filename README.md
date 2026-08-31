# TraceLens: Robust AI Image Detection Beyond Redistribution

TikTok TechJam Track 5: tell AI-generated images from real photographs under realistic resharing (JPEG compression, blur, resize, noise, color jitter, cropping).

**Final Score** = 0.50 × AUC_clean + 0.50 × AUC_robust

The submitted model is the **OpenCLIP ViT-H/14 spatial tower**, fine-tuned on WildFake. Checkpoint: `runs/spatial_tower/spatial_tower_wildfake/best.pt`. We designed a dual-tower system (CLIP spatial + RGPA forensic + gated fusion) and then removed every extra piece that did not raise official score. RGPA, fusion, and MBE stay as ablations; they are not in the demo.

To score a **hidden test folder** (no labels, no extra transforms): [Hidden-test inference](#hidden-test-inference). Interactive demo: [Run inference](#run-inference).

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

Interactive write-up: [`ablation/ablation.html`](ablation/ablation.html) (also the Streamlit Ablation tab). Serve cost of the spatial tower: [`throughput/README.md`](throughput/README.md).

---

## Robustness Evaluation Summary

Submitted model: OpenCLIP ViT-H/14 spatial tower. Protocol: official 15-condition grid on a 5,000-image WildFake val slice (50/50 real / AI). Acc / F1 use threshold 0.5. **Robust** is the mean of the 14 transformed conditions. **Official** = 0.50 × clean + 0.50 × robust.

### Clean vs transformed

| | AUC | AP | Acc | F1 | EER |
|---|---:|---:|---:|---:|---:|
| Clean | 0.9966 | 0.9968 | 0.9392 | 0.9354 | 0.0298 |
| Robust (14 transforms) | 0.9882 | 0.9893 | 0.9238 | 0.9183 | 0.0579 |
| **Official** | **0.9924** | **0.9931** | **0.9315** | **0.9269** | **0.0438** |

AUC drops **0.0084** from clean to the mean of transformed images. Ranking (AUC / AP) stays high; Acc / F1 drop more because the 0.5 threshold is conservative (clean precision 0.997, recall 0.881).

```
AUC  0.88                         1.00
     |                              |
Clean| ████████████████████████████ | 0.9966
Robust| ███████████████████████████ | 0.9882
Official| ███████████████████████████ | 0.9924
```

Bar length is linear on [0.88, 1.00].

### Per transform (vs clean)

Each row is one official degradation, applied alone. Δ is versus clean AUC.

| Condition | AUC | Δ vs clean | Acc |
|---|---:|---:|---:|
| Clean | 0.9966 | — | 0.9392 |
| JPEG 90 | 0.9964 | −0.0002 | 0.9356 |
| JPEG 70 | 0.9955 | −0.0011 | 0.9258 |
| JPEG 50 | 0.9938 | −0.0028 | 0.9306 |
| JPEG 30 | 0.9902 | −0.0064 | 0.9260 |
| Blur 0.5 | 0.9958 | −0.0008 | 0.9370 |
| Blur 1.0 | 0.9871 | −0.0095 | 0.9242 |
| Blur 2.0 | 0.9766 | −0.0200 | 0.9056 |
| Resize 1/2 | 0.9939 | −0.0027 | 0.9352 |
| Resize 1/4 | 0.9871 | −0.0095 | 0.9228 |
| Noise 2% | 0.9811 | −0.0155 | 0.9186 |
| Noise 5% | 0.9816 | −0.0150 | 0.9080 |
| **Noise 10% (worst)** | **0.9707** | **−0.0259** | **0.8998** |
| Color | 0.9938 | −0.0028 | 0.9374 |
| Crop 80% | 0.9914 | −0.0052 | 0.9264 |

Mild JPEG, light blur, color jitter, and 80% crop stay within ~0.005 AUC of clean. The grid is hardest on Gaussian noise and strong blur. No cell collapses: worst-condition AUC is 0.9707.

---

## Error Analysis

Submitted detector: OpenCLIP ViT-H/14 spatial model. Errors are counted on the 5,000-image WildFake val slice (2,500 real / 2,500 AI) at threshold 0.5. Fake = AI-generated. A **false positive** flags a real photo as AI; a **false negative** lets an AI image through as real.

| | Clean |
|---|---:|
| False positives | 6 |
| False negatives | 298 |
| Precision @ 0.5 | 0.997 |
| Recall @ 0.5 | 0.881 |

### False positives: rare, high-confidence when they happen

On clean images the model almost never accuses a real photo: **6 FP out of 2,500 reals (0.24%)**. Those six are not borderline — the strongest has P(AI) = 0.91. CLIP's global semantics can still read a real scene as generated when lighting, texture, or composition looks synthetic.

Heavy resharing inflates this class. Gaussian noise at 10% raises FP to **64**; blur σ=2.0 raises it to **43**. Additive grain and strong blur are the transforms that make real photos look least photographic to the tower.

### False negatives: the dominant error

**298 of 2,500 AI images** are missed on clean (11.9%). The worst misses are not near the threshold: several score P(AI) ≈ 0. Photorealistic generators (especially later diffusion / Midjourney-style images in WildFake) land on the same semantic manifold as real photos, and a spatial CLIP tower has little local forensic signal to fall back on.

Noise and blur also drive FN up (437 and 429). That matches the official grid: Noise 10% is the worst cell (AUC 0.9707). The model still ranks well; at 0.5 it simply stays on the "call it real" side of the logit.

### Where errors grow

Same 5k slice, threshold 0.5.

| Condition | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|
| Clean | 6 | 298 | 0.9973 | 0.8808 |
| Color | 11 | 302 | 0.9950 | 0.8792 |
| JPEG 30 | 10 | 360 | 0.9953 | 0.8560 |
| Crop 80% | 16 | 352 | 0.9926 | 0.8592 |
| Resize 1/4 | 21 | 365 | 0.9903 | 0.8540 |
| Blur 2.0 | 43 | 429 | 0.9797 | 0.8284 |
| **Noise 10%** | **64** | **437** | **0.9699** | **0.8252** |

JPEG / color / crop barely move FP. Noise and strong blur move both FP and FN.

### Trade-offs

**Threshold 0.5 is conservative on purpose.** Precision 0.997 vs recall 0.881. For a platform, accusing a real photo is worse than missing some AI. Lowering the threshold would cut FN and raise FP; we did not retune it on this val slice.

- **Spatial-only vs a forensic second tower.** RGPA matches Spatial on SID, then drops to official 0.906 on WildFake. Gated fusion still mixes in ~25% RGPA and loses most on noise. The submitted model is Spatial alone — fewer FPs under noise than the fused system, at the cost of no local-residual fallback on photoreal fakes.
- **Ranking vs operating point.** Official AUC 0.9924 looks almost saturated; Acc 0.9315 does not, because FN dominate at 0.5. AUC describes order; the product call is the 0.5 cut.
- **Robustness training vs extra eval enhancement.** Single-degradation training (30% clean) stays. MBE at eval hurts blur and crop and is not used. We accept a small clean/robust AUC gap (0.0084) rather than a preprocess the tower never saw.
- **Capacity vs speed.** Truncating ViT-H to 8 layers keeps clean AUC ~0.99 but collapses robust AUC to ~0.79. The 632M vision tower is what holds the grid; a smaller forensic substitute only sped the pipeline 22% and lost 0.087 AUC.

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

After clone, pull the submitted checkpoint (Git LFS, ~2.4 GB). If this file is a tiny text pointer instead of a real weight file, inference will fail:

```bash
git lfs install
git lfs pull
ls -lh runs/spatial_tower/spatial_tower_wildfake/best.pt
```

You do **not** need SID-Set, WildFake, or any training data to run inference.

---

## Hidden-test inference

Point `infer.py` at a directory of unlabeled images. The script recursively finds `jpg` / `jpeg` / `png` / `webp` / `bmp`, scores each image **as-is** (no JPEG / blur / crop added at test time), and writes contest JSON.

Run from the **repo root** so the default checkpoint path resolves.

```bash
source ~/techjam/venv/bin/activate   # or your own venv with requirements.txt + open_clip
cd /path/to/tiktok_bytecop

python infer.py \
    --input  /path/to/hidden_test \
    --output predictions.json
```

Default `--ckpt` is `runs/spatial_tower/spatial_tower_wildfake/best.pt`. Leave `--temperature` at `1.0`. A GPU is recommended (first load ~2.4 GB VRAM; batch 32 is the default). CPU works but is slow.

```bash
# optional: pick a free GPU on a shared box
CUDA_VISIBLE_DEVICES=0 python infer.py --input /path/to/hidden_test --output predictions.json

# larger folders
python infer.py --input /path/to/hidden_test --output predictions.json --batch-size 64 --workers 8
```

Output (`predictions.json`):

```json
[
  {"image_path": "/path/to/hidden_test/0001.jpg", "pred": 0.923001},
  {"image_path": "/path/to/hidden_test/0002.png", "pred": 0.041882}
]
```

`pred` is **P(AI-generated)** in `[0, 1]`. A decision at 0.5 is `fake` if `pred >= 0.5` else `real`; the official file is probabilities, not hard labels. `image_path` is the resolved absolute path.

| Need | Do not need |
|---|---|
| `requirements.txt` + `pip install -e models/open_clip` | SID-Set / WildFake / CIFAKE |
| Git LFS `best.pt` | Labels, `real/` / `fake/` layout |
| A directory of images | `evaluate.py` (that script is the labeled 15-condition grid) |
| GPU strongly recommended | RGPA / fusion checkpoints |

`--full` also writes `logit` and `label` for debugging. Single-image Python / HTTP: [`serve/README.md`](serve/README.md).

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

Hidden-test / contest directory scoring: [Hidden-test inference](#hidden-test-inference). `--full` also writes `logit` / `label`. HTTP API and Python import: [`serve/README.md`](serve/README.md).

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
