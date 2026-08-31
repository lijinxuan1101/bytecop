# Data

This directory is the data layer of the detector: three public datasets, the official six degradations, and the train/eval loaders.

The submitted model is **OpenCLIP ViT-H/14 Spatial** (WildFake fine-tune, `runs/spatial_tower/spatial_tower_wildfake/best.pt`). How we store and load images decides two things:

1. **Detection.** Stage 1 on SID-Set learns "is this AI-generated?". Stage 2 on WildFake covers open-source generators and in-the-wild images. CIFAKE is pipeline smoke only.
2. **Serving throughput.** At inference the bottleneck is the ViT-H forward pass, not disk I/O. At training the bottleneck is **JPEG decode**. WildFake val images average **~44 KB** (social-thumbnail size) and are the default serve-benchmark input. A 1024x1024 original raises decode from ~2 ms to ~15 ms, so CPU provisioning has to follow image size.

Image trees are gitignored (see the repo-root `.gitignore`). This README tracks scripts, loaders, and operational contracts only.

---

## How data connects to the model and to throughput

```
JPEG on disk
    |  AIGCDataset: path list + PIL decode + official 6 degradations
    v
224x224 RGB tensor (CLIP mean/std)
    |  Spatial: OpenCLIP ViT-H/14; ~1.9 GB on an A40 at batch=64
    v
logit / P(AI)
```

| Stage | Data fact | Throughput implication |
|---|---|---|
| Train | SID / WildFake are mostly high-res JPEG; `num_workers=12` | DataLoader decode dominates wall time, not the GPU |
| Eval | WildFake val is **66,069**; condition sweeps often use a 5k slice | Full val ~25 min / GPU; 15 conditions x 5k can run in parallel |
| Serve bench | Sampled from WildFake val, mean **44 KB** | Decode ~2.2 ms; naive **31.4 img/s** per A40, **44.6 img/s** after micro-batching |
| Full-res traffic | 1024x1024 JPEG | Decode ~15.1 ms; ~0.7 CPU cores per GPU for decode |
| Index | Parquet is used only by `prepare_*.py` to build path lists | **Train and serve never read parquet**; metadata scans would kill throughput |

Serve numbers and the rejected optimizations are in [`throughput/README.md`](../throughput/README.md): micro-batch 1.42x, ~**357 img/s** on 8x A40, tens of millions of images per day. `torch.compile`, larger batches, and a smaller forensic backbone were measured and are not the right path here.

---

## Three datasets (counts from disk)

Paths are relative to `data/datasets/`. Labels: `0` = real, `1` = AI-generated.

| Dataset | Train | Val | Train real / fake | Val real / fake | Role |
|---|---:|---:|---|---|---|
| **SID-Set** | 119,590 | 23,943 | 59,802 / 59,788 | 11,984 / 11,959 | Spatial stage 1: learn real vs AI-generated |
| **WildFake** | 3,238,953 | 66,069 | 732,794 / 2,506,159 | 14,956 / 51,113 | Spatial stage 2: fine-tune that produces the submitted weights |
| **CIFAKE** | 100,000 | 10,000 | 50,000 / 50,000 | 5,000 / 5,000 | Smoke: 32x32; not evidence of generalization |

SID-Set also has a calibration split: `real` 8,214 + `fake` 8,253 (`tampered` is skipped). WildFake counts come from the generator-aware path lists; the empty `wukong/` tree is dropped.

Official score (same degradations as this data layer):

```
Official = 0.5 x Clean + 0.5 x mean(14 degradation cells)
```

Submitted Spatial on a WildFake 5k val slice (threshold 0.5): Clean AUC **0.9966**, Robust **0.9882**, Official **0.9924**. Cell-level tables: [`ablation/robustness_evaluation.md`](../ablation/robustness_evaluation.md).

---

## Layout

```
data/
├── README.md                 this file
├── dataset.py                AIGCDataset + CLIP preprocess
├── transforms.py             official 6 degradations + 30/70 train policy
├── type_balanced_sampler.py  WildFake: balance fake side by generator
├── mbe.py                    multi-class balanced error (analysis only)
├── prepare_sid_set.py
├── prepare_cifake.py
├── prepare_wildfake.py
├── prepare_fusion_subset.py  50k fusion-ablation subset (not in the submitted model)
└── datasets/                 gitignored; machine path below
    ├── SID_Set_images/       or sid_set -> symlink here
    ├── CIFAKE/
    ├── cifake_full/          full CIFAKE mirror (preferred)
    ├── WildFake/
    └── WildFake_fusion_50k/  optional
```

On this machine the data root is:

```
/mnt/data1/xuting/tiktok_bytecop/data/datasets
```

In-repo relative path: `data/datasets/<name>`. Training configs should use the relative path.

---

## Loader

`AIGCDataset` (`dataset.py`) expects:

```
root/
  train|val/
    real/   *.jpg|png|webp|bmp
    fake/   *.jpg|png|webp|bmp   # or fake/<generator>/...
```

- Labels come from the **directory name**, not parquet.
- Default listing is recursive `rglob`. Million-file trees such as WildFake **must** use `--use_path_list`; otherwise startup spends minutes scanning disk and training throughput collapses.
- Corrupt images increment `n_failed` and do not abort the epoch.
- CLIP preprocess: Resize(224) -> CenterCrop(224) -> ToTensor -> OpenCLIP mean/std.
- Train augment runs before CLIP: `build_train_augment()`.

```python
from data.dataset import AIGCDataset
from data.transforms import build_train_augment

ds = AIGCDataset(
    "data/datasets/WildFake",
    split="train",
    augment=build_train_augment(),
    path_list="data/datasets/WildFake/train_paths.txt",
)
# ds[i] -> (3, 224, 224) float32, label int
```

### Official six degradations (`transforms.py`)

Values match the contest table. Each image draws **one** transform. Real and fake images share the same pipeline so the model cannot cheat on "was this processed?".

| Name | Allowed values |
|---|---|
| `jpeg_compression` | quality in {90, 70, 50, 30} |
| `gaussian_blur` | sigma in {0.5, 1.0, 2.0} |
| `resize` | scale in {0.5, 0.25} (downscale, then back to original size) |
| `gaussian_noise` | sigma in {0.02, 0.05, 0.10} |
| `color_jitter` | strength = 0.2 |
| `center_crop` | fraction = 0.8 |

Train: 30% clean + 70% one random transform. Eval: `evaluate.py --condition` sweeps clean and the 14 degradation cells.

Degradations are **CPU PIL/numpy**. They are negligible on the 44 KB serve-bench images. On SID/WildFake training they stack after JPEG decode, so worker count cannot be too small.

---

## SID-Set - stage 1

[HuggingFace · chuangchuangtan/SID-Set](https://huggingface.co/datasets/chuangchuangtan/SID-Set) (~161k images). Paper: [arXiv:2502.17107](https://arxiv.org/abs/2502.17107).

Three classes: `real` / `fake` / `tampered`. This task is binary, so **skip `tampered`**.

### On-disk layout

```
SID_Set_images/{train,val,test}/{real,fake,tampered}/
```

The HuggingFace helper defaults to `sid_set/`. If `SID_Set_images/` already exists, do not download a second copy; symlink:

```bash
cd data/datasets
ln -sfn SID_Set_images sid_set
```

### Download (once)

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login   # optional

python data/prepare_sid_set.py \
    --output_dir data/datasets/sid_set \
    --split all \
    --keep-parquet
```

| Flag | Effect |
|---|---|
| `--keep-parquet` | Keep `*.parquet`. **Do not** pass `--delete-parquet` when `sid_set` is a symlink into `SID_Set_images` |
| `--max-samples N` | Smoke subset |
| `--num-workers` | Parallel decode; default `min(8, CPU)` |

### Check

```bash
find data/datasets/SID_Set_images/train/real -type f | wc -l   # 59802
find data/datasets/SID_Set_images/train/fake -type f | wc -l   # 59788
find data/datasets/SID_Set_images/val/real   -type f | wc -l   # 11984
find data/datasets/SID_Set_images/val/fake   -type f | wc -l   # 11959
```

Stage-1 training: [`experiments/spatial_tower/configs/spatial_tower_sid.yaml`](../experiments/spatial_tower/configs/spatial_tower_sid.yaml). That checkpoint initializes stage 2; it is **not** the submitted file.

---

## WildFake - stage 2 (submitted weights)

[HuggingFace · Inf-imagine/WildFake](https://huggingface.co/datasets/Inf-imagine/WildFake). Paper: [arXiv:2402.11126](https://arxiv.org/abs/2402.11126).

Fake images sit under per-generator folders (`stable_diffusion_v1/`, `midjourney/`, ...). Real images sit under `real/`. Path-list stats **drop the empty `wukong/` tree**.

Classes are heavily imbalanced (train fake ~2.5M vs real ~0.73M). Training **must** enable the **type-balanced sampler** (`type_balanced_sampler.py`): each batch first samples generators uniformly, then images. Without it the model collapses onto frequent generators, official score looks inflated, and rare generators fail.

The sampler does not affect serving: it only reorders training indices. Serve benches should sample from **val**; they do not need generator balance.

### On this machine

```
data/datasets/WildFake/
    train/{real,fake}/ ...          # ~202 GB
    val/{real,fake}/   ...
    train_paths.txt                 # 3,238,953 lines
    val_paths.txt                   # 66,069 lines
    *_manifest.jsonl
    hf_download/                    # download cache; safe to delete
```

### Download (once)

There is no `wildfake.zip` under repo `data/`. Do not run `huggingface-cli download` against the **repo root** (it mixes git objects with the zip). Do not `git clone` the dataset repo.

```bash
mkdir -p /mnt/data1/xuting/tiktok_bytecop/data/datasets/WildFake
cd /mnt/data1/xuting/tiktok_bytecop/data/datasets/WildFake

huggingface-cli download Inf-imagine/WildFake \
    --repo-type dataset \
    --include "wildfake.zip" \
    --local-dir . \
    --local-dir-use-symlinks False
```

`wildfake.zip` is ~**118 GB**. Skip this if `train/` and `val/` are already complete.

### Unpack

```bash
cd /mnt/data1/xuting/tiktok_bytecop/data/datasets/WildFake
mkdir -p _unpack && cd _unpack
unzip -q ../wildfake.zip
```

The zip root may be `WildFake/train|val`, `home/WildFake/...`, or bare `train|val`. After unzip you should see:

```
find . -type d \( -name train -o -name val \) | head
```

Then align to the `AIGCDataset` root (`train/real` and `train/fake` next to the zip):

```bash
cd /mnt/data1/xuting/tiktok_bytecop/data/datasets/WildFake
# adjust the source path to whatever unzip produced
mv _unpack/WildFake/train _unpack/WildFake/val .
# or: mv _unpack/home/WildFake/train _unpack/home/WildFake/val .
```

After `train/` and `val/` look complete, `hf_download/`, `_unpack/`, and `wildfake.zip` can be deleted (~100+ GB back).

### Build path lists (required)

```bash
python data/prepare_wildfake.py --root data/datasets/WildFake
```

Writes `train_paths.txt` / `val_paths.txt` (`split/label/relpath`) and manifests. Then:

```bash
python data/prepare_wildfake.py --root data/datasets/WildFake --verify
```

Training configs must set:

```yaml
data:
  use_path_list: true
```

Stage 2: [`experiments/spatial_tower/configs/spatial_tower_wildfake.yaml`](../experiments/spatial_tower/configs/spatial_tower_wildfake.yaml). The run writes the submitted `best.pt`.

### Three failure modes

1. **Paths.** Zip nesting is not stable. The contract is "can we read `train/real` and `train/fake`?", not the HuggingFace preview layout.
2. **Path lists.** Without `*_paths.txt` the DataLoader `rglob`s ~3M files. The first epoch can sit silent for minutes with an idle GPU. That is not a slow model; the data layer deleted the throughput.
3. **Disk.** Peak unpack usage is far above 118 GB. Unpack on `/mnt/data1`, not the system disk.

---

## CIFAKE - smoke

[HuggingFace · bird-of-paradise/CIFAKE-10](https://huggingface.co/datasets/bird-of-paradise/CIFAKE-10) (CIFAR-10 reals + Stable Diffusion 1.4 fakes, 32x32).

This is **not** evidence that the submitted model generalizes: resolution and generator mix are a different regime from SID/WildFake. Use it only to check that the DataLoader, the six degradations, and one training step run.

The full mirror is `data/datasets/cifake_full/` (train 100k / val 10k). If you only have a partial `CIFAKE/`, rebuild:

```bash
python data/prepare_cifake.py --output_dir data/datasets/cifake_full
```

Config: [`experiments/spatial_tower/configs/spatial_tower_cifake.yaml`](../experiments/spatial_tower/configs/spatial_tower_cifake.yaml).

---

## Fusion ablation subset (not in the submitted model)

When dual-tower / RGPA ablations need a smaller slice than full WildFake:

```bash
python data/prepare_fusion_subset.py \
    --src data/datasets/WildFake \
    --dst data/datasets/WildFake_fusion_50k \
    --n-train 50000 \
    --n-val 5000
```

Defaults: 50k train, 5k val, 50/50 real/fake, fake side stratified by generator. Writes path lists (or `--copy` files). Submitted inference **does not** load this subset.

---

## Sanity checks

```bash
# SID
find data/datasets/SID_Set_images/train/real -type f | wc -l
find data/datasets/SID_Set_images/train/fake -type f | wc -l

# WildFake path-list lines = table counts above
wc -l data/datasets/WildFake/train_paths.txt
wc -l data/datasets/WildFake/val_paths.txt

# CIFAKE
find data/datasets/cifake_full/train/real -type f | wc -l
find data/datasets/cifake_full/val/real   -type f | wc -l
```

Loader smoke:

```python
from data.dataset import AIGCDataset
from data.transforms import build_train_augment

for name, split, plist in [
    ("SID_Set_images", "train", None),
    ("WildFake", "val", "data/datasets/WildFake/val_paths.txt"),
    ("cifake_full", "train", None),
]:
    ds = AIGCDataset(
        f"data/datasets/{name}",
        split=split,
        augment=build_train_augment(),
        path_list=plist,
    )
    x, y = ds[0]
    assert x.shape == (3, 224, 224)
```

---

## Related code

| File | Role |
|---|---|
| `data/dataset.py` | `AIGCDataset`, CLIP 224 preprocess |
| `data/transforms.py` | Official 6 degradations; train 30% clean / 70% one transform |
| `data/type_balanced_sampler.py` | WildFake balance by generator |
| `data/mbe.py` | Multi-class balanced error (analysis, not an official metric) |
| `data/prepare_sid_set.py` | SID parquet -> JPEG tree |
| `data/prepare_cifake.py` | CIFAKE -> `real/` `fake/` |
| `data/prepare_wildfake.py` | WildFake path lists + `--verify` |
| `data/prepare_fusion_subset.py` | 50k ablation subset |
| `throughput/README.md` | Spatial serve throughput and cost on A40 |
| `train.md` | Two-stage training commands |
| `evaluate.py` | 15-condition eval (clean + 14 cells) |
