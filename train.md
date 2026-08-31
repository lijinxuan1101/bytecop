# 两塔训练说明

OpenCLIP-H 空间塔和 RGPA 取证塔**分开训练**，数据混合必须一致。融合在 Stage 3，不在本页。

当前默认机器：**NVIDIA A40 48GB，FP32，不开 gradient checkpointing / BF16**。

方案见 `AI_image_detection_technical_proposal.md`。 RGPA 取证分支已写入该文档 §6。

---

## 1. 训练前

```bash
pip install -r requirements.txt
pip install -e models/open_clip
python scripts/download_clip.py          # Spatial 需要；RGPA 从零训
```

数据目录必须是：

```
data/datasets/SID_Set_images/
├── train/{real,fake}/
├── val/{real,fake}/
└── calibration/{real,fake}/   # 可选；有则训练结束做 temperature scaling
```

从 parquet 抽出图片：

```bash
python data/prepare_sid_set.py \
    --src  data/datasets/SID_Set \
    --dest data/datasets/SID_Set_images \
    --delete-parquet
```

`batch_size` 一律是**每卡**。有效 global batch = `batch_size × GPU数 × grad_accum_steps`。

两塔共用同一套退化：`augment: true`，`clean_prob: 0.3`（30% 干净，70% 随机一种官方变换）。不要只改其中一塔。

---

## 2. Spatial Tower（OpenCLIP ViT-H/14）

解冻最后 2 个 Transformer block + `ln_post` / `proj` + 分类头。输入走 **OpenCLIP normalize**。

| 项 | 值 |
|---|---|
| 配置 | `experiments/spatial_tower/configs/spatial_tower.yaml` |
| 卡数 | 4 × A40 |
| 每卡 batch | 16 |
| `grad_accum_steps` | 1 |
| global batch | 64 |
| 学习率 | head `3e-4` / backbone `1e-5` |
| 调度 | cosine，`warmup_ratio: 0.05` |
| epoch | 8 |
| 选模 | `val_auc` → `best.pt` |

```bash
# 推荐：读 yaml 里的 num_gpus，训完自动 eval
bash experiments/spatial_tower/run.sh

# 或直接 torchrun
torchrun --standalone --nproc_per_node=4 experiments/spatial_tower/train.py \
    --config experiments/spatial_tower/configs/spatial_tower.yaml \
    --data   data/datasets/SID_Set_images \
    --output runs/spatial_tower/spatial_tower
```

单卡：

```bash
python experiments/spatial_tower/train.py \
    --config experiments/spatial_tower/configs/spatial_tower.yaml
```

OOM 时把 yaml 改成 `batch_size: 8`、`grad_accum_steps: 2`，global batch 仍是 64。卡数不够就改 `num_gpus`，或启动时覆盖：

```bash
NUM_GPUS=2 bash experiments/spatial_tower/run.sh
```

---

## 3. Forensic Tower（RGPA）

从零训练。输入是 **像素 RGB `[0, 1]`，不做 OpenCLIP / ImageNet normalize**。`patch_size: 32`（224 → 7×7=49 patches）。SRM 高通核冻结。

| 项 | 值 |
|---|---|
| 配置 | `experiments/forensic_tower/configs/forensic_tower.yaml` |
| 卡数 | 2 × A40 |
| 每卡 batch | 64 |
| global batch | 128 |
| 学习率 | `5e-4`（整网，除 SRM） |
| 调度 | cosine，`warmup_ratio: 0.05` |
| epoch | 20 |
| 选模 | `val_auc` → `best.pt` |
| 早停 | `early_stop_patience: 5` |

```bash
bash experiments/forensic_tower/run.sh

torchrun --standalone --nproc_per_node=2 experiments/forensic_tower/train.py \
    --config experiments/forensic_tower/configs/forensic_tower.yaml \
    --data   data/datasets/SID_Set_images \
    --output runs/forensic_tower/forensic_tower
```

没有 `test/` 时，`run.sh` 会在 `val/` 上评。

两塔不要同时抢同一组 GPU。先训 Spatial（4 卡），再训 RGPA（2 卡）。

---

## 4. 输出

| 路径 | 内容 |
|---|---|
| `runs/spatial_tower/spatial_tower/` | 空间塔 |
| `runs/forensic_tower/forensic_tower/` | 取证塔 |

每个 run 目录：

| 文件 | 说明 |
|---|---|
| `config.yaml` | 本次训练用的配置副本 |
| `best.pt` | `val_auc` 最好的权重（无 `module.` 前缀，可直接给 `evaluate.py`） |
| `best_metrics.json` | 最佳 epoch 的 AUC / loss |
| `history.json` | 逐 epoch 曲线 |
| `calibrator.pkl` | calibration split 上的 temperature scaler |
| `calibration_metrics.json` | ECE / Brier |
| `tensorboard/` | 标量 |
| `val_predictions.json` | 仅 RGPA：val 上的 forensic logit，留给融合 |
| `aggregation_stats.json` | 仅 RGPA：高低残差权重统计 |

```bash
tensorboard --logdir runs/
```

看 `train/lr_head`（空间）或 `train/lr`（取证）：前 5% step 爬升，再 cosine 降到 0。`val/auc` 创新高时会打印 `*best*`。

---

## 5. 训练后评估

`run.sh` 在有 `best.pt` 时会自动跑。手动：

```bash
python evaluate.py \
    --backbone clip_h \
    --ckpt runs/spatial_tower/spatial_tower/best.pt \
    --data data/datasets/SID_Set_images/val \
    --calibrator runs/spatial_tower/spatial_tower/calibrator.pkl \
    --output runs/spatial_tower/spatial_tower/eval_results.json

python evaluate.py \
    --backbone rgpa \
    --ckpt runs/forensic_tower/forensic_tower/best.pt \
    --data data/datasets/SID_Set_images/val \
    --calibrator runs/forensic_tower/forensic_tower/calibrator.pkl \
    --output runs/forensic_tower/forensic_tower/eval_results.json
```

只训不评：`SKIP_EVAL=1 bash experiments/spatial_tower/run.sh`。

---

## 6. 配置对照（不要混用）

| | Spatial | RGPA |
|---|---|---|
| 输入 | CLIP mean/std | 像素 RGB，`normalize: none` |
| 分辨率 | 224 | 224，`patch_size: 32` |
| 退化 | `clean_prob: 0.3` | 必须相同 |
| 精度 | FP32 | FP32 |
| 选模 | `monitor: val_auc` | 同上 + 早停 5 epoch |

改超参只改对应 yaml，不要改 `train.py` 默认值。
