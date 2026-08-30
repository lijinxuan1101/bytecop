# Spatial tower

OpenCLIP ViT-H/14，解冻最后 2 个 block + `ln_post` / `proj` + 分类头。输入 CLIP normalize。

配置：`configs/spatial_tower.yaml`  
权重输出：`runs/spatial_tower/spatial_tower/`

## 当前这次续跑（2026-08-30）

8/29 训完 epoch 1 后，e2 末尾 NCCL 超时崩了。只有 `best.pt`（weights-only），没有 `last.pt`。

| 项 | 值 |
|---|---|
| 已完成 | epoch 1：`train=0.0759` `val_loss=0.0026` `val_auc=0.99999`（当时只有 clean val） |
| 本次 | epoch **2 → 8**，从 `best.pt` 恢复权重 |
| GPU | 物理卡 2,3,4,5（`--gpus`） |
| 学习率 | Adam 重置；常数 head `3e-4` / backbone `1e-5`（cosine 对不上） |
| 数据 | `data/datasets/SID_Set_images`：train 119590 / val 23943 / cal 16467 |
| batch | 每卡 16 × 4 = global 64，1869 step / epoch |

```bash
cd ~/tiktok_bytecop
source ~/techjam/venv/bin/activate

bash experiments/spatial_tower/run.sh \
  --gpus 2,3,4,5 \
  --resume runs/spatial_tower/spatial_tower \
  --skip-eval \
  --nccl-timeout 1800
```

启动日志应有：`rank 0 reads CLIP / ckpt, then broadcast`，`resume : .../best.pt (weights-only)`，`epochs : 2 → 8`。其它三张卡会先打 `No pretrained weights loaded`（只建结构，等 rank 0 广播），这是正常的。

这次训完会写 `last.pt`。以后续跑同一条 `--resume` 会优先读 `last.pt`（model + optim + sched）。

## `run.sh` 参数

| 参数 | 作用 |
|---|---|
| `--gpus 2,3,4,5` | 设置 `CUDA_VISIBLE_DEVICES`，卡数 = `nproc_per_node` |
| `--resume <dir\|file>` | 目录优先 `last.pt`，否则 `best.pt` |
| `--extra-epochs N` | 从 resume 点再训 N 个 epoch |
| `--skip-eval` | 训完不跑 15 条件 `evaluate.py` |
| `--skip-train` | 只评不训 |
| `--nccl-timeout 1800` | NCCL 心跳秒数；val 只在 rank 0，其它卡要等 |

不写 `--gpus` 时用 yaml 的 `num_gpus`（默认 4），对应机器上的 `cuda:0..3`。  
这台 8×A40 默认 `NCCL_P2P_DISABLE=1`。

## 多卡怎么跑

`run.sh` → `torchrun --nproc_per_node=N` → 每进程一张卡。

1. 所有 rank 建同样结构；**只有 rank 0** 读 `weights/clip_h/open_clip_pytorch_model.bin` 和 resume 文件
2. `broadcast_module` 把权重拷到其它卡，resume 的 epoch / history / optim 用 `broadcast_object`
3. 包 DDP 开训。Train 用 `DistributedSampler`，global batch = `16 × 卡数`

Val **不并行**：每个 epoch 结束 rank 0 连跑两遍（clean + 随机一种官方退化），其它卡 `barrier` 等，所以要加长 `--nccl-timeout`。

## Train / val

- **Train**：30% 原图，70% 随机一种官方退化（JPEG / blur / resize / noise / jitter / crop）
- **Val**：同一份 val 扫两遍——clean 原图、robust 每张随机一种退化。不是官方 15 格 Final Score
- 15 格评测：训完去掉 `--skip-eval`，或手动跑 `evaluate.py`

## 输出

| 文件 | 说明 |
|---|---|
| `best.pt` | `val_auc` 最好的整塔权重 |
| `last.pt` | 每个 epoch 末的完整断点 |
| `best_metrics.json` / `history.json` | 指标 |
| `tensorboard/` | `train/step_loss` 按 step；`val/loss_clean`、`val/loss_robust` 按 epoch |

```bash
tensorboard --logdir runs/ --port 6008 --bind_all
```
