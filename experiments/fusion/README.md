# Fusion：两塔冻结 + GatedFusion

CLIP-H / RGPA **不再更新**。只训 `models/gated_fusion.py` 里那个 2→8→2 MLP：

```
z_s, z_f  = 标准化后的两塔 logit
w         = softmax(MLP([z_s, z_f]))     # 每张图一对权重
fused     = w_s * z_s + w_f * z_f
prob      = sigmoid(fused / T)
```

流程：

1. 从 WildFake **train** 抽 5 万张（50/50 real/fake，fake 按 Architecture 轮转）做训练。
2. 从官方 **`data/datasets/WildFake_images/val`**（约 6.6 万）再类型均衡抽 5 千做 val，不是从 train 里切 holdout。
3. 冻结两塔，把这批图的 logit **抽一次**（train 带增强；val 各跑 clean + robust）。
4. 在缓存的 tensor 上训 MLP 40 epoch。

子集写在 `data/datasets/WildFake_fusion_50k/`（只有 manifest，指向原图）。Logit 缓存在 `runs/fusion/fusion_wildfake/logits_cache/`，齐了会跳过抽取。

## 跑

```bash
source ~/techjam/venv/bin/activate
# 可先单独抽子集；train.py 发现没有也会自己抽
python data/prepare_fusion_subset.py \
  --src data/datasets/WildFake_images \
  --dest data/datasets/WildFake_fusion_50k \
  --n 50000 --n-val 5000

FUSION_DATA=data/datasets/WildFake_images \
bash experiments/fusion/run.sh fusion_wildfake --gpus 1,2,3,4
```

输出目录 `runs/fusion/fusion_wildfake/` 会放齐三份权重：`clip_h.pt`、`rgpa.pt`、`fusion.pt`。只评官方 val 切片、不重训门控：

```bash
python experiments/fusion/eval.py \
  --ckpt runs/fusion/fusion_wildfake/fusion.pt \
  --val  data/datasets/WildFake_fusion_50k/val
```

val 来自 `WildFake_images/val` 抽的 5 千张；logit 已缓存则直接打分。缺缓存时加 `--extract`（只跑冻结两塔，不训 MLP）。


```bash
tensorboard --logdir_spec \
gated_fusion:runs/fusion/fusion_wildfake/tensorboard,\
spatial_wildfake:runs/spatial_tower/spatial_tower_wildfake/tensorboard,\
forensic_wildfake:runs/forensic_tower/forensic_tower_wildfake/tensorboard \
  --port 6008 --bind_all
```

| 配置 | 值 |
| --- | --- |
| Spatial ckpt | `runs/fusion/fusion_wildfake/clip_h.pt`（源自 spatial WildFake `best.pt` step 12000） |
| Forensic ckpt | `runs/fusion/fusion_wildfake/rgpa.pt`（源自 forensic WildFake `best.pt` step 25000） |
| 子集 train | 约 4.5–5 万，来自 WildFake **train** |
| val | 5 千，来自官方 `WildFake_images/val`（类型均衡），clean + robust |
| 缓存 | `runs/fusion/fusion_wildfake/logits_cache/` |

`entropy_coef: 0`。CLIP val 远强于 RGPA，门控可能几乎全压在 CLIP 上；想强迫两边都用再把 yaml 里这项调成 `0.01` 一类。
