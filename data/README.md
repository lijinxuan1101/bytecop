# data/ 目录说明

训练脚本只认「图片目录树」，不读 parquet / zip。原始数据通过软链挂进来，抽出结果单独放，默认训练路径再软链过去。

```
原始下载（软链 → ~/techjam/raw/）
  data/SID_Set          HuggingFace parquet（约 131G）
  data/CIFAKE           CIFAKE 原图
  data/WildFake         ModelScope 包（含官方划分脚本）

抽出后的训练目录
  data/SID_Set_processed/     SID 正式集（约 36G JPEG）
  data/cifake_full/           CIFAKE 全量抽出
  data/cifake_smoke/          CIFAKE 冒烟子集
  ~/techjam/raw/WildFake_extracted   WildFake 解压图（约 1.2T，仓库内无软链）

训练默认入口（别名，不是拷贝）
  data/datasets/SID_Set_images     →  ../SID_Set_processed
  data/datasets/WildFake_images    prepare_wildfake.py 的输出
```

代码（会进 git）：`dataset.py`、`transforms.py`、`prepare_sid_set.py`、`prepare_cifake.py`、`prepare_wildfake.py`、`type_balanced_sampler.py`。上面的数据目录都在 `.gitignore` 里。

`AIGCDataset` 要求（SID / WildFake / CIFAKE 都一样）：

```
<root>/train/{real,fake}/
<root>/val/{real,fake}/
<root>/calibration/{real,fake}/   # 可选
```

`real=0`，`fake=1`。指到 parquet / zip 根目录会找不到图。

---

## SID_Set

正式训练集。链接：[https://huggingface.co/datasets/saberzl/SID_Set](https://huggingface.co/datasets/saberzl/SID_Set)

- 论文数据：SIDA（社交媒体图像伪造检测 / 定位）
- 内容：真实照片（OpenImages V7）、整图 AI 生成（full synthetic）、局部篡改（tampered，带 mask）
- 规模：官方写 30 万。HuggingFace 公开 **train 21 万 + val 3 万 = 24 万**（三类约各 8 万）；test 6 万不公开，防泄漏
- 体积：下载约 131G parquet
- 本仓库：只用 real + full synthetic，丢掉 tampered；抽到 `SID_Set_processed`

### 三条路径

| 路径 | 类型 | 作用 |
| --- | --- | --- |
| `SID_Set` | 软链 → `~/techjam/raw/SID_Set` | 原始 283 个 parquet。官方公开集 24 万张：real / full synthetic / tampered 各约 8 万。**不能直接拿去训。** |
| `SID_Set_processed` | 真实目录 | `prepare_sid_set.py` 抽出的 JPEG。只有 real + full synthetic，丢掉 tampered。 |
| `datasets/SID_Set_images` | 软链 → `SID_Set_processed` | `train.py` / `run.sh` 的默认 `--data`。和上一份是同一盘文件，没有复制。 |

```
SID_Set/data/*.parquet
        │
        ▼  python data/prepare_sid_set.py --src data/SID_Set --dest data/SID_Set_processed
SID_Set_processed/
    train/{real,fake}/
    val/{real,fake}/
    calibration/{real,fake}/
        ▲
datasets/SID_Set_images ──┘
```

当前抽出份数（见 `SID_Set_processed/manifest.json`）：

| split | real | fake | skipped（主要是 tampered） |
| --- | --- | --- | --- |
| train | 59,802 | 59,788 | 60,062 |
| val | 11,984 | 11,959 | 11,703 |
| calibration | 8,214 | 8,253 | 8,235 |

- **train / val**：训练循环使用。
- **calibration**：全部 epoch 结束后做 temperature scaling，不参与选模。
- **tampered（label=2）**：真图局部篡改，不是整张 AI 生成；本仓库是二分类「真 / 整图 AI」，抽图时跳过。

不要加 `--delete-parquet`，会顺着软链删掉 `~/techjam/raw/SID_Set`。

### 训练怎么接

```bash
--data data/datasets/SID_Set_images
```

默认随机 `DistributedSampler`（多卡切开 + shuffle），不是 WildFake 那套类型轮转。Val 只在每个 epoch 末跑一遍；中间 ckpt 不按比例切（详见下面 WildFake 的 Val / Ckpt 表）。

---

## CIFAKE

流水线 / 冒烟，**非正式集**。链接：[https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

- 内容：CIFAR-10 真实图 + Stable Diffusion 1.4 生成图，**32×32**
- 规模：real / fake 各 6 万，共 12 万
- 限制：分辨率太低，放大到 224 也补不回高频，**不能用来判断取证分支是否有效**
- 本仓库：`prepare_cifake.py` 抽到 `cifake_full` / `cifake_smoke`

---

## WildFake

SID 训完后的全类型微调数据。链接：[https://modelscope.cn/datasets/hy2628982280/WildFake/summary](https://modelscope.cn/datasets/hy2628982280/WildFake/summary)

- 论文：Hong et al., AAAI 2025
- 内容：社区收集的 AI 图（Civitai / Midjourney 等），按生成器族分层：GAN / Diffusion / Other，以及 architecture、weight、时间版本
- 规模：官方 metadata 约 train 295 万 + test 74 万（以 CSV 为准）。上游 39 个 zip，解压后约 1.2T
- 本仓库：用全量能落地的图做微调，**覆盖全部生图类型**（不是官方四套 leave-one-out）。`split_train_test/*.py` 是作者原脚本，训练不调用

### 下载：不要直接用 snapshot_download

modelscope 1.39.1 拿不全这份数据。`hub/file_download.py` 用 `Range` 头续传，但完整性校验拿的是**最后一次 ranged 响应**的 `Content-Length`——续传后那只是剩余字节数，不是文件总长：

```python
total = int(content_length)      # 续传后 = 剩余字节数
if total != downloaded_length:   # 因此必然不等
    os.remove(temp_file.name)    # 已下的几十 GB 被删光
    raise FileDownloadError(...)
```

而 `API_FILE_DOWNLOAD_TIMEOUT` 硬编码 60 秒（无环境变量可调），`Retry(total=5)` 又是整个文件**累计**5 次、中途成功不重置。50 GB 的文件在 ~20MB/s 链路上要跑 45 分钟，撞满 5 次 60 秒停顿几乎必然，于是一定会触发续传，于是一定失败。

实测（2026-08-29）：30 个小 zip 全部完好，**9 个 50 GB 级的大 zip 全军覆没**，共 421 GB。用 `scripts/refetch_wildfake.py` 补：

```bash
python scripts/refetch_wildfake.py --root ~/techjam/raw/WildFake --dry-run   # 先看缺哪些
python scripts/refetch_wildfake.py --root ~/techjam/raw/WildFake            # 再补
```

它自己管 `.part` 断点、无限重试、按 `Content-Range` 里的总长校验，跑完逐字节对得上上游。

### 解压

`part_N.zip` 内是裸文件（无顶层目录），所以必须一 zip 一目录，否则 part_1..7 会全部糊在一起：

```bash
cd ~/techjam/raw/WildFake
find Images -type f -name "*.zip" -print0 |
while IFS= read -r -d '' z; do
    rel="${z#Images/}"; name="${rel%.zip}"
    dest="$HOME/techjam/raw/WildFake_extracted/$name"
    mkdir -p "$dest"
    unzip -q -n "$z" -d "$dest" || echo "FAILED: $z"
done
```

`-n` 不覆盖已存在文件，所以补下新 zip 后原样重跑即可，已解压的会被跳过。

### 原始包（`data/WildFake` → `~/techjam/raw/WildFake`）

训练不读这里的 zip。标签和官方划分在 CSV 里：

```
WildFake/
├── Images/                         # 39 个 zip，约 1.2T
│   ├── GAN_based.zip
│   ├── Other_based.zip
│   ├── Diffusion_based/
│   │   ├── ADM.zip  DALLE.zip  DDIM.zip  DDPM.zip  Imagen.zip  VQDM.zip
│   │   ├── Midjourney/{Advanced,Typical}/part_*.zip
│   │   └── SD/
│   │       ├── personalizedSD.zip  SDwithAdaptor.zip
│   │       └── originalSD/{Advanced,Typical}/part_*.zip
│   └── Real/{afhq,celebahq,church,coco,ffhq,imagenet,laion5b,wukong}.zip
├── label_csv_files/                # 每源一份，列见下
└── split_train_test/csv_file/      # 官方 train/test
    ├── total_split/{train,test}_metadata.csv    # 2,856,568 / 714,156
    ├── cross_generators/
    ├── cross_architectures/
    ├── cross_weights/
    └── cross_times/
```

CSV 统一 8 列：`Generator, Architecture, Weight, Category, IsAdvanced, IsFake, Image_path, Num`。`IsFake=1` 生成图，`0` 真实图。`Image_path` 形如 `./GAN_based/Typical/styleGAN/.../img000000.jpg`，指向的是作者机器上的解压树，**不是** `Images/` 里的 zip，也和本机 `WildFake_extracted` 深度不完全一致。

### 解压后图片（`~/techjam/raw/WildFake_extracted`）

一 zip 一目录。zip 自带同名顶层目录的会再套一层；`part_N.zip` 是裸文件，多一个 `part_k/`。写 loader 用 `**`，别假设固定深度。

```
WildFake_extracted/
├── Diffusion_based/
│   ├── ADM/ADM/imgs/
│   ├── DALLE/DALLE/{Typical,Advanced}/
│   ├── DDIM/DDIM/{imgs_bedroom,imgs_CC9K}/
│   ├── DDPM/DDPM/{imgs_bedroom,imgs_CC9K,imgs_church}/
│   ├── Imagen/Imagen/{backpack_Chanel, bag_LV, ...}/
│   ├── VQDM/VQDM/img/
│   ├── Midjourney/
│   │   ├── Advanced/part_1 … part_7/     # 图直接在 part 下，没有 mj_v5/
│   │   └── Typical/part_1 … part_4/
│   └── SD/
│       ├── originalSD/
│       │   ├── Advanced/part_1 … part_7/{SDv2-..., cloth, hat, ...}/
│       │   └── Typical/part_1 … part_3/{SDv15-dpmsolver-25-15K, ...}/
│       ├── personalizedSD/personalizedSD/{dreambooth,finetune}/
│       └── SDwithAdaptor/SDwithAdaptor/{controlnet,lora,lycris}/
├── GAN_based/GAN_based/
│   ├── Typical/{BigGAN, starGAN, styleGAN}/
│   └── Advanced/{DF-GAN, GALIP, GigaGAN}/
├── Other_based/Other_based/
│   ├── Typical/{VQGAN, VQVAE}/
│   └── Advanced/{MAE, MAGE}/
└── Real/
    ├── afhq/afhq/afhq_v2/{train,test}/{cat,dog,wild}/     # 31,933
    ├── celebahq/celebahq/data1024x1024/                   # 30,000
    ├── church/church/church/train/                        # 83,352
    ├── coco/coco/coco2017/{train,val,test}2017/           # 163,846
    ├── ffhq/ffhq/images/                                  # 70,000
    ├── imagenet/imagenet/val/n*/                          # 96,788
    ├── laion5b/laion5b/imgs/                              # 271,831
    └── wukong/wukong/                                     # 0 张
```

CSV `Image_path` 对不上解压树，读图时要补层或 glob：

| CSV `Image_path` | 实际文件 |
| --- | --- |
| `./GAN_based/Typical/styleGAN/...` | `GAN_based/GAN_based/Typical/styleGAN/...` |
| `./Real/coco/coco2017/test2017/img000000.jpg` | `Real/coco/coco/coco2017/test2017/img000000.jpg` |
| `./Diffusion_based/Midjourney/Advanced/mj_v5/<hash>.png` | `…/Advanced/part_3/<hash>.png`（没有 `mj_v5/`） |
| `./Diffusion_based/SD/originalSD/Typical/SDv15-…/<hash>.png` | `…/Typical/part_3/SDv15-…/<hash>.png` |

### 三个会踩的坑

1. **`Real/wukong.zip` 在上游就是 164 字节的空档**（只含一个空目录），但 `label_csv_files/real_wukong.csv` 有 19.8 MB 标签。建图像清单时必须排除 wukong，否则会得到一堆指向不存在文件的条目。
2. **两类 zip 的内部结构不同，导致解压后深度不一致**。`GAN_based.zip` / `coco.zip` / `ADM.zip` / `personalizedSD.zip` 这类**自带同名顶层目录**，配上「一 zip 一目录」就成了双层嵌套（见上树）。`part_N.zip` 是裸文件，多一层 `part_k/`。
3. **各目录 part 数量不同**，上游编号本身是连续的，但总数要按目录查，不能硬编码：`Midjourney/Advanced` 1..7、`Midjourney/Typical` 1..4、`SD/originalSD/Advanced` 1..7、`SD/originalSD/Typical` 1..3。

### 建成训练入口

官方 train/test 和四套 cross-* **不用**。按 CSV 的 Category 切：`n ≥ 50` 为 98% train / 2% val，更小的类全进 train。丢掉 wukong 和解析失败的路径。

```bash
python data/prepare_wildfake.py --dry-run          # 只看解析命中率
python data/prepare_wildfake.py                   # 写软链 + manifest
python data/prepare_wildfake.py --no-links        # 只写 manifest，训练照样能跑
```

默认：

- `--extracted ~/techjam/raw/WildFake_extracted`
- `--csv-dir data/WildFake/label_csv_files`
- `--dest data/datasets/WildFake_images`

```
WildFake_images/
  train/{real,fake}/<Architecture>/<Category>/...
  val/{real,fake}/...
  train/manifest.csv
  val/manifest.csv
  manifest.json          # batch_sampler: type_balanced
```

`AIGCDataset` 优先读各 split 下的 `manifest.csv`（绝对路径指向 extracted），不必 rglob 两三百万软链。根目录 `manifest.json` 带 `batch_sampler: type_balanced` 时，训练自动走下面这套抽样；**这份 json 写完之前不要开训**。

### 切分策略

- 官方 `train_metadata.csv` / `test_metadata.csv` 和四套 cross-* **不用**。
- 类型 = CSV **Architecture**（18 种 fake；real 去掉 wukong 后 7 个源）。每个 **Category** 都进 train，避免某种设定整锅划去 val。
- Category 内：`n ≥ 50` → **98% train / 2% val**（至少 1 张 val）；`n < 50` → **全进 train**。
- 丢掉 wukong 和解析失败的路径。无 calibration。固定 `--seed`（默认 42）。
- 干跑命中（2026-08-30）：盘上 3,305,037 张；CSV 丢掉 wukong 265,700 + 未解析 2；train 3,238,953（real 732,794 / fake 2,506,159），val 66,069。

### 抽样与训练策略

标签仍是二分类 `real=0 / fake=1`，模型看不到「这是 SD 还是 MJ」。覆盖类型靠 **train 里每种 Architecture 都有图** + **按类型轮转采样**。实现：`data/type_balanced_sampler.py`。

**Train（`TypeBalancedBatchSampler`）**

| 项 | 规则 |
| --- | --- |
| 何时启用 | `--data` 指向本目录，且 `manifest.json` 里 `batch_sampler: type_balanced` |
| Batch | **一半 real、一半 fake**（batch_size 必须偶数，如 64 → 32+32） |
| Fake 池 | **每个 Architecture 一个下标列表**（不是每个叶子文件夹） |
| Fake 怎么走 | epoch 开始对每个池 `shuffle`，用指针从头走到尾；按架构名顺序各取 1 张再转圈；抽干的类型退出轮转 |
| 一轮 | **以 fake 扫完为准**，每张 fake 一轮只用一次 |
| Real | 另建一池，同样 shuffle；走完再 shuffle **循环**（约 73 万张真图一轮会重复 ~3.4 次） |
| 不重复怎么记 | **不写盘、不建 set**。没抽过 = 指针后面还有；抽过 = 指针已跨过。O(1)，不影响解码/GPU |
| 下一 epoch | `set_epoch` 换种子再 shuffle，所有 fake 重新算没抽过 |
| DDP | 每种架构的 fake 按 `rank::world_size` 切开，各卡不相交 |

**Val**

普通读 `val/manifest.csv`，**不** 1:1、**不** 轮转。每次 val 扫两遍：clean + 随机一种官方变换（和 SID 相同）。

| 项 | WildFake | SID |
| --- | --- | --- |
| 何时跑 | 每 **1000** optimizer step，再加 epoch 末（最后一步不是 1000 倍数时） | 只在 epoch 末 |
| yaml | `val_every_steps: 1000` | 不写 / `0` |
| CLI | `--val-every N`（覆盖 yaml） | 同上，一般不用 |

WildFake 一轮约 2 万 step，会 val 约 20 次。val 6.6 万张、每次两遍，比 SID 慢一截，是故意的。

**Ckpt**

| 项 | WildFake | SID |
| --- | --- | --- |
| 中间点 | 每 **1/20** epoch → `ckpts/e1_05.pt` … `e1_95.pt`（19 个） | 默认不按比例切 |
| yaml | `ckpt_every_frac: 0.05` | 不写则只靠 `last.pt` |
| CLI | `--ckpt-every-frac 0.05` | 同上 |
| 期末 | `last.pt`（完整断点：model + optim + sched） | 每个 epoch 末写 |
| 最优 | `best.pt`（`val_auc` 最好；中途 val 也能刷新） | 只在 epoch 末比 |

Early stop 只数 **epoch 末** 的 val，中途 1000 步一次不会把 patience 耗光。WildFake yaml 里 `early_stop_patience: 0`（只 finetune 1 epoch）。

**怎么接到 SID 权重**

- `--data data/datasets/WildFake_images`
- 读 SID `best.pt` **只恢复权重**，Adam 重置。不要 `last.pt` 全量 resume（SID cosine 已到 1e-6）。
- 常数微调 lr：RGPA `5e-5`；Spatial head `3e-5` / backbone `1e-6`。
- 输出写到新 run 目录，不覆盖 SID。
- 增强仍是 30% clean / 70% 官方变换。
- 先 1 个 epoch，再用 SID `evaluate.py` 看 Final Score 有没有掉。

**记录什么**

TensorBoard / `history.json` 按 `global_step` 记 loss、AUC、lr，**不按架构记「这个 batch 抽了谁」**。类型均衡由采样器保证。不要每张图写 json，那会拖慢训练。

```bash
# 软链和 manifest.json 都齐了再跑。先激活 venv，否则找不到 torchrun。
# --resume 指向 SID 目录时会读 best.pt（不是 last.pt），
# 输出目录不同 → 新 history / TensorBoard / ckpts，epoch 从 1 起。
# 别用 GPU 0（这台机器上常被 vLLM 占着）。
source ~/techjam/venv/bin/activate

FORENSIC_DATA=data/datasets/WildFake_images \
bash experiments/forensic_tower/run.sh forensic_tower_wildfake \
  --gpus 6,7 \
  --resume runs/forensic_tower/forensic_tower \
  --extra-epochs 1 \
  --skip-eval \
  --nccl-timeout 1800
# → runs/forensic_tower/forensic_tower_wildfake/

SPATIAL_DATA=data/datasets/WildFake_images \
bash experiments/spatial_tower/run.sh spatial_tower_wildfake \
  --gpus 1,2,3,4 \
  --resume runs/spatial_tower/spatial_tower \
  --extra-epochs 1 \
  --skip-eval \
  --nccl-timeout 1800
# → runs/spatial_tower/spatial_tower_wildfake/
```

## Fusion 5 万子集

门控 MLP 不吃全量 WildFake。`data/prepare_fusion_subset.py` 从 WildFake **train** 抽约 5 万做训练，再从官方 **val**（6.6 万）类型均衡抽 5 千做验证，只写 manifest：

`data/datasets/WildFake_fusion_50k/{train,val}/manifest.csv`

抽 logit + 训门控见 `experiments/fusion/README.md`。

