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

训练默认入口（别名，不是拷贝）
  data/datasets/SID_Set_images  →  ../SID_Set_processed
```

代码（会进 git）：`dataset.py`、`transforms.py`、`prepare_sid_set.py`、`prepare_cifake.py`。上面的数据目录都在 `.gitignore` 里。

---

## 三个数据集

### SID_Set（正式训练）

- 链接：[https://huggingface.co/datasets/saberzl/SID_Set](https://huggingface.co/datasets/saberzl/SID_Set)
- 论文数据：SIDA（社交媒体图像伪造检测 / 定位）
- 内容：真实照片（OpenImages V7）、整图 AI 生成（full synthetic）、局部篡改（tampered，带 mask）
- 规模：官方写 30 万。HuggingFace 公开 **train 21 万 + val 3 万 = 24 万**（三类约各 8 万）；test 6 万不公开，防泄漏
- 体积：下载约 131G parquet
- 本仓库：只用 real + full synthetic，丢掉 tampered；抽到 `SID_Set_processed`



### CIFAKE（流水线 / 冒烟，非正式集）

- 链接：[https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- 内容：CIFAR-10 真实图 + Stable Diffusion 1.4 生成图，**32×32**
- 规模：real / fake 各 6 万，共 12 万
- 限制：分辨率太低，放大到 224 也补不回高频，**不能用来判断取证分支是否有效**
- 本仓库：`prepare_cifake.py` 抽到 `cifake_full` / `cifake_smoke`



### WildFake（跨生成器泛化测试）

- 链接：[https://modelscope.cn/datasets/hy2628982280/WildFake/summary](https://modelscope.cn/datasets/hy2628982280/WildFake/summary)
- 论文：Hong et al., AAAI 2025
- 内容：社区收集的 AI 图（Civitai / Midjourney 等），按生成器族分层：GAN / Diffusion / Other，以及 architecture、weight、时间版本
- 规模：官方 metadata 约 train 295 万 + test 74 万（以 CSV 为准）。上游 39 个 zip，解压后约 1.2T
- 本仓库：SID 训完后再做未见生成器测试。`split_train_test/*.py` 是作者原脚本，路径仍指向他们机器，训练不会调用

#### 下载：不要直接用 snapshot_download

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

#### 解压

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

#### 三个会踩的坑

1. **`Real/wukong.zip` 在上游就是 164 字节的空档**（只含一个空目录），但 `label_csv_files/real_wukong.csv` 有 19.8 MB 标签。建图像清单时必须排除 wukong，否则会得到一堆指向不存在文件的条目。
2. **两类 zip 的内部结构不同，导致解压后深度不一致**。`GAN_based.zip` / `coco.zip` / `ADM.zip` / `personalizedSD.zip` 这类**自带同名顶层目录**，配上"一 zip 一目录"的解压约定就成了双层嵌套：`GAN_based/GAN_based/`、`Real/coco/coco/`、`SD/personalizedSD/personalizedSD/`。而 `part_N.zip` 是裸文件，只有一层。写 glob 要用 `**` 递归，别假设固定深度。
3. **各目录 part 数量不同**，上游编号本身是连续的，但总数要按目录查，不能硬编码：`Midjourney/Advanced` 1..7、`Midjourney/Typical` 1..4、`SD/originalSD/Advanced` 1..7、`SD/originalSD/Typical` 1..3。

---



## SID 三条路径


| 路径                        | 类型                           | 作用                                                                                |
| ------------------------- | ---------------------------- | --------------------------------------------------------------------------------- |
| `SID_Set`                 | 软链 → `~/techjam/raw/SID_Set` | 原始 283 个 parquet。官方公开集 24 万张：real / full synthetic / tampered 各约 8 万。**不能直接拿去训。** |
| `SID_Set_processed`       | 真实目录                         | `prepare_sid_set.py` 抽出的 JPEG。只有 real + full synthetic，丢掉 tampered。               |
| `datasets/SID_Set_images` | 软链 → `SID_Set_processed`     | `train.py` / `run.sh` 的默认 `--data`。和上一份是同一盘文件，没有复制。                               |


关系：

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


| split       | real   | fake   | skipped（主要是 tampered） |
| ----------- | ------ | ------ | --------------------- |
| train       | 59,802 | 59,788 | 60,062                |
| val         | 11,984 | 11,959 | 11,703                |
| calibration | 8,214  | 8,253  | 8,235                 |


- **train / val**：训练循环使用。
- **calibration**：全部 epoch 结束后做 temperature scaling，不参与选模。
- **tampered（label=2）**：真图局部篡改，不是整张 AI 生成；本仓库是二分类「真 / 整图 AI」，抽图时跳过。

不要加 `--delete-parquet`，会顺着软链删掉 `~/techjam/raw/SID_Set`。

---



## 训练怎么接

```bash
# 空间塔 / 取证塔默认都读这条
--data data/datasets/SID_Set_images
```

`AIGCDataset` 要求：

```
<root>/train/{real,fake}/
<root>/val/{real,fake}/
<root>/calibration/{real,fake}/   # 可选
```

`real=0`，`fake=1`。指到 `data/SID_Set`（parquet）会找不到图。