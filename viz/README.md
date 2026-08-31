# TraceLens 前端

本机启动（仓库根目录，见根目录 [`README.md`](../README.md) Run inference）：

```bash
source ~/techjam/venv/bin/activate
cd /home/xuting/tiktok_bytecop
streamlit run viz/app.py --server.port 8508
```

打开 http://localhost:8508 。没有 GPU 也可以跑；Detect 在无 CUDA 时会跳过打分。

`runOnSave = true`，改完存盘浏览器自己重跑，不必重启进程。

## 版面

```
┌ Adjustments ┬ TraceLens ────────────────────────────────┐
│  Crop       │  IMG_4021.jpeg · 4000 × 3000 · 7.3 MB    │
│  Resample   │  ┌ Original ─┬ Adjusted ─┬ Details ────┐ │
│  Blur       │  │           │           │ Format  JPEG│ │
│  Noise      │  │   原图    │  调整后    │ Size   7.3MB│ │
│  Color      │  │  固定不动 │           │ …           │ │
│  JPEG       │  └───────────┴───────────┴─────────────┘ │
│  Advanced ▸ │                                          │
└─────────────┴──────────────────────────────────────────┘
```

左侧栏是调整面板，右上角 `Reset` 在没有调整时是禁用的。主区三栏：原图 / 调整后 /
详情卡片。没施加调整时，中间是**和图片同尺寸**的空位，点一下是「填进去」而不是
「撑开」，所以左边原图一个像素都不动。

`Advanced` 折叠里放随机种子和施加顺序说明。

## 视觉

系统字体（SF Pro → system-ui），页面 `#f5f5f7`，内容装白色圆角卡片，单一强调色
`#0071e3`，分隔线用 7% 黑的发丝线。分段控件按平台画法做：灰色轨道 + 一枚白色浮起的
选中段。图片圆角 10px 配两层阴影。

去掉的东西：衬线字、纸黄底、暗红、`Table 1.` / `Figure 1.` / `(a)(b)` / `Specimen`
这类论文腔，以及每个控件下面那段解释文字（改挂 `help=` 悬浮提示）。
属性从 11 项砍到 8 项，SHA-256 降级成卡片脚注。

## 图片尺寸

`FIGURE_BOX = (340, 270)`，`_display_width()` 按宽高比算出恰好装进去的宽度，
直接传给 `st.image(width=…)`。

**不能走 CSS**：Streamlit 把每个元素各自包进独立容器，`st.markdown` 吐出的裸
`<div>` 会被自动闭合，最后和图片是兄弟节点而不是父节点，写在 `.wrapper img` 上的
规则一条都命中不了。

## 六类调整

只给**官方允许值**（`data/transforms.py` 的 `_ALLOWED` 会校验，越界直接抛），
最左 `Off`，向右递增。施加顺序模拟转发链路：

```
crop → resample → blur → noise → color → JPEG
```

官方评测的 15 个条件每次只开一档；叠加是本页的扩展。

`Blur` 的半径是绝对像素、`Crop`/`Resample` 是比例——所以调整必须在**原始分辨率**上
做。`_preview_copy()` 缩的那份只送浏览器显示，不参与任何计算。

## 高斯噪声为什么另写了一份

`data/transforms.py` 的噪声是逐字节的 Python 循环，一张 12 MP 图 **24.4 秒**，
点一下等不起。`_noise_fast()` 用 numpy 重写，同样 `σ×255`、同样四舍五入后截断到
`[0,255]`，实测 σ=0.05 时噪声 std 12.767 vs 上游 12.749（理论 12.75），**0.59 秒**。
噪声逐通道 i.i.d.，换随机流统计上无差别，何况 `evaluate.py:179` 调 `apply_transform`
时本来就不传 seed、本来就不可复现。其余五类仍直接调上游实现。

> 顺带：`data/dataset.py:200` 在**全分辨率**图上做增强，训练时约 12% 的样本会走到
> 这个噪声分支，dataloader 很可能一直被它拖着。改上游会变增强的随机流、影响已有
> 实验的可复现性，所以这里没动。

## 接检测器

`_pictures()` 里已经有原分辨率的调整结果，接上即可：

```python
from serve.spatial_backend import SpatialDetector

@st.cache_resource
def _detector():
    return SpatialDetector()          # 建一次，反复用

record = _detector().score_pil(adjusted, path=photo.name)
# {"image_path", "pred", "logit", "label"}
```

传**原分辨率**的图，别传预览图。模型自己做 `Resize(224)` + `CenterCrop(224)`，
提前缩放会改变它看到的重采样痕迹。加载模型后才需要 `CUDA_VISIBLE_DEVICES=1`
（别用 0，vLLM 占着 39 G）。
