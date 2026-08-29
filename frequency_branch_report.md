# 频域分支与退化感知门控融合

TikTok TechJam 2026 · Track 5 · Optional Upgrade

对应赛题研讨会 slide 2 的 "Optional upgrade: add a frequency branch (FFT / DCT features) and fuse it with the spatial branch"。

**状态**：代码已实现并跑通，接线正确性已验证。**有效性尚未验证** —— CIFAKE 分辨率不足以支撑结论，需在 SID_Set 上重跑消融。

---

## 1. 设计动机

### 1.1 频域信号从哪来

Diffusion 与 GAN 生成图像的过程包含反复的上采样。这个过程会在高频带留下准周期性的痕迹。真实照片的高频主要来自传感器噪声与真实细节，不具备周期性。因此频谱视角可以暴露纯语义骨干不会强调的伪影。

### 1.2 为什么必须加门控

这是本方案与常规"加个 FFT 分支"的核心区别。

官方评分为：

```
Final Score = 0.50 × AUC_clean + 0.50 × AUC_robust
```

而官方鲁棒性矩阵中的 JPEG 压缩、高斯模糊、Resize 下采样，**破坏的恰好是频域分支所依赖的那个频带**。

固定权重融合会遇到这个问题：

```
干净图  → 频域分支输出可信      → 应当采纳
压缩图  → 频域分支依然在输出    → 但依据已被破坏，实为噪声
```

固定权重无法区分这两种情况，会把噪声以同等权重加入最终判断。由于 robust 占 50% 权重，净效果可能为负。

**门控的作用**：从图像自身估计退化程度，据此决定频域分支的话该听多少。

---

## 2. 架构

```
                    ┌─ 空域塔 (OpenCLIP ViT-H/14) ──────→ spatial_logit
                    │
输入图像 ───────────┤
                    │                    ┌──→ freq_logit
                    └─ 频域分支 ─────────┤
                       (FFT log 幅度谱)  └──→ degradation = [hf_ratio, blockiness]
                                                    │
                                                    ↓
                                            gate = σ(MLP(degradation))
                                                    │
                    logit = spatial_logit + gate × freq_logit
```

### 2.1 频谱提取管线

```
反归一化 → 亮度通道 Y → 减均值 → Hann 窗 → 2D FFT
→ fftshift → log(1+|·|) → 样本级标准化 → 小 CNN
```

每一步都有理由，去掉任意一步都会引入无关变量：

| 步骤 | 原因 |
|---|---|
| 反归一化 | 需要在物理像素尺度上做 FFT，归一化后的张量做 FFT 无物理意义 |
| 亮度通道 | 对 RGB 三通道分别做 FFT，主要编码的是色彩平衡而非伪影 |
| 减均值 | 去除 DC 分量，它只反映整体亮度，与生成痕迹无关 |
| **Hann 窗** | **不加窗时图像左右边界的不连续会在频谱产生十字形伪影，强度远超真实生成指纹** |
| 样本级标准化 | 绝对能量反映曝光条件，不是伪造证据 |

### 2.2 融合规则：加法调制而非凸组合

```python
logit = spatial_logit + gate * freq_logit      # 本方案
# 而不是
logit = w * spatial_logit + (1 - w) * freq_logit   # 常见写法
```

差别在**失败时的下界**：

- 凸组合下，若频域分支输出垃圾而权重未降到 0，会**污染**空域判断，结果可能劣于纯空域塔。
- 加法调制下，`gate → 0` 时表达式退化为 `spatial_logit`，即纯空域塔本身。

**即：该分支最坏情况是无用，不会是有害。**

这个下界已通过 `gate_mode="off"` 的消融实测确认（见 §4.2）。

### 2.3 三种门控模式（用于消融）

| `gate_mode` | 行为 | 用途 |
|---|---|---|
| `"learned"` | `gate = σ(MLP(degradation))` | 主方案 |
| `"fixed"` | `gate = 0.5`（常数） | 对照：门控是否优于固定权重 |
| `"off"` | 分支不参与，直接返回 `spatial_logit` | 健全性检查：应与纯空域塔一致 |

---

## 3. 退化描述子

门控的输入是一个二维向量。用两个信号而非一个，是因为**它们覆盖不相交的失效模式**。

### 3.1 hf_ratio：径向高频能量占比

频谱上距中心半径 > 0.5（归一化半宽）的能量，占总能量的比例。

**实测有效范围**（512×512 随机噪声图，见 §4.1）：

| 退化 | hf_ratio | 相对 clean |
|---|---|---|
| clean | 0.6669 | — |
| blur σ=0.5 | 0.6485 | −2.8% |
| **blur σ=2.0** | **0.5172** | **−22%** ✅ |
| **resize 0.25×** | **0.5272** | **−21%** ✅ |

### 3.2 hf_ratio 的盲区：JPEG

**这是本方案实测得到的最重要的发现。**

| 退化 | hf_ratio |
|---|---|
| clean | 0.6669 |
| jpeg q=90 | 0.6669 |
| jpeg q=70 | 0.6670 |
| jpeg q=50 | 0.6666 |
| jpeg q=30 | 0.6683 |

**五档数值几乎完全相同，q=30 甚至略高于 clean。**

**成因**：JPEG 量化在移除自然高频的同时，引入了 8×8 块边界。块边界本身是阶跃边缘，也是强高频信号。两个效应在径向能量比这个粗粒度指标下相互抵消。质量因子越低，块效应越强，因此 q=30 反而略高。

**这不是实现 bug，是"高频能量占比"这个特征本身的结构性局限。**

> ⚠️ **对使用 SRM 高通残差的方案同样适用。** SRM 滤波器是空域高通卷积核，与 FFT 取高频是傅里叶对，二者在数学上抓取同一类信息。任何基于"高频能量强弱"的退化估计，都会在 JPEG 上遇到同样的抵消问题。官方矩阵 15 个条件中有 4 个是 JPEG，这是最大的一块。

### 3.3 blockiness：针对 JPEG 的补充信号

既然 JPEG 的特征是引入 8×8 网格边界，就直接在空域测量这个网格。

测量网格线**上**的梯度能量与网格**外**的梯度能量之比：

- 未压缩图像没有偏好网格，比值接近 1
- JPEG 压缩后网格线凸显，比值 > 1，且随质量下降单调上升

**已知风险（尚未证伪）**：CLIP 预处理会将图像 resize 到 224×224，这会打乱原图的 8×8 网格对齐。若实测发现 blockiness 在 resize 之后同样失效，则需要将该指标移到 resize 之前计算，这需要改动数据管线。

---

## 4. 实测结果

### 4.1 频谱管线验证（512×512 随机噪声图）

用随机噪声图测试的理由：高频丰富，退化效果最明显。若连噪声图上都看不出响应，真实照片上更不可能。

```
clean       hf_ratio=0.6669
jpeg90      hf_ratio=0.6669
jpeg70      hf_ratio=0.6670
jpeg50      hf_ratio=0.6666
jpeg30      hf_ratio=0.6683
blur0.5     hf_ratio=0.6485
blur2       hf_ratio=0.5172
resize0.25  hf_ratio=0.5272
```

**结论**：模糊与缩放检测有效；JPEG 检测失效（原因见 §3.2）。

### 4.2 参数量与结构验证

```
learned  gate=0.720 freq=0.45M total=0.633B
fixed    gate=0.500 freq=0.45M total=0.633B
off      gate=0.000 freq=0.45M total=0.633B
```

- 频域分支仅 **0.45 M** 参数
- 融合塔总量 **0.633 B**（OpenCLIP ViT-H/14 视觉塔 632 M + 频域 0.45 M + 头）
- 距 2 B 上限余量充足

> 📌 **附带修正**：`clip_tower.py` 的 docstring 写 "Vision encoder parameters: ~986 M"，**该数值有误**。986 M 是 ViT-H-14 完整 checkpoint（vision + text）的参数量，而代码中 `self.backbone = clip_model.visual` 只取视觉塔。实测：
> ```
> full=986M  visual=632M
> ```
> 这直接影响双塔方案的 2 B 余量计算：按 632 M 算，CLIP-H + DINOv3-H+ 约 1.47 B，而非 1.83 B。建议同步修正 docstring 与 README 中的参数量声明。

### 4.3 CIFAKE 四组消融

**数据**：CIFAKE 子集，train 1000 real + 1000 fake，val 500+500，calibration 500+500（与 val 不重叠）。2 epochs，batch 64。

| 配置 | val AUC |
|---|---|
| `clip_h`（纯空域） | 0.9949 |
| `fuse_clip_off` | 0.9942 |
| `fuse_clip_fixed` | 0.9939 |
| `fuse_clip`（门控） | 0.9943 |

**健全性检查通过**：`fuse_clip_off` 与 `clip_h` 相差 0.0007，确认融合层在分支关闭时正确退化为纯空域塔，加法调制的下界保证成立。

**有效性无法判定**：四个数值全部落在 0.001 区间内，统计上不可区分。CIFAKE（32×32、单一生成器 SD1.4、1000 张训练样本）在 0.99 附近饱和，无法区分这些架构。

### 4.4 门控行为诊断（CIFAKE）

对训练后的 `fuse_clip` 模型，在 64 张 val fake 图上施加各种退化：

```
cond           gate      hf  blocky
clean        0.6733  0.4263  0.0000
jpeg90       0.6733  0.4262  0.0000
jpeg70       0.6733  0.4266  0.0000
jpeg50       0.6733  0.4269  0.0000
jpeg30       0.6733  0.4262  0.0000
blur0.5      0.6733  0.4247  0.0000
blur2        0.6731  0.4347  0.0000
resize0.5    0.6733  0.4231  0.0000
resize0.25   0.6732  0.4304  0.0000
```

**两个问题：**

**① gate 在所有条件下恒定 0.6733** —— 这不是 bug。CIFAKE 是 32×32，上采样到 224 时 bicubic 插值产生的高频完全淹没了原图的退化痕迹。门控的输入本身没有变化，输出自然恒定。这是数据集的物理限制，换 SID_Set 才能观察真实行为。

**② blocky 全为精确的 0.0000** —— 这是实现问题。`blockiness_score` 的接入 patch 未生效，`degradation` 仍为一维，`fusion_tower` 中 `degradation.shape[1] > 1` 判为假而填零。**下方 §5 给出的代码是已修正版本**，包含 blockiness 且 `degradation` 为二维。使用前请以 §6.2 的检查确认。

---

## 5. 完整代码

### 5.1 `models/freq_branch.py`

```python
"""Frequency-domain branch for AIGC detection.

Rationale
---------
Diffusion and GAN generators synthesise images through repeated upsampling,
which leaves quasi-periodic traces in the high-frequency band.  Real photographs
have high-frequency content dominated by sensor noise and genuine detail, which
is not periodic.  A spectral view can therefore expose artefacts that a purely
semantic backbone does not emphasise.

Known weakness (measured, must be reported rather than hidden)
--------------------------------------------------------------
JPEG compression, Gaussian blur and downscale-upscale all attenuate exactly the
band this branch relies on.  Under the official robustness matrix -- which
weights AUC_robust at 50% and includes four JPEG levels -- an unconditioned
frequency branch can hurt the final score.  This module therefore also emits a
degradation descriptor so the fusion layer can down-weight it when the signal
is gone.

Pipeline
--------
    de-normalise -> luminance -> zero-mean -> Hann window -> 2D FFT
    -> fftshift -> log magnitude -> per-sample standardisation -> small CNN

The Hann window matters: without it the discontinuity at the image border
produces a cross-shaped artefact in the spectrum far stronger than any
generator fingerprint.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _hann_2d(size: int, device, dtype) -> torch.Tensor:
    """Separable 2D Hann window of shape ``[1, 1, size, size]``."""
    w = torch.hann_window(size, periodic=False, device=device, dtype=dtype)
    return (w[:, None] * w[None, :])[None, None]


class SpectrumExtractor(nn.Module):
    """Turn a normalised image batch into a standardised log-magnitude spectrum.

    Args:
        size: Spectrum resolution.  Input is resized to ``size x size`` before
            the FFT so spectra are comparable across inputs.
        norm_mean: Per-channel mean used by the upstream preprocessing.
        norm_std: Per-channel std used by the upstream preprocessing.
        use_window: Apply a Hann window before the FFT.

    Shape:
        input  ``[B, 3, H, W]``  (normalised)
        output ``[B, 1, size, size]``
    """

    def __init__(
        self,
        *,
        size: int = 256,
        norm_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        norm_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
        use_window: bool = True,
    ) -> None:
        super().__init__()
        self.size = size
        self.use_window = use_window
        self.register_buffer("norm_mean", torch.tensor(norm_mean).view(1, 3, 1, 1))
        self.register_buffer("norm_std", torch.tensor(norm_std).view(1, 3, 1, 1))
        # ITU-R BT.601 luma coefficients
        self.register_buffer(
            "luma", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img = x * self.norm_std + self.norm_mean
        y = (img * self.luma).sum(dim=1, keepdim=True)

        if y.shape[-1] != self.size or y.shape[-2] != self.size:
            y = F.interpolate(
                y, size=(self.size, self.size), mode="bilinear", align_corners=False
            )

        # Remove DC so the spectrum does not encode overall brightness
        y = y - y.mean(dim=(2, 3), keepdim=True)

        if self.use_window:
            y = y * _hann_2d(self.size, y.device, y.dtype)

        spec = torch.fft.fft2(y.float(), norm="ortho")
        spec = torch.fft.fftshift(spec, dim=(-2, -1))
        mag = torch.log1p(spec.abs())

        # Absolute energy reflects exposure, not forgery
        mean = mag.mean(dim=(1, 2, 3), keepdim=True)
        std = mag.std(dim=(1, 2, 3), keepdim=True)
        return (mag - mean) / (std + 1e-6)


def radial_hf_ratio(spectrum: torch.Tensor, *, inner: float = 0.5) -> torch.Tensor:
    """Fraction of spectral energy outside a central radius.

    A cheap, differentiable proxy for how much high-frequency content survived.
    Measured response: blur sigma=2.0 gives -22%, resize 0.25x gives -21%.

    Blind to JPEG by construction -- see ``blockiness_score``.

    Args:
        spectrum: ``[B, 1, S, S]`` fftshifted log-magnitude spectrum.
        inner: Radius threshold as a fraction of the half-width.

    Returns:
        ``[B, 1]`` ratio in ``[0, 1]``.
    """
    _, _, s, _ = spectrum.shape
    coords = (
        torch.arange(s, device=spectrum.device, dtype=torch.float32) - (s - 1) / 2.0
    )
    rr = (coords[:, None] ** 2 + coords[None, :] ** 2).sqrt() / ((s - 1) / 2.0)
    hf_mask = (rr > inner).to(spectrum.dtype)[None, None]

    energy = spectrum.abs()
    hf = (energy * hf_mask).sum(dim=(1, 2, 3))
    total = energy.sum(dim=(1, 2, 3)) + 1e-6
    return (hf / total).unsqueeze(1)


def blockiness_score(x_luma: torch.Tensor, *, block: int = 8) -> torch.Tensor:
    """Spatial-domain JPEG blocking artefact measure.

    ``radial_hf_ratio`` alone cannot see JPEG.  Measured across q=90/70/50/30 it
    returns 0.6669 / 0.6670 / 0.6666 / 0.6683 -- effectively constant -- because
    quantisation removes natural high frequencies while introducing 8x8 block
    edges that are themselves high-frequency, and the two effects cancel in a
    radial energy ratio.

    This measures gradient energy *on* the block grid relative to gradient
    energy *off* it.  Uncompressed images have no preferred grid so the ratio is
    near 1; JPEG pushes it above 1 as quality drops.

    Args:
        x_luma: ``[B, 1, H, W]`` luminance in roughly ``[0, 1]``.
        block: JPEG block size.

    Returns:
        ``[B, 1]`` blockiness ratio.
    """
    dh = (x_luma[..., :, 1:] - x_luma[..., :, :-1]).abs()
    dv = (x_luma[..., 1:, :] - x_luma[..., :-1, :]).abs()

    on_w = (torch.arange(dh.shape[-1], device=x_luma.device) + 1) % block == 0
    on_h = (torch.arange(dv.shape[-2], device=x_luma.device) + 1) % block == 0

    on = dh[..., :, on_w].mean(dim=(1, 2, 3)) + dv[..., on_h, :].mean(dim=(1, 2, 3))
    off = dh[..., :, ~on_w].mean(dim=(1, 2, 3)) + dv[..., ~on_h, :].mean(dim=(1, 2, 3))
    return (on / (off + 1e-6)).unsqueeze(1)


class FrequencyBranch(nn.Module):
    """Small CNN over the log-magnitude spectrum.

    Args:
        size: Spectrum resolution.
        width: Base channel width.
        out_dim: Feature dimension returned alongside the logit.
        dropout: Dropout before the branch head.
        norm_mean / norm_std: Upstream preprocessing constants.

    Returns from :meth:`forward`:
        ``(logit [B], feature [B, out_dim], degradation [B, 2])``

        ``degradation`` holds the radial high-frequency ratio and the blockiness
        score.  Two signals are needed because they cover disjoint failure
        modes: hf_ratio detects blur and rescaling but is blind to JPEG, while
        blockiness detects JPEG specifically.
    """

    def __init__(
        self,
        *,
        size: int = 256,
        width: int = 32,
        out_dim: int = 256,
        dropout: float = 0.1,
        norm_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        norm_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        super().__init__()
        self.extractor = SpectrumExtractor(
            size=size, norm_mean=norm_mean, norm_std=norm_std
        )

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.GELU(),
            )

        self.stem = nn.Sequential(
            block(1, width),
            block(width, width * 2),
            block(width * 2, width * 4),
            block(width * 4, width * 8),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(nn.Linear(width * 8, out_dim), nn.GELU())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, 1))
        self.out_dim = out_dim

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spectrum = self.extractor(x)
        hf = radial_hf_ratio(spectrum)

        img = x * self.extractor.norm_std + self.extractor.norm_mean
        luma = (img * self.extractor.luma).sum(dim=1, keepdim=True)
        block = blockiness_score(luma)

        h = self.pool(self.stem(spectrum)).flatten(1)
        feat = self.proj(h)
        logit = self.head(feat).squeeze(1)
        return logit, feat, torch.cat([hf, block], dim=1)

    def param_count(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
```

### 5.2 `models/fusion_tower.py`

```python
"""Spatial + frequency fusion tower.

Fusion rule
-----------
    logit = spatial_logit + gate * freq_logit

Additive modulation rather than a convex combination, so a gate driven to zero
degrades the model to the pure spatial tower instead of to something worse.
This bounds the downside of adding the branch, which matters because the
frequency signal is destroyed by exactly the degradations the official matrix
tests.  Verified empirically: gate_mode="off" reproduces the spatial tower to
within 0.0007 AUC.

The gate is predicted from a degradation descriptor extracted from the same
image: the radial high-frequency energy ratio and a JPEG blockiness score.
Two signals are needed because they cover disjoint failure modes -- measured
hf_ratio drops ~22% under blur and ~21% under 0.25x rescaling, but is flat
across JPEG q=90..30 (0.6669 / 0.6670 / 0.6666 / 0.6683).

Set ``gate_mode="fixed"`` to ablate the gate against a constant weight, and
``gate_mode="off"`` to disable the branch entirely (sanity check).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.freq_branch import FrequencyBranch

# CLIP preprocessing constants, kept in sync with train._clip_preprocess
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class FusionTower(nn.Module):
    """Spatial backbone fused with a degradation-aware frequency branch.

    Args:
        variant: ``"clip_h"``, ``"pe_l"`` or ``"pe_g"``.
        unfreeze_blocks: Trailing transformer blocks to unfreeze.
        proj_dim: Spatial projection width.
        dropout: Dropout in both heads.
        spectrum_size: FFT resolution for the frequency branch.
        freq_width: Base channel width of the frequency CNN.
        gate_mode: ``"learned"``, ``"fixed"`` or ``"off"``.
        fixed_gate: Constant weight used when ``gate_mode="fixed"``.
        pretrained: Load pretrained spatial weights.
    """

    def __init__(
        self,
        *,
        variant: str = "clip_h",
        unfreeze_blocks: int = 4,
        proj_dim: int = 512,
        dropout: float = 0.1,
        spectrum_size: int = 256,
        freq_width: int = 32,
        gate_mode: str = "learned",
        fixed_gate: float = 0.5,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if gate_mode not in ("learned", "fixed", "off"):
            raise ValueError(f"gate_mode must be learned/fixed/off, got {gate_mode!r}")

        self.gate_mode = gate_mode
        self.fixed_gate = fixed_gate
        self.variant = variant

        if variant == "clip_h":
            from models.clip_tower import CLIPTower

            self.spatial = CLIPTower(
                unfreeze_blocks=unfreeze_blocks, proj_dim=proj_dim, dropout=dropout
            )
            self.input_size = 224
            norm_mean, norm_std = _CLIP_MEAN, _CLIP_STD
        else:
            from models.pe_tower import PETower

            self.spatial = PETower(
                variant=variant,
                unfreeze_blocks=unfreeze_blocks,
                proj_dim=proj_dim,
                dropout=dropout,
                pretrained=pretrained,
            )
            self.input_size = self.spatial.input_size
            cfg = self.spatial.data_config
            norm_mean = tuple(cfg["mean"])
            norm_std = tuple(cfg["std"])

        self.frequency = FrequencyBranch(
            size=spectrum_size,
            width=freq_width,
            dropout=dropout,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        # Gate over the degradation descriptor.  Bias initialised positive so
        # the branch starts contributing rather than being switched off at
        # step 0.  in_dim is probed from the branch so it stays correct if the
        # descriptor gains further signals.
        with torch.no_grad():
            probe = torch.zeros(1, 3, self.input_size, self.input_size)
            _, _, degradation = self.frequency(probe)
        self.gate_net = nn.Sequential(
            nn.Linear(degradation.shape[1], 16), nn.GELU(), nn.Linear(16, 1)
        )
        nn.init.constant_(self.gate_net[-1].bias, 1.0)

    def forward(
        self, x: torch.Tensor, *, return_parts: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Return a scalar logit per image.

        Args:
            x: ``[B, 3, H, W]`` preprocessed for the spatial backbone.
            return_parts: Also return the individual logits, the gate and the
                degradation descriptor, for diagnostics and ablation tables.
        """
        spatial_logit = self.spatial(x)

        if self.gate_mode == "off":
            if not return_parts:
                return spatial_logit
            zero = torch.zeros_like(spatial_logit)
            return {
                "logit": spatial_logit,
                "spatial_logit": spatial_logit,
                "freq_logit": zero,
                "gate": zero,
                "hf_ratio": zero,
                "blockiness": zero,
            }

        freq_logit, _, degradation = self.frequency(x)

        if self.gate_mode == "fixed":
            gate = torch.full_like(spatial_logit, self.fixed_gate)
        else:
            gate = torch.sigmoid(self.gate_net(degradation)).squeeze(1)

        logit = spatial_logit + gate * freq_logit

        if not return_parts:
            return logit
        return {
            "logit": logit,
            "spatial_logit": spatial_logit,
            "freq_logit": freq_logit,
            "gate": gate,
            "hf_ratio": degradation[:, 0],
            "blockiness": (
                degradation[:, 1]
                if degradation.shape[1] > 1
                else torch.zeros_like(gate)
            ),
        }

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def param_count(self) -> dict[str, int]:
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "spatial": sum(p.numel() for p in self.spatial.parameters()),
            "frequency": sum(p.numel() for p in self.frequency.parameters()),
        }
```

### 5.3 `train.py` 的三处改动

以下脚本用锚点替换，三个 `assert` 全部通过才写入文件，任一锚点缺失则报错且不修改。

```python
p = 'train.py'
s = open(p).read()

# --- 1. _build_model: 新增融合塔分支 ---
old = '''    else:
        raise ValueError(f"Unknown backbone: {backbone!r}. Choose 'clip_h' or 'dino_h'.")'''
new = '''    elif backbone.startswith("fuse_"):
        from models.fusion_tower import FusionTower
        variant = {
            "fuse_clip": "clip_h", "fuse_clip_fixed": "clip_h",
            "fuse_clip_off": "clip_h",
        }[backbone]
        gate_mode = {
            "fuse_clip": "learned", "fuse_clip_fixed": "fixed",
            "fuse_clip_off": "off",
        }[backbone]
        model = FusionTower(
            variant=variant, unfreeze_blocks=4, proj_dim=512,
            dropout=0.1, gate_mode=gate_mode,
        )
    else:
        raise ValueError(
            f"Unknown backbone: {backbone!r}. "
            "Choose 'clip_h', 'dino_h' or one of the 'fuse_clip*' variants."
        )'''
assert old in s, "build_model anchor missing"
s = s.replace(old, new)

# --- 2. _preprocess_fn: 融合塔复用 CLIP 预处理 ---
old = '''def _preprocess_fn(backbone: str) -> T.Compose:
    if backbone == "clip_h":
        return _clip_preprocess()
    return _dino_preprocess()'''
new = '''def _preprocess_fn(backbone: str) -> T.Compose:
    if backbone == "clip_h" or backbone.startswith("fuse_clip"):
        return _clip_preprocess()
    return _dino_preprocess()'''
assert old in s, "preprocess anchor missing"
s = s.replace(old, new)

# --- 3. argparse choices ---
old = 'parser.add_argument("--backbone", choices=["clip_h", "dino_h"], required=True)'
new = ('parser.add_argument("--backbone", '
       'choices=["clip_h", "dino_h", "fuse_clip", "fuse_clip_fixed", "fuse_clip_off"], '
       'required=True)')
assert old in s, "argparse anchor missing"
s = s.replace(old, new)

open(p, 'w').write(s)
print("train.py patched OK")
```

> 若仓库中已加入 PE-Core tower（`pe_l` / `pe_g`），锚点文本会不同，需相应调整；`FusionTower` 本身已支持 `variant="pe_l"` / `"pe_g"`。

---

## 6. 复现步骤

### 6.1 前置条件

```bash
# CLIP ViT-H/14 (DFN-5B) 权重，约 3.9 GB
python scripts/download_clip.py
ls -lh weights/clip_h/open_clip_pytorch_model.bin
```

依赖与主仓库一致，无额外包。频域分支只用到 `torch.fft`。

### 6.2 结构与接线检查

```bash
# 检查 1：degradation 必须是二维
python -c "
import torch
from models.freq_branch import FrequencyBranch
fb = FrequencyBranch().eval()
with torch.no_grad(): _, _, d = fb(torch.randn(2, 3, 224, 224))
print('degradation shape:', d.shape)   # 期望 torch.Size([2, 2])
assert d.shape[1] == 2, 'blockiness not wired in'
print('OK')
"

# 检查 2：三种门控模式与参数量
python -c "
import torch
from models.fusion_tower import FusionTower
for mode in ['learned', 'fixed', 'off']:
    m = FusionTower(variant='clip_h', gate_mode=mode).eval()
    n = m.param_count()
    with torch.no_grad(): o = m(torch.randn(2, 3, 224, 224), return_parts=True)
    print(f\"{mode:8s} gate={o['gate'].mean():.3f} \"
          f\"freq={n['frequency']/1e6:.2f}M total={n['total']/1e9:.3f}B\")
"
```

预期输出：

```
degradation shape: torch.Size([2, 2])
OK
learned  gate=0.720 freq=0.45M total=0.633B
fixed    gate=0.500 freq=0.45M total=0.633B
off      gate=0.000 freq=0.45M total=0.633B
```

### 6.3 频谱管线验证

**用高分辨率图测，不要用 CIFAKE。**

```bash
python -c "
import torch, numpy as np
from PIL import Image
from models.freq_branch import FrequencyBranch
from data.transforms import apply_transform
import torchvision.transforms as T

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
fb = FrequencyBranch(norm_mean=MEAN, norm_std=STD).eval()
tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])

img = Image.fromarray((np.random.rand(512, 512, 3) * 255).astype('uint8'))
print(f\"{'cond':11s} {'hf':>8s} {'blocky':>8s}\")
for name, kw in [('clean', None),
                 ('jpeg90', ('jpeg_compression', 90)),
                 ('jpeg70', ('jpeg_compression', 70)),
                 ('jpeg50', ('jpeg_compression', 50)),
                 ('jpeg30', ('jpeg_compression', 30)),
                 ('blur0.5', ('gaussian_blur', 0.5)),
                 ('blur2', ('gaussian_blur', 2.0)),
                 ('resize0.25', ('resize', 0.25))]:
    im = img if kw is None else apply_transform(img, kw[0], value=kw[1])
    with torch.no_grad(): _, _, d = fb(tf(im).unsqueeze(0))
    print(f'{name:11s} {d[0,0]:8.4f} {d[0,1]:8.4f}')
"
```

**判据**：
- `hf` 列：`clean > blur0.5 > blur2`，`resize0.25` 明显低于 clean —— 已验证成立
- `blocky` 列：随 JPEG 质量下降单调上升 —— **尚未验证**，是当前最需要确认的一项

### 6.4 四组消融

```bash
mkdir -p logs
for bb in clip_h fuse_clip_off fuse_clip_fixed fuse_clip; do
  CUDA_VISIBLE_DEVICES=1 python train.py --backbone $bb \
    --data <DATASET_ROOT> \
    --output runs/sid_$bb --epochs 10 --batch-size 64 --workers 32 \
    > logs/sid_$bb.log 2>&1
  echo "=== $bb done ==="
done

for f in logs/sid_*.log; do echo -n "$(basename $f .log): "; grep "Best val AUC" $f; done
```

四组含义：

| `--backbone` | 作用 |
|---|---|
| `clip_h` | 纯空域基线 |
| `fuse_clip_off` | 健全性检查，应与 `clip_h` 一致 |
| `fuse_clip_fixed` | 固定权重融合 |
| `fuse_clip` | 退化感知门控融合 |

**必须使用同一份数据划分、同一套增强策略、同一个评估脚本，结果才可比。**

### 6.5 门控行为诊断

```bash
python -c "
import torch, glob
from PIL import Image
from models.fusion_tower import FusionTower
from data.transforms import apply_transform
import train

m = FusionTower(variant='clip_h', gate_mode='learned')
ck = torch.load('runs/sid_fuse_clip/best.pt', map_location='cpu', weights_only=False)
sd = ck.get('model', ck.get('state_dict', ck))
print(m.load_state_dict(sd, strict=False))
m.eval().cuda()
tf = train._preprocess_fn('fuse_clip')

paths = sorted(glob.glob('<DATASET_ROOT>/val/fake/*'))[:64]
imgs = [Image.open(p).convert('RGB') for p in paths]

conds = [('clean', None),
         ('jpeg90', ('jpeg_compression', 90)), ('jpeg70', ('jpeg_compression', 70)),
         ('jpeg50', ('jpeg_compression', 50)), ('jpeg30', ('jpeg_compression', 30)),
         ('blur0.5', ('gaussian_blur', 0.5)), ('blur2', ('gaussian_blur', 2.0)),
         ('resize0.5', ('resize', 0.5)), ('resize0.25', ('resize', 0.25))]
print(f\"{'cond':11s} {'gate':>7s} {'hf':>7s} {'blocky':>7s}\")
for name, kw in conds:
    b = torch.stack([tf(im if kw is None else apply_transform(im, kw[0], value=kw[1]))
                     for im in imgs]).cuda()
    with torch.no_grad(): o = m(b, return_parts=True)
    print(f\"{name:11s} {o['gate'].mean():7.4f} \"
          f\"{o['hf_ratio'].mean():7.4f} {o['blockiness'].mean():7.4f}\")
"
```

**这张表是判断门控是否真正学到东西的核心证据。** 若 gate 随退化程度单调下降，说明门控学会了"什么时候该闭嘴"；若恒定不变，说明退化描述子在该数据集上没有区分度。

**若能画出「横轴 JPEG 质量因子、纵轴 gate 均值」的曲线并呈单调下降，这是报告中最有说服力的一张图。**

---

## 7. 判定准则

**唯一判据：官方 Final Score。**

```
Final Score = 0.50 × AUC_clean + 0.50 × AUC_robust
```

结果填入下表：

| 配置 | AUC_clean | AUC_robust | **Final Score** | 最差条件 AUC |
|---|---|---|---|---|
| `clip_h` | | | | |
| `fuse_clip_off` | | | | |
| `fuse_clip_fixed` | | | | |
| `fuse_clip` | | | | |

决策规则：

| 结果 | 决定 |
|---|---|
| `fuse_clip` Final Score 最高 | 采用门控融合 |
| `fuse_clip_fixed` ≈ `fuse_clip` | 门控未学到东西，用固定权重即可 |
| 两个融合都不如 `clip_h` | **砍掉频域分支，如实报告负结果** |
| `fuse_clip_off` 与 `clip_h` 差异明显 | 接线有问题，先排查再看其余结果 |

**Clean AUC 单独升高但 Final Score 下降时，应当砍掉。** 由于 robust 占 50%，只看 clean 会做出错误决策。

**负结果同样有价值。** "我们实现了频域分支，实测发现其退化描述子对 JPEG 失明，门控无法在四档 JPEG 上正确降权，因此 Final Score 未获提升，最终未采用" —— 这段带数据的分析本身就是一份合格的误差分析材料，比塞入一个未经验证的模块更有说服力。

---

## 8. 已知限制

1. **hf_ratio 对 JPEG 失明**，成因见 §3.2。已加入 blockiness 作为补充信号，但该信号在 224 resize 之后是否仍然有效，尚未验证。

2. **blockiness 的 resize 对齐问题**：CLIP 预处理会将图像缩放到 224×224，可能打乱原图 8×8 网格。若 §6.3 中 blocky 列不呈单调上升，需将该指标移到 resize 之前计算，这会改动数据管线。

3. **频谱输入是 resize 后的图像**：bicubic 插值本身会改变高频成分。理想做法是给频域分支一份原分辨率的 crop。当前为最小改动实现，未采用。

4. **CIFAKE 无法用于验证本方案**：32×32 上采样到 224 时，插值产生的高频完全淹没原图退化痕迹，门控输入恒定。CIFAKE 仅可用于跑通训练链路。

5. **未做跨生成器验证**：频域指纹是生成器特异的，跨生成器时最易失效。WildFake 按生成器分层，可做留一生成器交叉验证，这是检验本方案泛化能力的关键实验，尚未进行。

---

## 9. 与 RGPA 方案的关系

若同时存在基于 SRM 残差的局部取证分支（RGPA），需注意以下几点。

### 9.1 信号重叠

**SRM 高通残差与 FFT 高频谱在数学上是傅里叶对。** SRM 滤波器是空域高通卷积核，FFT 取高频是频域高通，二者抓取同一类信息。直接堆叠收益递减。

### 9.2 真正的互补维度

不是"空域 vs 频域"，而是：

| | RGPA | 本方案 |
|---|---|---|
| 空间粒度 | **局部**（7×7 patch，可定位） | **全局**（整图谱，抓周期性指纹） |
| 权重驱动自 | patch 间的**相对**残差强弱 | 整图的**绝对**退化程度 |
| 决定什么 | 图内哪些区域更重要 | 该分支整体是否可信 |
| 作用域 | 分支内部 | 分支与空域塔之间 |

**两个机制正交，可以叠加。**

### 9.3 建议的合并形式

```python
logit = spatial_logit + gate(degradation) × rgpa_logit
```

RGPA 的 patch 聚合完整保留，外面套一层退化感知门控。`FusionTower` 中将 `self.frequency` 替换为 RGPA 模块即可，门控逻辑无需改动 —— 唯一要求是 RGPA 的 `forward` 返回 `(logit, feature, degradation)` 三元组。

### 9.4 对 RGPA 直接适用的实测发现

- **§3.2 的 JPEG 抵消效应对 SRM 残差同样成立。** RGPA 风险清单中"JPEG 会直接改变高通残差"的判断正确，但方向不是简单衰减 —— 自然高频被移除的同时块边缘被引入，在能量类指标上可能相互抵消。建议实测确认。

- **CIFAKE 上任何低层信号都测不出来。** RGPA 文档中"CIFAKE 只用于跑通训练链路，不用于判断 RGPA 效果"这一判断，本方案的实测数据支持（门控在所有退化条件下恒定 0.6733）。

- **RGPA 的固定权重融合缺少下界保证。** 建议改为加法调制，或至少在消融中加入等价于 `gate_mode="off"` 的对照组，确认融合在分支失效时不会劣于纯空域塔。

### 9.5 建议的推进顺序

1. 两个分支各自在 SID_Set 上跑完消融，得到各自的 Final Score
2. 依据数据决定：都有效则合并；只有一个有效则保留该一个；都无效则交纯空域单塔
3. **不要在两个分支都未被证明有价值之前做集成工作**

---

## 附录：文件清单

| 文件 | 说明 |
|---|---|
| `models/freq_branch.py` | 频谱提取、退化描述子、频域 CNN |
| `models/fusion_tower.py` | 融合塔，三种门控模式 |
| `train.py` | 三处锚点替换，见 §5.3 |

新增依赖：无。
新增参数量：0.45 M（融合塔总计 0.633 B，2 B 限制内余量充足）。
