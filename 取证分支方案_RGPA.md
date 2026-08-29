# 低层取证分支设计：Residual-Guided Patch Aggregation（RGPA）

> 本文档对应v10主方案第5.3节"F4：局部patch取证（后续升级）"的详细设计。经过三轮团队review修订：第一轮修正命名与信号叠加问题；第二轮修正patch/高通顺序与图内标准化；本轮（第三轮）修正输入尺寸处理方式、单patch边界情况、标准化效果的解释方向，以及两处表述精度问题。**本模块仅在v10 Stage 2的F1（整图SRM）证明有效后才启动，当前时间预算下只做"OpenCLIP → 整图SRM → RGPA → Logit Fusion"这一条主线。**

---

## 0. 本轮（第三轮）修订摘要

| 问题 | 严重程度 | 修正 |
|---|---|---|
| 动态resize到可整除尺寸，引入插值痕迹、改变高频残差、与官方Resize退化叠加 | **必须修正** | 固定输入尺寸224×224、patch size 32×32，7×7=49个patch，不做额外的可整除性resize；空间分支与取证分支共享同一次标准几何预处理 |
| N=1时标准差计算无意义，可能产生非数值结果 | **必须修正** | 明确当N=1时跳过双向标准化和聚合，直接输出唯一patch特征；RGPA仅在N>1的数据（如SID-Set）上启用，CIFAKE只测试整图SRM链路 |
| 标准化效果的方向性解释写反了 | 表述修正 | 改为：残差尺度较大或patch间差异明显时，softmax容易产生尖锐分布；残差尺度较小或patch间差异微弱时，softmax容易趋于均匀；图内标准化用于统一不同图像间的相对尺度，而非直接对应"纹理丰富/简单" |
| "残差提取下沉到patch级"表述不准确（残差提取仍在整图完成） | 表述修正 | 改为"整图完成固定高通残差提取后，将残差表示从整图级编码改为patch级共享编码，并引入高低残差双向软聚合" |
| "性能差异只能归因于patch聚合"的因果表述过强（RGPA输出维度是F1的两倍，分类头也更大） | 表述修正 | 改为"两者共享相同的残差提取器和patch编码器主体，参数规模接近，主要结构差异来自patch级双向聚合"，不做强因果排他表述 |

---

## 1. 与v10主方案的对齐关系

v10主方案Stage 2定义了低层取证消融序列，本模块的定位：

| 阶段 | 内容 | 状态 |
|---|---|---|
| F1 | 整图SRM-inspired固定残差 | **必做，是RGPA的直接对照组** |
| F2/F3（FFT、NPD、BayarConv） | 其他候选信号 | 本轮时间预算下暂缓 |
| **RGPA（原F4）** | **残差引导的局部patch聚合（本文档主体）** | 仅在F1证明有效后启动 |

核心研究问题是单变量对照：**在整图完成固定高通残差提取后，将残差表示从整图级编码改为patch级共享编码、并引入高低残差双向软聚合，是否比整图级编码更有效**。这是一个受控的单变量实验，不需要同时引入FFT、NPD等其他信号源。

---

## 2. 核心架构（本轮修正输入尺寸处理）

### 2.1 输入尺寸：固定224×224，不做动态Resize

**问题**：此前版本采用"退化图像 → resize到可整除尺寸 → 高通残差"的流程。这个额外的resize步骤本身会引入插值痕迹、改变相邻像素关系和高频残差、与官方Resize退化叠加造成混淆，且"向下取最近可整除倍数"并不是真正意义上的最近尺寸（当原尺寸略超过某个倍数时，该策略总是向下裁切，产生不对称偏差）。

**修正**：固定标准输入尺寸为224×224，patch size取32×32：

$$
224 / 32 = 7, \qquad 7 \times 7 = 49 \text{个patch}
$$

不需要为patch划分额外做任何resize操作。图像的标准化几何预处理（resize/crop到224×224）只在整个pipeline中执行一次，空间分支（OpenCLIP-H）与取证分支（RGPA）共享这同一次几何处理结果，二者看到的是完全相同的输入图像。

若未来需要支持其他不能被32整除的输入尺寸，优先采用轻微裁剪（crop到最接近的可整除尺寸）或边缘补齐（padding），而不是再引入一次任意缩放。

### 2.2 完整数据流

```
图像
  ↓
概率性单退化（与v10 Stage 1/2使用完全相同的退化采样分布）
  ↓
标准几何预处理：resize/crop到224×224（与空间分支共享，仅此一次）
  ↓
整图SRM-inspired固定高通残差（在224×224残差图上一次性完成）
  ↓
残差图切patch：32×32网格，共49个patch，不重叠，纯空间操作
  ↓
共享轻量CNN（各patch共享权重，逐patch独立编码）
  ↓
每个patch输出：残差特征向量 z_i + 残差能量 a_i
  ↓
若 N=1（如调试用的极小patch配置）：直接输出该patch特征，跳过标准化与双向聚合
若 N>1（正式配置，N=49）：
    图内标准化：â_i = (a_i − μ_a) / (σ_a + ε)
    高、低残差双向软聚合（基于â_i的softmax加权池化）
  ↓
forensic logit
```

### 2.3 N=1时的处理（本轮新增）

当patch划分结果为N=1时（例如误用极小的patch配置、或图像本身小于一个patch），残差能量只有一个数值，计算标准差在数学上没有意义，某些实现下可能产生NaN或除零问题。

**明确规则**：
- N=1时，**跳过图内标准化和双向聚合**，直接把该唯一patch的特征向量作为forensic feature输出（不做高低聚合的拼接，因为只有一个patch，高低聚合退化为对同一个特征重复两次，没有意义）
- **RGPA的双向聚合机制只在N>1的数据上启用**。对于CIFAKE这类分辨率过低、任何合理patch size下都可能产生N=1的数据，仅用于测试"整图SRM残差提取→轻量CNN"这条链路本身是否能跑通，不测试RGPA的双向聚合部分
- 正式的RGPA效果验证使用SID-Set等能在224×224标准输入下产生49个patch的数据

### 2.4 图内标准化的效果说明（本轮修正方向性描述）

不同图片、不同退化条件下，残差能量的绝对尺度存在差异。若直接对未标准化的绝对残差值做softmax：

- **残差尺度较大、或patch间差异明显时**，softmax容易产生过度尖锐的权重分布（集中到极少数patch）
- **残差尺度较小、或patch间差异微弱时**，softmax容易趋于均匀分布（丧失区分度）

这两种情况都会让固定的温度参数 $\tau$ 在不同图片上表现不一致。图内标准化的作用是**统一不同图像之间的相对尺度**，使权重分布主要取决于patch间的相对残差差异，而不是被图像本身的绝对残差量级或退化类型左右：

$$
\hat{a}_i = \frac{a_i - \mu_a}{\sigma_a + \epsilon}
$$

$$
w_i^{high} = \text{softmax}(\hat{a}_i / \tau), \qquad w_i^{low} = \text{softmax}(-\hat{a}_i / \tau)
$$

$$
z^{high} = \sum_i w_i^{high} \cdot z_i, \qquad z^{low} = \sum_i w_i^{low} \cdot z_i, \qquad z_{forensic} = \text{concat}(z^{high}, z^{low})
$$

**已知局限（如实记录）**：当一张图片内部patch残差分布本身接近均匀时，$w^{high} \approx w^{low}$，此时$z^{high} \approx z^{low}$，双向聚合此时不提供额外信息，这是机制本身的性质，不是需要修复的问题。结果分析阶段应记录高低权重分布的差异程度，作为诊断该机制何时起作用的依据。

---

## 3. 核心代码骨架（本轮修正）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SRMInspiredResidual(nn.Module):
    """单个固定高通核，作用于整图（224×224），不作用于单个patch"""
    def __init__(self):
        super().__init__()
        kernel = torch.tensor([
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0]
        ], dtype=torch.float32) / 4.0
        self.register_buffer('kernel', kernel.view(1, 1, 5, 5).repeat(3, 1, 1, 1))

    def forward(self, x):
        # x: [B, 3, 224, 224]，与空间分支共享同一次几何预处理后的输入
        return F.conv2d(x, self.kernel, padding=2, groups=3)


def image_to_patches(residual_map, patch_size=32):
    """对已算好的残差图做纯空间切分。224/32=7，固定产生7x7=49个patch，不需要额外resize"""
    B, C, H, W = residual_map.shape
    assert H % patch_size == 0 and W % patch_size == 0, \
        f"输入尺寸{H}x{W}必须能被patch_size={patch_size}整除，请在几何预处理阶段固定为224x224"
    p = patch_size
    patches = residual_map.unfold(2, p, p).unfold(3, p, p)
    patches = patches.contiguous().view(B, C, -1, p, p).permute(0, 2, 1, 3, 4)
    return patches  # [B, N, C, p, p]，标准配置下 N=49


class SharedPatchEncoder(nn.Module):
    """逐patch共享权重的轻量CNN编码器，输入已经是残差图切出来的patch"""
    def __init__(self, feat_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.proj = nn.Linear(64, feat_dim)

    def forward(self, patch):
        feat = self.conv(patch).flatten(1)
        feat = self.proj(feat)
        residual_energy = patch.pow(2).mean(dim=(1, 2, 3))
        return feat, residual_energy


class BidirectionalWeightedPooling(nn.Module):
    """图内标准化 + 高低残差双向软聚合；N=1时退化为直接输出"""
    def __init__(self, tau=1.0, eps=1e-6):
        super().__init__()
        self.tau = tau
        self.eps = eps

    def forward(self, patch_feats, residual_energy):
        # patch_feats: [B, N, feat_dim]; residual_energy: [B, N]
        B, N, D = patch_feats.shape

        if N == 1:
            # N=1时标准差无意义，跳过标准化与双向聚合，直接复制输出以保持维度一致
            z = patch_feats.squeeze(1)  # [B, D]
            forensic_feature = torch.cat([z, z], dim=-1)  # 维度对齐，但不代表真实的高低聚合
            weight_divergence = torch.zeros(B, device=patch_feats.device)
            return forensic_feature, weight_divergence

        mu = residual_energy.mean(dim=1, keepdim=True)
        sigma = residual_energy.std(dim=1, keepdim=True)
        a_hat = (residual_energy - mu) / (sigma + self.eps)  # 图内标准化

        w_high = torch.softmax(a_hat / self.tau, dim=1).unsqueeze(-1)
        w_low = torch.softmax(-a_hat / self.tau, dim=1).unsqueeze(-1)

        z_high = (patch_feats * w_high).sum(dim=1)
        z_low = (patch_feats * w_low).sum(dim=1)

        forensic_feature = torch.cat([z_high, z_low], dim=-1)
        weight_divergence = (w_high.squeeze(-1) - w_low.squeeze(-1)).abs().mean(dim=1)
        return forensic_feature, weight_divergence


class RGPA(nn.Module):
    """Residual-Guided Patch Aggregation：固定224x224输入，整图先做高通，残差图再切patch"""
    def __init__(self, patch_size=32, feat_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.srm = SRMInspiredResidual()
        self.encoder = SharedPatchEncoder(feat_dim)
        self.pooling = BidirectionalWeightedPooling()
        self.head = nn.Linear(feat_dim * 2, 1)  # 输出维度是F1 baseline的两倍（因拼接了high/low）

    def forward(self, x):
        # x假定已经是224x224，与空间分支共享的标准几何预处理结果，本模块不再做resize
        residual_map = self.srm(x)
        patches = image_to_patches(residual_map, self.patch_size)  # 标准配置下N=49

        B, N, C, p, _ = patches.shape
        patches_flat = patches.reshape(B * N, C, p, p)
        feats, energy = self.encoder(patches_flat)
        feats = feats.view(B, N, -1)
        energy = energy.view(B, N)

        forensic_feature, weight_divergence = self.pooling(feats, energy)
        forensic_logit = self.head(forensic_feature)
        return forensic_logit, forensic_feature, weight_divergence


class WholeImageSRMBaseline(nn.Module):
    """F1对照组：整图SRM残差，不切patch，复用SharedPatchEncoder以保持encoder主体一致"""
    def __init__(self, feat_dim=128):
        super().__init__()
        self.srm = SRMInspiredResidual()
        self.encoder = SharedPatchEncoder(feat_dim)
        self.head = nn.Linear(feat_dim, 1)  # 注意：输出维度是RGPA分类头的一半

    def forward(self, x):
        residual_map = self.srm(x)
        feat, _ = self.encoder(residual_map)
        logit = self.head(feat)
        return logit, feat
```

**关于容量可比性的准确表述**：`WholeImageSRMBaseline`与`RGPA`共享同一个`SRMInspiredResidual`残差提取器和`SharedPatchEncoder`编码器主体，两者参数规模接近。但RGPA最终拼接了`[z_high; z_low]`，输出特征维度是F1 baseline的两倍，分类头`nn.Linear`的输入维度也相应更大。因此**不宜声称"两者性能差异只能归因于patch聚合"**，更准确的表述是：**两者共享相同的残差提取器和patch编码器主体，参数规模接近，主要结构差异来自patch级双向聚合**（这一差异本身包含了输出维度翻倍带来的轻微容量增加，但该差异远小于两个完全不同架构之间的容量差距，不需要为此单独增加控制变量实验）。

---

## 4. 参数量估算

| 模块 | 参数量 |
|---|---|
| SRM-inspired高通核 | 0（固定，不参与训练） |
| 共享Patch CNN编码器 | 数万 |
| 双向加权池化 | 0（无可学习参数） |
| 分类头（RGPA，输入维度为F1的两倍） | 数百 |
| **合计** | **数万级别** |

叠加在v10主方案OpenCLIP-H视觉塔（约632M）之上，对2B预算无实质影响。

---

## 5. 实验路线

### 5.1 最小必要路线（保持不变）

```
OpenCLIP-H空间单塔（v10 Stage 1，已有）
        ↓
整图SRM baseline（F1，WholeImageSRMBaseline，输入224×224）
        ↓
RGPA（本文档主体，同样输入224×224，与F1共享几何预处理）
        ↓
标准化加权Logit融合（OpenCLIP-H spatial logit + RGPA forensic logit）
```

只保留一组核心对照：整图SRM vs RGPA，回答"将残差表示从整图级编码改为patch级共享编码并引入双向聚合，是否比整图级编码更有效"。F0/F2/F3/多档patch size/feature concat本轮暂缓。

### 5.2 与v10 Stage 2/3的衔接

| 步骤 | 内容 | 对应v10阶段 |
|---|---|---|
| 1 | 训练WholeImageSRMBaseline（F1），输入224×224，获得Clean/Robust AUC | Stage 2 |
| 2 | 训练RGPA，相同224×224输入、相同数据划分、相同退化分布、共享encoder主体 | Stage 2延伸 |
| 3 | 比较两者Final Score与最差退化条件AUC，确认patch级聚合是否有增益 | Stage 2→3过渡 |
| 4 | 若RGPA优于F1，将RGPA的forensic logit与OpenCLIP-H的spatial logit做标准化加权融合 | Stage 3A |
| 5 | （仅在时间充裕且Stage 3A显示明显互补时）考虑feature concat | Stage 3B，非必须 |

**退出条件**：只有RGPA相对F1的提升可复现、稳定，才继续推进到Stage 3；否则直接使用F1或跳过整个低层取证分支。

---

## 6. CIFAKE适用性说明（本轮更新）

- CIFAKE图像原始分辨率为32×32，与本方案固定的224×224标准输入不匹配
- 若强行将CIFAKE的32×32图像resize到224×224再按32×32切patch，得到的7×7=49个patch中的局部细节大多来自插值算法，并非原始图像的真实取证信号，patch间的差异主要反映插值伪影而非生成模型的真实残差
- **结论**：CIFAKE仅用于测试"整图SRM残差提取→轻量CNN"这条链路（即WholeImageSRMBaseline）本身能否跑通（张量形状、前向反向传播是否报错），**不用于测试RGPA的patch划分与双向聚合部分**
- RGPA的正式效果验证使用SID-Set等原始分辨率足以支撑224×224标准输入、且patch间局部细节真实可信的数据

---

## 7. 报告中的方法论表述（本轮修正措辞）

> 受AIDE（ICLR 2025）"局部取证特征与全局语义特征融合"思路启发，我们在v10主方案F1（整图SRM-inspired残差）证明具备独立判别力后，提出Residual-Guided Patch Aggregation（RGPA）：在整图（224×224）完成固定高通残差提取后，将残差表示从整图级编码改为patch级（32×32，共49个patch）共享编码，并引入基于图内标准化残差能量的高低双向软聚合。该设计与AIDE的DCT硬性top-k patch选择不同：（1）使用可微的双向softmax加权替代硬性选择，保留全部patch信息；（2）残差能量在参与聚合前先做图内z-score标准化，使权重分布反映patch间的相对关系，而不受不同图像绝对残差尺度和退化类型的影响；（3）通过与整图SRM baseline的对照验证patch级编码与双向聚合的增量价值——两者共享相同的残差提取器和patch编码器主体、参数规模接近，主要结构差异来自patch级双向聚合，而非默认这一改动必然有效。

---

## 8. 待验证的风险点

- **双向聚合的有效边界**：当patch残差分布均匀时双向聚合退化为重复特征，需在结果分析中报告weight_divergence统计量，说明该机制在哪些图像/退化条件下真正起作用
- **鲁棒性优先**：SRM-inspired残差对JPEG压缩、模糊、resize等退化高度敏感，必须在v10完整Robust评测矩阵下验证RGPA与F1的对比，不能仅看Clean AUC
- **数据对齐检查**：引入本分支后需重新确认真实图与AI图是否经过完全相同的编码、压缩、分辨率处理流程（对应v10第6.2节DDA原则）
- **输入尺寸的强约束**：本版本要求输入严格为224×224（或其他能被32整除的固定尺寸），若上游几何预处理逻辑发生变动，需同步检查`image_to_patches`的整除性assert是否仍然成立
