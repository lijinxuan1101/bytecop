# 低层取证分支：Residual-Guided Patch Aggregation（RGPA）

RGPA是主技术方案Stage 2的取证分支。它与OpenCLIP-H在Stage 3进行logit融合。

## 1. 目标与定位

OpenCLIP-H提供全局语义和结构信息；RGPA通过固定高通残差、patch级编码和双向软聚合提取局部像素取证信息。

RGPA回答：在整图完成固定高通残差提取后，patch级共享编码与高低残差双向聚合能否提供独立于OpenCLIP-H的可复现收益？

```text
OpenCLIP-H空间单塔
    ↓
RGPA
    ↓
OpenCLIP-H + RGPA标准化加权logit融合
```

## 2. 模型结构

### 2.1 输入

- 输入固定为224×224 RGB图像；
- patch size固定为32×32，共7×7=49个patch；
- 空间分支与RGPA共享概率性单退化和几何预处理；
- 两个分支不共享通道归一化：OpenCLIP使用官方normalize，RGPA使用像素尺度RGB。

固定224×224是当前时间预算下的工程选择。它可能损失高分辨率原图中的取证痕迹，原始分辨率或多尺度patch不纳入当前主线。

### 2.2 数据流

```text
RGB图像
  ↓
概率性单退化
  ↓
一次性几何预处理至224×224
  ↓
整图SRM-inspired固定高通残差
  ↓
残差图切分为49个32×32 patch
  ↓
共享轻量CNN逐patch编码
  ↓
图内标准化残差能量
  ↓
高、低残差双向软聚合
  ↓
forensic logit
```

高通滤波必须先作用于整图，再切patch，避免逐patch卷积padding制造人工边界。

### 2.3 双向软聚合

对第 `i` 个patch计算残差能量，并在同一张图内部标准化：

$$
a_i=E_{\mathrm{residual},i}
\qquad
\hat{a}_i=\frac{a_i-\mu_a}{\sigma_a+\epsilon}
$$

分别计算高、低残差权重：

$$
w_i^{high}=\mathrm{softmax}\left(\frac{\hat{a}_i}{\tau}\right)
\qquad
w_i^{low}=\mathrm{softmax}\left(-\frac{\hat{a}_i}{\tau}\right)
$$

聚合patch特征：

$$
z^{high}=\sum_i w_i^{high}z_i
\qquad
z^{low}=\sum_i w_i^{low}z_i
$$

$$
z_{\mathrm{forensic}}
=\mathrm{concat}\left(z^{high},z^{low}\right)
$$

图内标准化使权重表达patch在当前图像中的相对残差强弱，降低不同图像和退化条件绝对能量尺度的影响。当patch残差分布接近均匀时，高低聚合可能得到相似特征；这属于机制边界，应记录权重差异用于结果解释。

## 3. 训练

RGPA从零训练，SRM-inspired高通核保持冻结。训练使用与空间分支相同的数据划分和概率性单退化分布，输入为224×224像素尺度RGB。

## 4. 实验流程与退出条件

| 步骤 | 实验 | 退出条件 |
|---|---|---|
| 1 | RGPA | 输出Clean、各类退化、Robust AUC和Final Score，确认取证分支具有独立价值，且最差退化条件未明显恶化 |
| 2 | OpenCLIP-H + RGPA logit融合 | 标准化两类logit后选择固定权重，Final Score稳定提升 |

如果RGPA无明显独立价值，则不进入融合；如果融合无收益，则最终使用OpenCLIP-H空间单塔。

## 5. 数据适用性

CIFAKE原始分辨率为32×32，放大到224×224后产生的局部细节主要来自插值，因此只用于跑通RGPA训练链路，不用于判断取证分支效果。

RGPA的正式训练与评估使用SID-Set等原始分辨率足以支持局部取证的数据，并在格式、压缩与分辨率对齐后排除数据来源捷径。

## 6. 主要风险

- JPEG、Blur、Resize和Noise会直接改变高通残差，必须完整评估Robust AUC；
- 真实图与AI图必须采用完全相同的编码、几何处理和退化采样规则；
- 固定224×224可能损失高分辨率取证痕迹；
- 当高低权重接近时，双向聚合可能退化为近似重复特征。

## 7. 方法表述

> 受AIDE局部取证与全局语义融合思路启发，RGPA先在整图上提取SRM-inspired固定高通残差，再对残差patch进行共享编码，并依据图内标准化残差能量进行高低双向软聚合。它以软权重保留全部patch信息。实验上先训练RGPA，再用logit融合验证与OpenCLIP-H的互补。
