# AI生成图像检测 — 技术方案 (v4)

> TikTok TechJam 赛题：构建一个能区分AI生成图像与真实图像的检测器，要求在真实世界后处理场景下（压缩、缩放、模糊、色彩调整等）保持鲁棒性，并对未见过的生成器具备泛化能力。

> **v4决策说明**：在算力充足、目标优先追求最终效果的前提下，主方案直接采用H级双塔：`CLIP ViT-H/14 + DINOv3 ViT-H+`。B/L级不再作为前置训练阶段，仅在需要快速排查数据管线时作为调试替代。仍保留两个H级单塔的消融结果，用于验证融合收益。官方Final test采用单一退化，最终模型输出在独立calibration split上重新校准，全部模块总参数量必须严格小于2B。

---

## 1. 问题背景

- 生成式AI能以极低成本、大规模产出照片级真实图像，带来虚假信息、身份冒用、电商欺诈、版权等平台风险。
- 检测的核心难点有两个：
  - **Generalization（泛化性）**：新生成器（diffusion、GAN、下个月的新模型）会留下和训练集不同的"指纹"。
  - **Robustness（鲁棒性）**：图像在真实传播链路中会被压缩、裁剪、模糊、调色，这些操作会破坏检测器依赖的部分信号。

---

## 2. 项目约束

| 约束 | 内容 |
|---|---|
| 时间 | 几天内交（Hackathon节奏） |
| 算力（本地训练） | 充足 |
| 模型参数量上限 | 单模型/组合总量 **< 2B** |
| 工程原则 | 直接采用H级模型追求效果，但优先使用简单logit融合，并保留可独立提交的H级单塔；不在没有收益证据时继续叠加Cross-Attention等复杂模块 |

---

## 3. 核心思路修正：不预设分工，用消融验证

**之前版本的问题**：直接假设"CLIP管语义、DINOv3管高频纹理"，并据此设计双塔架构。这个假设未经验证，且与后续跨生成器检测研究对DINOv3表征的分析不符。

**修正后的立场**：

> CLIP提供语言监督形成的高层视觉表征；相关跨生成器检测研究发现，DINOv3的可迁移检测能力更多依赖全局低频线索，而非生成器特定的高频伪影（见参考文献第2项）。两者是否互补、如何互补，需要通过单塔与融合的消融实验验证，不预设分工。

因此，H级双塔是当前实现起点，但是否作为最终提交模型仍由消融结果决定；如果融合没有稳定提升，最终退回表现最好的H级单塔。

---

## 4. 模型选型

### 4.1 候选backbone

| 模型 | 参数量(vision-only) | 说明 |
|---|---|---|
| CLIP ViT-B/16 | ~86M | 仅作快速pipeline调试替代 |
| DINOv3 ViT-S/B | 21M~86M | 仅作快速pipeline调试替代 |
| CLIP ViT-L/14 | ~304M | 备用降级方案 |
| DINOv3 ViT-L | ~300M | 备用降级方案 |
| **CLIP ViT-H/14** | **~986M** | **主方案语义表征塔** |
| **DINOv3 ViT-H+** | **~840M** | **主方案自监督视觉表征塔** |

模型决策：**直接训练H级主方案，不把B/L级实验设为启动条件。**不过，“H级参数更多”不等于“双塔一定优于单塔”，因此最终报告仍须包含CLIP-H、DINOv3-H+和H+H融合三组结果。

### 4.2 参数量核算

| 组合方案 | 参数量 |
|---|---|
| 单塔 B/S 级 baseline | 21M~86M |
| 单塔 L 级 | ~300M |
| 双塔 L+L 融合（若消融证明值得） | ~604M |
| **双塔 H+H 主方案** | **backbone约1.826B；低于2B上限，但必须把投影层、分类头及融合模块计入最终总量** |

---

## 5. 架构设计与实验路线

### 5.1 主线：H级双塔与必要消融

```
输入图片
  → 官方现实变换增强
  ├→ CLIP ViT-H/14 → CLIP logit
  └→ DINOv3 ViT-H+ → DINO logit
  → 两塔logit融合
  → 独立calibration split上重新做 temperature scaling
  → 校准后的AIGC概率
```

正式实验直接使用两个H级backbone。可以顺序训练两塔以降低显存压力，并保存各塔独立输出，以便在同一数据划分上完成必要消融：

| 实验 | 目的 |
|---|---|
| ① CLIP-H单塔 | 语言监督H级baseline |
| ② DINOv3-H+单塔 | 自监督视觉H级baseline |
| ③ H+H两塔logit平均 | 主融合方式；融合后重新校准 |
| ④ H+H特征concat | 仅当③仍有明显改进空间且显存、参数预算允许时尝试 |
| ⑤ + FFT频域分支 | 可选；验证频域信号是否带来额外提升 |

**答辩叙事逻辑**：选择H级模型提高表征容量 → 用官方增强和数据对齐保证鲁棒性 → 用两个H级单塔消融证明双塔是否互补 → 仅保留能提升官方Final Score的融合方式。若双塔没有稳定优于最佳H级单塔，则最终提交表现更好的单塔，而不是为了复杂度强行融合。

### 5.2 Cross-Attention融合：移出主计划，列为可选延伸

**移出原因**：
- 两个backbone的token维度、token数量、预处理分辨率未必对齐，需要额外投影层
- 此前给出的示例代码未实现token对齐、mask、池化等必要环节，不能直接跑通
- 复杂度高、调试成本高，在①-⑤都还没跑完前投入这个方向风险大

**保留位置**：作为logit融合与特征concat完成后的深化方向，而不是主线的一部分。

### 5.3 FFT频域分支：需要更严谨的实现

之前给出的实现过于粗糙，直接对RGB三通道做FFT，容易学习到亮度和图像尺寸差异这类无关变量，而不是真正的生成伪影。改进方向：

- 先转换到亮度通道（luminance）再做频谱分析，而非直接对RGB操作
- 去除或抑制直流分量（DC component），它主要反映整体亮度均值，不是伪影信号
- 对频谱做样本级标准化（normalize），避免不同图像整体能量差异干扰
- 仅在H级单塔与logit融合完成后引入，先确认双塔本身有效后再加

---

## 6. 训练策略

### 6.1 冻结 + 轻量微调

- 预训练backbone已具备强泛化特征（CLIP在4亿图文对，DINOv3在17亿图像上训练）
- 只微调最后2-4层transformer block + 新增分类头，冻结其余部分
- 目的：避免在相对小规模训练集上过拟合到已见过的生成器，保留预训练泛化能力

### 6.2 数据增强

**官方评估变换集（评估时必须覆盖，不能自行删减）：**

| 变换 | 参数 | 真实场景对应 |
|---|---|---|
| JPEG压缩 | quality = 90/70/50/30 | 社交媒体re-encode、消息传输 |
| 高斯模糊 | σ = 0.5/1.0/2.0（多档强度都要测） | 失焦、截图平滑 |
| **Resize（downscale→upscale）** | **0.5× / 0.25× 缩小后放大回原尺寸** | **缩略图生成、CDN resize——官方明确要求的测试项，不能因为"钟意crop"而删掉** |
| 高斯噪声 | σ = 0.02/0.05/0.10 | 低光传感器噪声 |
| 色彩抖动 | 亮度/对比度/饱和度 ±20% | 滤镜App、自动增强 |
| 中心裁剪 | crop 80% | 头像裁剪、构图 |

**关于Resize vs Crop的关系（修正）**：
- 之前版本因为SAFE论文的insight（crop优于down-sample）而直接从增强表里删除了Resize，这个理解有偏差。SAFE的insight适用于**模型预处理阶段**（怎么把图像调整到网络输入尺寸时尽量保细节，优先用crop而非强行缩放）；但**评估鲁棒性时，官方明确要求测试down-sample/up-scale这类退化**，两者不冲突，应该同时存在：
  - 模型预处理：优先用裁剪保留细节
  - 训练时的鲁棒增强 + 评估：仍然要包含Resize/down-sample，让模型学会应对这种真实会发生的退化

**增强触发逻辑（修正，之前定义不清晰）**：
官方Final test对每张图片只施加一种退化，不测试多种退化组合。因此，**主训练配置必须以clean或single transform为主，并与官方测试分布保持一致**。double transform只能作为可选的附加训练实验，用于探索真实传播链路中的组合退化；不得混入官方验证矩阵，也不能默认其一定有效。

主训练配置：

```python
import random

def apply_official_style_augmentation(image):
    # 与官方单一退化测试口径保持一致
    if random.random() < 0.3:
        return image

    transform = random.choice(TRANSFORM_POOL)
    return transform(image)
```

可选的double-transform配置必须作为单独实验运行，并与主配置进行消融比较。只有当它能提升或至少不损害官方single-transform矩阵上的平均AUC与最差条件AUC时，才考虑纳入最终训练。

**两个关键坑（保留自v1，仍然成立）：**
1. 模型预处理阶段用crop保留细节，但训练/评估的增强集里必须包含官方要求的Resize退化项（见上）。
2. **真实图和AI图必须用同一套后处理流程对齐**，不能一边处理一边不处理——否则模型会学到"有没有被压缩过"这种虚假捷径特征（DDA, NeurIPS 2025）。

### 6.3 概率校准

校准流程统一如下：
- 从训练数据之外划出独立的 **calibration split**，且按生成器、原始图像来源和图像族隔离；
- 单塔结果可分别做temperature scaling，用于独立报告；
- 两塔融合优先平均 **logits**，而不是直接平均已经校准的概率；
- 对最终融合logit在calibration split上重新做 **temperature scaling**；单塔温度不能直接沿用到融合结果；
- calibration split只用于拟合校准参数，不用于模型选择、早停或超参数搜索；
- 报告 **ECE (Expected Calibration Error)** 和 **Brier Score**，而不是只看accuracy/AUC
- 这一步是官方要求"输出calibrated probability"的具体落地方式

### 6.4 数据防泄漏

之前版本未提及，是一个容易踩的坑：
- 数据集划分应按**生成器来源、原始图像来源、图像族**为单位切分train/val/test
- 避免同一张真实图片、或它衍生出的生成/编辑版本，同时出现在train和test里——否则会得到虚高的、不可靠的评估分数

---

## 7. 数据集

| 数据集 | 规模（待核实） | 特点 | 用途 |
|---|---|---|---|
| **CIFAKE** | 12万张，32×32低分辨率，单一生成器(SD1.4) | 分辨率极低，放大喂给ViT-L基本无法验证高频取证和高分辨率鲁棒性 | **仅用于Day1的冒烟测试**：验证Dataset/DataLoader能跑通、loss是否下降、输出格式是否正确。**不用于验证模型有效性** |
| **SID_Set** | 官方数据卡确认总数约30万，含Real/Full Synthetic/Tampered三类标签 | 社交媒体场景图片，逼真度高 | 正式训练主力数据（用Real+Full Synthetic两类）。**各类实际数量以下载后的metadata统计为准，不预先假设三类严格各10万均衡** |
| **WildFake** | 大规模，层级结构 | 覆盖GAN到diffusion多种生成器，来自Civitai/Midjourney等真实社区 | Cross-generator泛化测试（选训练时未见过的生成器子集） |

**修正**：Day1不应该只用CIFAKE验证——由于CIFAKE分辨率过低，无法反映真实场景的高分辨率鲁棒性问题。建议**Day1就尽早引入少量SID_Set或WildFake的高分辨率子集**做小规模验证，CIFAKE只做最基础的pipeline冒烟测试。

---

## 8. 评估策略

### 8.1 鲁棒性评估矩阵（补齐官方参数全集）

之前版本的评估矩阵只挑了每种变换的一档参数，遗漏了Resize、Noise、Color Jitter，以及多档Blur强度。应覆盖官方候选参数的完整集合：

| Condition | 说明 |
|---|---|
| Clean | 原始未处理图像 |
| JPEG q=90 / 70 / 50 / 30 | 四档都测，不只测q30 |
| Blur σ=0.5 / 1.0 / 2.0 | 三档都测 |
| Resize 0.5× / 0.25× (down→up) | 官方明确要求的测试项 |
| Gaussian Noise σ=0.02 / 0.05 / 0.10 | 三档都测 |
| Color Jitter ±20% | — |
| Crop 80% | — |
| Unseen generator | 训练时未见过的生成器（WildFake子集） |

### 8.2 指标

- **主指标**：ROC AUC（threshold-free，对类别不均衡鲁棒）
- 校准质量：ECE、Brier Score（见6.3）

### 8.3 官方Final Score

官方研讨会材料明确给出最终评分公式：

```
Final Score = 0.50 × AUC_clean + 0.50 × AUC_robust
```

其中：

- `AUC_clean`：干净测试集上的ROC AUC；
- `AUC_robust`：官方鲁棒性测试集上的ROC AUC；
- 两部分各占50%。

团队可以额外报告各退化条件AUC、平均退化AUC、最差条件AUC、ECE和Brier Score用于诊断，但这些辅助指标不能替代或改名混淆官方Final Score。

### 8.4 单一退化与附加组合退化评估

- **官方评估矩阵**：每个样本只应用一种退化，覆盖第8.1节列出的全部参数；
- **附加研究矩阵（可选）**：可以测试JPEG+Crop等组合退化，但必须单独标注为团队扩展实验；
- 模型选择首先依据官方Final Score，其次观察各条件AUC与最差条件AUC，避免平均值掩盖特定退化下的崩溃。

---

## 9. 方法对比实验：DIRE (Diffusion Reconstruction Error)

作为对比baseline纳入最终方法比较，而非替代主方案。**优先级降低，仅在核心系统（第5.1节①-⑤）完成后再尝试。**

### 9.1 原理

用预训练扩散模型对图像做"加噪→去噪"重建，计算重建误差：AI生成图像通常被扩散模型更精确重建（误差小），真实照片重建误差更大。

```python
def compute_dire(image, diffusion_model, scheduler, num_steps=20):
    latent = encode_to_latent(image)
    noised = ddim_inverse(latent, diffusion_model, scheduler, num_steps)
    reconstructed = ddim_sample(noised, diffusion_model, scheduler, num_steps)
    dire_map = torch.abs(latent - reconstructed)
    return dire_map
```

### 9.2 时间预估修正

之前估计"半天到一天"偏乐观。DDIM inversion、latent尺度对齐、扩散模型版本选择、以及推理本身的计算成本，都可能超出预期。**建议不预设固定时间上限，只在核心系统完成、且仍有余量时间时才启动，作为锦上添花的对比实验，而非必须交付项。**

---

## 10. Trade-off 讨论（报告核心部分）

- **Robustness vs Clean accuracy**：重度数据增强会让clean AUC略降，但robust AUC通常有提升，需用实际消融数据说话，而非预设"肯定值得"。
- **Generalization vs Specialization**：针对某一生成器调优的检测器在该生成器上分数高，但在新生成器上会明显下降，这是预期现象。
- **Complexity vs Feasibility**：双塔、cross-attention、FFT分支理论上能进一步提升精度，但每一步都有实现和调试成本；本方案的核心策略是**用消融实验证明复杂度的必要性，而不是默认复杂架构更好**。

---

## 11. 分阶段执行计划（P0–P3）

| 优先级 | 阶段 | 必须完成的任务 | 数据集与退出条件 |
|---|---|---|---|
| **P0** | Day 1 | 打通H级双塔数据管线、训练、推理、官方单退化增强和输出格式；建立独立calibration split；可临时用B级权重做代码冒烟测试，但正式实验直接使用H级 | CIFAKE仅做pipeline冒烟测试；少量SID_Set/WildFake高分辨率数据做H级端到端验证。退出条件：H级模型可复现输出clean/robust AUC及官方Final Score |
| **P0** | Day 2前半 | 用SID_Set训练CLIP-H与DINOv3-H+；保存两塔独立logit；完成官方全部single-transform矩阵、泄漏检查和单塔校准 | 退出条件：两个H级单塔均有完整结果，且至少一个形成可提交方案 |
| **P1** | Day 2后半–Day 3前半 | H+H logit平均并重新校准；与两个H级单塔公平比较 | WildFake未见生成器子集验证；只有融合稳定提升官方Final Score才作为最终模型 |
| **P2** | Day 3余量 | 特征concat融合；可选double-transform训练消融 | 必须证明不损害官方single-transform表现，并记录复杂度与推理成本 |
| **P3** | Day 4余量 | FFT分支、DIRE或Cross-Attention探索 | 仅在P0/P1交付物、鲁棒性表格、FP/FN分析和演示均已完成后启动 |

最终交付以P0完整可运行方案为底线；P1–P3均为有数据支撑才保留的增量，而不是必须堆叠的组件。

---

## 12. 参考文献

- DINOv3 (Meta AI, 2025)：https://arxiv.org/abs/2508.10104
- Rethinking Cross-Generator Image Forgery Detection through DINOv3（关键发现：DINOv3依赖全局低频可迁移线索，而非生成器特定高频伪影）：https://arxiv.org/abs/2511.22471
- Intermediate Representations are Strong AI-Generated Image Detectors (CLIP vs DINOv2中间层特征对比)：https://arxiv.org/pdf/2605.04358
- Raising the bar of AI-generated image detection with CLIP (Cozzolino et al., CVPR 2024)
- SAFE (KDD 2025)：模型预处理阶段crop优于强制缩放的insight（不替代评估阶段的Resize退化测试）
- DDA (NeurIPS 2025)：真实图与合成图后处理对齐、避免频率偏置的insight
- Robust Deepfake Detection: NTIRE 2026 Challenge方案（DINOv2-Giant多流融合）：https://arxiv.org/pdf/2604.25889
