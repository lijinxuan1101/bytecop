# AI生成图像检测 — 技术方案 (v7)

> TikTok TechJam 赛题：构建一个能区分AI生成图像与真实图像的检测器，要求在真实世界后处理场景下（压缩、缩放、模糊、色彩调整等）保持鲁棒性，并对未见过的生成器具备泛化能力。

> **v6决策说明**：主线与官方建议统一为“预训练空间backbone baseline → 官方增强与完整评估 → 显式FFT/DCT频域分支 → logit融合验证决策互补 → feature concat作为最终候选 → 概率校准”。空间塔采用`OpenCLIP ViT-H/14 (laion2b_s32b_b79k)`，频域塔采用轻量CNN处理亮度通道的频谱或高通残差。DINOv3降为可选对照，不再作为默认第二塔。Day 1仍先完成最小空间单塔baseline，不预先搭建双塔管道。

> **v7更新**：新增CIFAKE Stage 1方案思路验证。先完成OpenCLIP-H linear probe，再逐步解冻最后2个、可选最后4个Transformer block。Stage 1直接复用项目中已有的Robust评测函数，不在本节重复定义退化操作。

---

## 1. 问题背景

- 生成式AI能以极低成本、大规模产出照片级真实图像，带来虚假信息、身份冒用、电商欺诈、版权等平台风险。
- 检测的核心难点有两个：
  - **Generalization（泛化性）**：新生成器（diffusion、GAN、下个月的新模型）会留下和训练集不同的"指纹"。
  - **Robustness（鲁棒性）**：图像在真实传播链路中会被压缩、裁剪、模糊、调色，这些操作会破坏检测器依赖的部分信号。

---

## 2. 项目约束


| 约束       | 内容                                                        |
| -------- | --------------------------------------------------------- |
| 时间       | 几天内交（Hackathon节奏）                                         |
| 算力（本地训练） | 充足                                                        |
| 模型参数量上限  | 单模型/组合总量 **< 2B**                                         |
| 工程原则     | 空间塔使用H级，但先完成最小单塔baseline；频域塔和融合必须逐级加入，并通过频带消融、退化测试与Final Score证明收益 |


---



## 3. 核心思路：显式构造空间域与频率域互补

**之前版本的问题**：直接假设“CLIP管语义、DINOv3管高频纹理”。但两个模型接收的都是RGB自然图像，DINOv3也不是显式频域模型，因此这种双塔只能验证不同预训练表征的互补，不能保证不同频域的互补。

**修正后的立场**：

> OpenCLIP空间塔接收正常RGB图像，学习内容、结构和高层视觉表征；FFT/DCT频域塔接收显式频谱或高通残差，学习频率能量分布、周期模式和生成器伪影。两种输入域由架构明确分离，再通过logit与feature concat实验验证实际互补性。

因此，**OpenCLIP-H空间单塔baseline是实现起点**。频域塔是官方建议的optional upgrade；如果空间—频域融合没有稳定提升，最终仍提交表现更可靠的空间单塔。DINOv3仅作为时间允许时的额外对照，不承担默认“高频塔”角色。

---



## 4. 模型选型



### 4.1 候选backbone


| 模型                                 | 参数量(vision-only) | 说明                                        |
| ---------------------------------- | ---------------- | ----------------------------------------- |
| OpenCLIP ViT-L/14                  | ~304M            | 空间塔备用降级方案                                 |
| **OpenCLIP ViT-H/14**              | **视觉塔约632M；完整图文模型约986M，须以实际checkpoint核算为准** | **主空间塔；checkpoint `laion2b_s32b_b79k`** |
| **轻量FFT/DCT CNN**                | **目标 <50M**     | **主频域塔；输入log-magnitude频谱或高通残差**          |
| DINOv3 ViT-H+                      | ~840M            | 可选表征对照，不是默认频域塔                         |
| SigLIP2 ViT-giant-opt-patch16-384  | ~1.87B           | 备选主塔（若CLIP-H欠拟合再启用）；HF: `google/siglip2-giant-opt-patch16-384` |


模型决策：**baseline直接使用OpenCLIP-H，不从双塔开始。**完成空间单塔和官方评估后，再训练轻量频域塔；先用logit融合验证决策互补，再训练feature concat融合头。最终报告至少包含空间单塔、频域单塔、logit融合和feature concat四组结果。

**关于SigLIP2-giant的定位**：
- 单塔参数量约1.87B已接近官方2B上限，只能搭配极小分类头；是否还能增加频域塔必须按实际加载参数重新核算。
- 因此仅作为**CLIP-H baseline在clean/robust AUC上明显欠拟合、且距离2B预算仍有余量**时的备选升级方案，而不是默认主塔。
- 若启用SigLIP2-giant，方案将改为**单塔+更强增强/更长训练**路线，取代原双塔计划，并在报告中明确说明取舍。

### 4.2 参数量核算


| 组合方案               | 参数量                                               |
| ------------------ | ------------------------------------------------- |
| 单塔 L 级             | ~300M                                             |
| **OpenCLIP-H空间塔 + 轻量频域塔** | **预计明显低于2B；必须按实际加载模块统计，不计未加载的文本塔** |
| OpenCLIP-H + DINOv3-H+（可选对照） | 仅作额外实验；不再是主方案 |
| 单塔 SigLIP2-giant（若启用则放弃双塔） | ~1.87B backbone，加上头几乎打满2B                          |


---



## 5. 架构设计与实验路线



### 5.1 主线：空间单塔baseline逐步升级为空间—频域双塔

```
阶段A：最小空间baseline
输入图片 → 基础预处理 → OpenCLIP ViT-H/14视觉塔 → 线性分类头 → spatial logit

阶段B：baseline鲁棒化
阶段A → 官方单退化训练增强 → 完整clean/robust评估 → 首个可提交空间模型

阶段C：增加显式频域塔
几何尺寸调整后的图片 → 亮度通道 → FFT/DCT或高通残差 → 轻量CNN → frequency logit

阶段D：低成本验证决策互补
standardized spatial logit + frequency logit → 加权logit融合 → 比较各退化条件AUC

阶段E：特征级融合
spatial feature + frequency feature → 各自LayerNorm/Projection → concat → MLP → fused logit

阶段F：最终校准
最终候选logit → 独立calibration split上的temperature scaling → 最终概率
```

每个阶段都必须先形成可复现结果，再进入下一阶段：


| 实验              | 目的                                      |
| --------------- | --------------------------------------- |
| ① OpenCLIP-H空间单塔 | 首个baseline；验证训练收敛、推理和clean AUC，不加入频域逻辑 |
| ② 空间塔 + 官方增强 | 验证增强对clean/robust AUC的影响，形成首个可提交版本 |
| ③ FFT/DCT频域单塔 | 独立验证频域信号；重点报告clean、JPEG、Blur、Resize条件 |
| ④ 标准化加权logit融合 | 低成本验证决策级互补，避免两塔logit尺度不同造成假融合 |
| ⑤ LayerNorm/Projection + feature concat | 最终双塔候选；验证分类头之前是否存在更深层互补 |
| ⑥ DINOv3或其他backbone | 可选对照，不影响主线交付 |


**答辩叙事逻辑**：OpenCLIP-H建立空间baseline → 官方增强提升真实场景鲁棒性 → 显式FFT/DCT分支捕捉生成频谱伪影 → logit融合验证决策互补 → feature concat学习更深层空间—频域关系 → 仅保留能提升官方Final Score和unseen-generator AUC的模块。这一结构直接对应官方“spatial branch + optional frequency branch”建议。

### 5.2 Cross-Attention融合：移出主计划，列为可选延伸

**移出原因**：

- 空间ViT输出token，而轻量频域CNN默认输出特征图；表示类型、维度和分辨率均不对齐，需要额外token化与投影
- 此前给出的示例代码未实现token对齐、mask、池化等必要环节，不能直接跑通
- 复杂度高、调试成本高，在①-⑤都还没跑完前投入这个方向风险大

**保留位置**：仅作为空间—频域logit融合与feature concat完成后的深化方向，不属于主线。

### 5.3 FFT/DCT频域分支

空间分支和频域分支只共享几何尺寸调整，不共享OpenCLIP通道归一化。推荐数据流：

```text
decoded RGB image → official degradation（训练时可选一种）→ resize/crop
    ├→ OpenCLIP normalize → OpenCLIP-H视觉塔
    └→ luminance → FFT/DCT/high-pass → 频谱标准化 → 轻量CNN
```

频域实现原则：

- 先转换到亮度通道（luminance）再做频谱分析，而非直接对RGB操作
- 去除或抑制直流分量（DC component），它主要反映整体亮度均值，不是伪影信号
- 对频谱做样本级标准化（normalize），避免不同图像整体能量差异干扰
- 第一版使用`fftshift(log1p(abs(FFT(Y))))`的完整二维幅度谱，不提前硬切高频
- 后续通过低频遮挡、中频环、高频外围和高通残差做频带消融，证明模型实际依赖的频段
- 频域塔先独立训练分类头，确认其在clean条件有效且退化曲线合理后再融合

---



## 6. 训练策略



### 6.1 冻结 + 轻量微调

- OpenCLIP空间backbone先完全冻结，仅训练线性分类头，建立linear-probe baseline
- baseline稳定后，只微调最后2-4层transformer block + 分类头，冻结其余部分
- 目的：避免在相对小规模训练集上过拟合到已见过的生成器，保留预训练泛化能力
- 轻量频域CNN从头训练；先独立训练，再在feature concat阶段视显存和稳定性决定是否联合微调



### 6.2 数据增强

**官方评估变换集（评估时必须覆盖，不能自行删减）：**


| 变换                            | 参数                         | 真实场景对应                                           |
| ----------------------------- | -------------------------- | ------------------------------------------------ |
| JPEG压缩                        | quality = 90/70/50/30      | 社交媒体re-encode、消息传输                               |
| 高斯模糊                          | σ = 0.5/1.0/2.0（多档强度都要测）   | 失焦、截图平滑                                          |
| **Resize（downscale→upscale）** | **0.5× / 0.25× 缩小后放大回原尺寸** | **缩略图生成、CDN resize——官方明确要求的测试项，不能因为"钟意crop"而删掉** |
| 高斯噪声                          | σ = 0.02/0.05/0.10         | 低光传感器噪声                                          |
| 色彩抖动                          | 亮度/对比度/饱和度 ±20%            | 滤镜App、自动增强                                       |
| 中心裁剪                          | crop 80%                   | 头像裁剪、构图                                          |


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
- 空间塔与频域塔的logit尺度可能不同，诊断融合前先用validation统计量标准化，再搜索少量固定权重；
- 两塔融合优先组合 **logits**，而不是直接平均已经校准的概率；
- 对最终选定的logit或feature-concat模型在calibration split上重新做 **temperature scaling**；单塔温度不能直接沿用到融合结果；
- calibration split只用于拟合校准参数，不用于模型选择、早停或超参数搜索；
- 报告 **ECE (Expected Calibration Error)** 和 **Brier Score**，而不是只看accuracy/AUC
- 这一步是官方要求"输出calibrated probability"的具体落地方式



### 6.4 数据防泄漏

之前版本未提及，是一个容易踩的坑：

- 数据集划分应按**生成器来源、原始图像来源、图像族**为单位切分train/val/test
- 避免同一张真实图片、或它衍生出的生成/编辑版本，同时出现在train和test里——否则会得到虚高的、不可靠的评估分数

---



## 7. 数据集


| 数据集          | 规模（待核实）                                         | 特点                                             | 用途                                                                           |
| ------------ | ----------------------------------------------- | ---------------------------------------------- | ---------------------------------------------------------------------------- |
| **CIFAKE**   | 12万张，32×32低分辨率，单一生成器(SD1.4)                     | 分辨率极低，放大喂给ViT-L基本无法验证高频取证和高分辨率鲁棒性              | **仅用于Day1的冒烟测试**：验证Dataset/DataLoader能跑通、loss是否下降、输出格式是否正确。**不用于验证模型有效性**    |
| **SID_Set**  | 官方数据卡确认总数约30万，含Real/Full Synthetic/Tampered三类标签 | 社交媒体场景图片，逼真度高                                  | 正式训练主力数据（用Real+Full Synthetic两类）。**各类实际数量以下载后的metadata统计为准，不预先假设三类严格各10万均衡** |
| **WildFake** | 大规模，层级结构                                        | 覆盖GAN到diffusion多种生成器，来自Civitai/Midjourney等真实社区 | Cross-generator泛化测试（选训练时未见过的生成器子集）                                           |


**修正**：Day1不应该只用CIFAKE验证——由于CIFAKE分辨率过低，无法反映真实场景的高分辨率鲁棒性问题。建议**Day1就尽早引入少量SID_Set或WildFake的高分辨率子集**做小规模验证，CIFAKE只做最基础的pipeline冒烟测试。

---



## 8. 评估策略



### 8.1 鲁棒性评估矩阵（补齐官方参数全集）

之前版本的评估矩阵只挑了每种变换的一档参数，遗漏了Resize、Noise、Color Jitter，以及多档Blur强度。应覆盖官方候选参数的完整集合：


| Condition                           | 说明                     |
| ----------------------------------- | ---------------------- |
| Clean                               | 原始未处理图像                |
| JPEG q=90 / 70 / 50 / 30            | 四档都测，不只测q30            |
| Blur σ=0.5 / 1.0 / 2.0              | 三档都测                   |
| Resize 0.5× / 0.25× (down→up)       | 官方明确要求的测试项             |
| Gaussian Noise σ=0.02 / 0.05 / 0.10 | 三档都测                   |
| Color Jitter ±20%                   | —                      |
| Crop 80%                            | —                      |
| Unseen generator                    | 训练时未见过的生成器（WildFake子集） |




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
- **Complexity vs Feasibility**：空间—频域双塔、feature concat和cross-attention都增加实现成本；本方案先用轻量频域塔和logit融合验证价值，再决定是否联合训练复杂融合头。
- **Frequency signal vs Robustness**：FFT/DCT伪影可能在clean图片上明显，却会被JPEG、Blur和Resize破坏；频域分支必须在官方退化矩阵下验证，不能只凭clean AUC保留。

---



## 11. Stage 1：方案思路验证

### 11.1 实验目标

使用小规模CIFAKE数据验证OpenCLIP-H单塔的训练、推理与评测链路是否可靠，并初步比较冻结特征与局部微调的差异。本阶段只用于方案验证和流程排错，不用于判断最终模型的真实泛化能力，也不用于决定FFT/SRM分支是否有效。

CIFAKE原始图像分辨率仅为32×32；放大至OpenCLIP输入尺寸不会恢复已经缺失的高频细节，因此本阶段分数不作为正式方案上限。

### 11.2 实验设置

#### 数据划分

| 数据集 | 划分 | Real（0） | AI（1） | 总数 |
| ------ | ---- | --------: | -----: | ---: |
| CIFAKE | Train | 4,000 | 4,000 | 8,000 |
| CIFAKE | Validation | 1,000 | 1,000 | 2,000 |

- 固定随机种子并保存样本ID；
- Train与Validation不得包含重复或近重复图片；
- S1、S2、可选S3以及后续低层分支验证使用同一划分；
- 当前Validation结果属于阶段性验证结果，不表述为独立测试集结果。

#### 实验指标

直接调用项目中已经定义的Robust评测函数，不在本节重复描述退化类型与参数。统一报告：

$$
FinalScore=0.5AUC_{clean}+0.5AUC_{robust}
$$

### 11.3 OpenCLIP单塔baseline

#### 11.3.1 实验S1：Linear probe

- 冻结整个OpenCLIP视觉backbone；
- 只训练线性二分类头；
- 使用clean图片训练；
- 不加入官方退化增强；
- 根据Validation AUC选择checkpoint。

**实验目的**：验证OpenCLIP原始预训练特征是否已经包含真实图与AI图的可分信息，并建立低成本、变量受控的空间单塔基准。

#### 11.3.2 实验S2：局部微调最后2个Block

- 解冻OpenCLIP最后2个Transformer block；
- 其余backbone参数保持冻结；
- 训练解冻部分与分类头；
- 使用与S1完全相同的数据划分和基础预处理；
- 仍然只使用clean图片训练。

**实验目的**：验证少量任务适配能否提升检测性能，同时观察Robust AUC是否因适配而下降。

#### 11.3.3 实验S3：局部微调最后4个Block（可选）

仅当S2相对S1产生稳定收益时启动：

- 解冻最后4个Transformer block；
- 其余设置与S2保持一致；
- 将S3作为独立实验记录，不把“解冻2～4层”混写为同一配置。

### 11.4 结果记录

| 实验 | 训练范围 | AUC_clean | AUC_robust | Final Score |
| ---- | -------- | ---------: | ----------: | ----------: |
| S1 | Linear head |  |  |  |
| S2 | Head + 最后2个Block |  |  |  |
| S3（可选） | Head + 最后4个Block |  |  |  |

除汇总表外，保存每个Validation样本的ID、真实标签、logit、概率与评测条件，供后续FFT/SRM互补性分析使用。

### 11.5 Stage 1退出条件

- S1能够完整训练、保存并恢复checkpoint；
- Clean与已有Robust评测函数均能正常运行；
- S1与S2完成相同数据划分下的公平比较；
- 训练结果可复现，且未发现明显数据泄漏；
- 输出完整结果表与逐样本预测；
- 确定进入正式数据实验时采用linear probe、最后2层微调或可选最后4层微调。

---



## 12. 分阶段执行计划（P0–P3）


| 优先级    | 阶段              | 必须完成的任务                                                                     | 数据集与退出条件                                                                           |
| ------ | --------------- | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **P0** | Stage 1 | 按第11节使用CIFAKE完成OpenCLIP-H单塔S1 linear probe与S2最后2层微调；S3仅在S2稳定提升后启动 | 固定8,000/2,000划分，复用已有Robust评测函数。退出条件：checkpoint可恢复、预测可复现、输出AUC_clean、AUC_robust、Final Score和逐样本结果 |
| **P0** | Stage 2 | 将Stage 1确认的OpenCLIP-H训练配置迁移到SID-Set正式数据，并加入官方single-transform增强与完整评估 | SID-Set主训练。退出条件：完成数据泄漏检查，输出完整评测结果，形成首个可提交空间单塔版本 |
| **P1** | Day 2后半 | 实现并独立训练FFT/DCT轻量频域塔；使用与空间塔完全相同的数据划分与退化样本 | 输出频域单塔在Clean、JPEG、Blur、Resize等条件下的AUC和频带消融结果；确认没有只学习数据集来源捷径 |
| **P1** | Day 3前半 | 保存两个单塔对相同validation样本的logit；标准化后测试固定权重logit融合 | 统计预测相关性、错误重合和条件AUC；若融合稳定提升或错误分歧明显，进入feature concat |
| **P2** | Day 3后半 | 分别LayerNorm/Projection后进行feature concat，训练轻量MLP融合头；与最佳logit融合公平比较 | WildFake未见生成器子集验证；只有提升官方Final Score且不显著损害最差退化条件才作为最终模型 |
| **P2** | Day 4前半 | 对最终候选使用独立calibration split做temperature scaling；报告ECE/Brier和FP/FN分析 | calibration split不参与模型选择；输出最终鲁棒性表格和错误分析 |
| **P3** | Day 4余量 | DINOv3对照、double-transform消融、DCT替代FFT、DIRE或Cross-Attention | 仅在空间baseline、频域塔、融合、校准、演示和交付材料全部完成后启动 |


最终交付以“OpenCLIP-H空间单塔 + 官方增强 + 完整评估”的P0版本为底线。主创新候选是“空间塔 + 显式FFT/DCT频域塔”；logit融合负责低成本验证决策互补，feature concat负责验证分类头之前的特征互补。只有带来可复现收益的增量才进入最终模型。

---



## 13. 参考文献

- DINOv3 (Meta AI, 2025)：[https://arxiv.org/abs/2508.10104](https://arxiv.org/abs/2508.10104)
- Rethinking Cross-Generator Image Forgery Detection through DINOv3（关键发现：DINOv3依赖全局低频可迁移线索，而非生成器特定高频伪影）：[https://arxiv.org/abs/2511.22471](https://arxiv.org/abs/2511.22471)
- Intermediate Representations are Strong AI-Generated Image Detectors (CLIP vs DINOv2中间层特征对比)：[https://arxiv.org/pdf/2605.04358](https://arxiv.org/pdf/2605.04358)
- Raising the bar of AI-generated image detection with CLIP (Cozzolino et al., CVPR 2024)
- SAFE (KDD 2025)：模型预处理阶段crop优于强制缩放的insight（不替代评估阶段的Resize退化测试）
- DDA (NeurIPS 2025)：真实图与合成图后处理对齐、避免频率偏置的insight
- Robust Deepfake Detection: NTIRE 2026 Challenge方案（DINOv2-Giant多流融合）：[https://arxiv.org/pdf/2604.25889](https://arxiv.org/pdf/2604.25889)
