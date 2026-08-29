# AI生成图像检测 — 技术方案 (v13)

> TikTok TechJam 赛题：构建一个能区分AI生成图像与真实图像的检测器，要求在真实世界后处理场景下（压缩、缩放、模糊、色彩调整等）保持鲁棒性，并对未见过的生成器具备泛化能力。

---

## 1. 问题背景

- 生成式AI能以极低成本、大规模产出照片级真实图像，带来虚假信息、身份冒用、电商欺诈、版权等平台风险。
- 检测的核心难点有两个：
  - **Generalization（泛化性）**：新生成器（diffusion、GAN、下个月的新模型）会留下和训练集不同的"指纹"。
  - **Robustness（鲁棒性）**：图像在真实传播链路中会被压缩、裁剪、模糊、调色，这些操作会破坏检测器依赖的部分信号。

---

## 2. 项目约束


| 约束       | 内容                                                                 |
| -------- | ------------------------------------------------------------------ |
| 时间       | 几天内交（Hackathon节奏）                                                  |
| 算力（本地训练） | 充足                                                                 |
| 模型参数量上限  | 单模型/组合总量 **< 2B**                                                  |
| 工程原则     | 先完成OpenCLIP-H空间baseline，再验证轻量RGPA取证分支，最后用logit融合证明互补；所有增量以Final Score为准 |


---



## 3. 核心思路：空间语义与局部残差互补

> OpenCLIP-H空间分支接收正常RGB图像，学习内容、结构和高层视觉表征；RGPA取证分支在整图完成SRM-inspired高通残差提取后，对残差patch进行共享编码和高低残差双向软聚合。两个分支独立训练，再通过logit融合验证决策互补。

因此，**OpenCLIP-H空间单塔baseline是实现起点**。RGPA是可选的局部取证增量；如果融合没有稳定提升Final Score，最终仍提交表现更可靠的OpenCLIP-H空间单塔。

---



## 4. 模型选型



### 4.1 候选backbone


| 模型                    | 参数量(vision-only)                            | 说明                                      |
| --------------------- | ------------------------------------------- | --------------------------------------- |
| OpenCLIP ViT-L/14     | ~304M                                       | 空间塔备用降级方案                               |
| **OpenCLIP ViT-H/14** | **视觉塔约632M；完整图文模型约986M，须以实际checkpoint核算为准** | **主空间塔；checkpoint** `laion2b_s32b_b79k` |
| **RGPA轻量取证分支**         | **数万级；以实际加载统计为准**                      | **SRM-inspired残差 + patch共享编码 + 双向软聚合**         |


模型决策：**baseline直接使用OpenCLIP-H，不从双分支开始。**完成空间单塔后训练RGPA；取证分支有效后，再与OpenCLIP-H进行标准化加权logit融合。feature concat不属于最低交付要求。

### 4.2 参数量核算


| 组合方案                         | 参数量                                |
| ---------------------------- | ---------------------------------- |
| 单塔 L 级                       | ~300M                              |
| **OpenCLIP-H空间塔 + RGPA取证分支**    | **约632M加数万级取证分支；必须按实际加载模块统计，不计未加载的文本塔** |


---



## 5. 架构设计与实验路线



### 5.1 主线：两个独立分支训练 + 最终融合

```
Stage 1：OpenCLIP-H空间分支
输入图片 → clean保留或概率性单退化 → OpenCLIP ViT-H/14视觉塔 → 分类头 → spatial logit

Stage 2：轻量低层取证分支
同一输入图片 → 相同概率性单退化 → RGPA → forensic logit

Stage 3：双分支融合
standardized spatial logit + forensic logit → 加权logit融合 → 必要时feature concat → 最终校准概率
```

每个阶段都必须先形成可复现结果，再进入下一阶段：


| 实验                                             | 目的                                        |
| ---------------------------------------------- | ----------------------------------------- |
| Stage 1：OpenCLIP-H空间单塔                         | 在概率性单退化训练下选择最佳空间分支，并输出spatial logit       |
| Stage 2：RGPA取证单塔                               | 训练RGPA，输出forensic logit |
| Stage 3A：标准化加权logit融合                          | 低成本验证决策级互补，避免两个分支logit尺度不同造成假融合           |
| Stage 3B：LayerNorm/Projection + feature concat | 仅在互补得到证明后验证分类头之前的深层互补                     |


**答辩叙事逻辑**：在统一的概率性单退化训练分布下，先得到OpenCLIP-H空间分支，再训练RGPA取证分支，最后用logit融合验证语义与局部取证证据是否互补。只有能提升Final Score且不明显损害最差退化条件的模块才进入最终模型。

### 5.2 复杂融合：不纳入主线

Cross-Attention与feature concat会增加表示对齐和联合训练成本。当前主线只做标准化加权logit融合；只有融合已经显示稳定互补且仍有时间时，才考虑feature concat。

### 5.3 RGPA低层取证分支

空间分支与取证分支只共享概率性单退化和224×224几何预处理，不共享通道归一化：

```text
共享RGB图像
    ├→ OpenCLIP官方normalize → OpenCLIP-H → spatial logit
    └→ 整图SRM-inspired残差 → RGPA → forensic logit
```

Stage 2训练RGPA：将整图残差切成32×32 patch做共享编码，并用图内标准化残差能量进行高低双向软聚合。CIFAKE只用于跑通RGPA训练链路，正式效果使用SID-Set等高分辨率数据验证。

RGPA的架构、输入约束、聚合公式、实现注意事项和完整实验说明见：[取证分支方案_RGPA.md](./取证分支方案_RGPA.md)。

Stage 2统一报告Clean AUC、各类退化AUC、Robust AUC和Final Score，并保存与OpenCLIP-H相同Validation样本上的forensic logit。若RGPA无独立价值或最差退化条件明显恶化，则放弃低层取证分支。

---



## 6. 训练策略



### 6.1 冻结 + 轻量微调

- OpenCLIP空间backbone先完全冻结，仅训练线性分类头，建立linear-probe baseline
- baseline稳定后，只微调最后2-4层transformer block + 分类头，冻结其余部分
- 目的：避免在相对小规模训练集上过拟合到已见过的生成器，保留预训练泛化能力
- RGPA独立训练；Stage 3只融合其logit



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


**概率性单退化训练逻辑**：
官方Final test对每张图片只施加一种退化，不测试多种退化组合。因此，Stage 1空间分支和Stage 2低层取证分支统一采用以下训练分布：

$$
x' =
\begin{cases}
x, & r < p_{clean} \\
T_k(x), & r \ge p_{clean}
\end{cases}
\qquad r \sim U(0,1)
$$

其中 `T_k` 从已有的官方单退化集合中随机抽取，每个样本最多施加一种退化。`p_clean` 作为训练超参数通过Validation确定；两个分支使用相同的clean保留概率、退化采样概率与强度分布，真实图和AI图也使用完全相同的采样规则。

定义扰动概率：

$$
p_{degradation}=1-p_{clean}
$$

由于最终指标对 `AUC_clean` 和 `AUC_robust` 各赋予0.5权重，主训练配置采用：

$$
p_{degradation}=0.5,\qquad p_{clean}=0.5
$$

该设置用于对齐最终评测中clean与robust的相对重要性。训练损失与AUC并不完全等价，因此保留以下三组验证：

| 配置 | Clean概率 | 扰动概率 | 定位 |
| ---- | ----------: | ----------------: | ---- |
| P30 | 0.7 | 0.3 | 偏Clean对照 |
| **P50** | **0.5** | **0.5** | **主配置；与Final Score的0.5/0.5权重对齐** |
| P70 | 0.3 | 0.7 | 当前配置；偏Robust对照 |

三组实验保持其他设置一致，依据Validation Final Score选择扰动概率，并检查Clean AUC、Robust AUC和最差退化条件。每个样本最多施加一种退化；真实图和AI图必须采用相同的后处理与退化采样规则。



### 6.3 概率校准

- 从训练数据之外划出独立的 **calibration split**，且按生成器、原始图像来源和图像族隔离；
- OpenCLIP空间分支与RGPA取证分支的logit尺度可能不同，诊断融合前先用validation统计量标准化，再搜索少量固定权重；
- 融合 **logits**，不直接平均概率；
- 对最终融合logit在calibration split上重新做temperature scaling；
- calibration split只用于拟合校准参数，不用于模型选择、早停或超参数搜索；
- 报告ECE与Brier Score。



### 6.4 数据防泄漏

- 数据集划分应按**生成器来源、原始图像来源、图像族**为单位切分train/val/test
- 避免同一张真实图片、或它衍生出的生成/编辑版本，同时出现在train和test里——否则会得到虚高的、不可靠的评估分数

---



## 7. 数据集


| 数据集          | 规模（待核实）                                         | 特点                                             | 用途                                                                           |
| ------------ | ----------------------------------------------- | ---------------------------------------------- | ---------------------------------------------------------------------------- |
| **CIFAKE**   | 12万张，32×32低分辨率，单一生成器(SD1.4)                     | 分辨率极低，放大喂给ViT-L基本无法验证高频取证和高分辨率鲁棒性              | **仅用于Day1的冒烟测试**：验证Dataset/DataLoader能跑通、loss是否下降、输出格式是否正确。**不用于验证模型有效性**    |
| **SID_Set**  | 官方数据卡确认总数约30万，含Real/Full Synthetic/Tampered三类标签 | 社交媒体场景图片，逼真度高                                  | 正式训练主力数据（用Real+Full Synthetic两类）。**各类实际数量以下载后的metadata统计为准，不预先假设三类严格各10万均衡** |
| **WildFake** | 大规模，层级结构                                        | 覆盖GAN到diffusion多种生成器，来自Civitai/Midjourney等真实社区 | Cross-generator泛化测试（选训练时未见过的生成器子集）                                           |


---



## 8. 评估策略



### 8.1 鲁棒性评估矩阵（补齐官方参数全集）


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



## 9. Trade-off 讨论

- **Robustness vs Clean accuracy**：重度数据增强会让clean AUC略降，但robust AUC通常有提升，需用实际消融数据说话，而非预设"肯定值得"。
- **Generalization vs Specialization**：针对某一生成器调优的检测器在该生成器上分数高，但在新生成器上会明显下降，这是预期现象。
- **Complexity vs Feasibility**：RGPA与feature concat都会增加实现成本；本方案训练RGPA后只用logit融合判断互补。
- **Residual signal vs Robustness**：SRM-inspired残差可能在clean图片上明显，却会被JPEG、Blur、Resize和Noise改变；RGPA必须在官方退化矩阵下验证，不能只凭clean AUC保留。

---



## 10. Stage 1：OpenCLIP-H空间分支



### 10.1 实验目标

使用小规模CIFAKE数据验证OpenCLIP-H单塔在概率性单退化训练下的训练、推理与评测链路是否可靠，并比较冻结特征与局部微调的差异。本阶段输出最佳空间分支及其逐样本spatial logit，为Stage 3融合提供固定输入。

CIFAKE原始图像分辨率仅为32×32；放大至OpenCLIP输入尺寸不会恢复已经缺失的高频细节，因此本阶段分数不作为正式方案上限。

### 10.2 实验设置



#### 数据划分


| 数据集    | 划分         | Real（0） | AI（1） | 总数    |
| ------ | ---------- | ------- | ----- | ----- |
| CIFAKE | Train      | 4,000   | 4,000 | 8,000 |
| CIFAKE | Validation | 1,000   | 1,000 | 2,000 |


- 固定随机种子并保存样本ID；
- Train与Validation不得包含重复或近重复图片；
- S1、S2、可选S3以及后续低层分支验证使用同一划分；
- 当前Validation结果属于阶段性验证结果，不表述为独立测试集结果。



#### 实验指标

直接调用项目中已经定义的Robust评测函数，不在本节重复描述退化类型与参数。统一报告：

$$
\mathrm{FinalScore}
=0.5\,\mathrm{AUC}_{clean}
+0.5\,\mathrm{AUC}_{robust}
$$

### 10.3 OpenCLIP单塔baseline



#### 10.3.1 实验S1：Linear probe

- 冻结整个OpenCLIP视觉backbone；
- 只训练线性二分类头；
- 使用第6.2节定义的clean保留 + 概率性单退化训练；
- 真实图与AI图使用相同的退化采样规则；
- 根据Validation Final Score选择checkpoint。

**实验目的**：验证OpenCLIP原始预训练特征是否已经包含真实图与AI图的可分信息，并建立低成本、变量受控的空间单塔基准。

#### 10.3.2 实验S2：局部微调最后2个Block

- 解冻OpenCLIP最后2个Transformer block；
- 其余backbone参数保持冻结；
- 训练解冻部分与分类头；
- 使用与S1完全相同的数据划分和基础预处理；
- 继续使用与S1完全相同的概率性单退化训练分布。

**实验目的**：验证少量任务适配能否提升检测性能，同时观察Robust AUC是否因适配而下降。

#### 10.3.3 实验S3：局部微调最后4个Block（可选）

仅当S2相对S1产生稳定收益时启动：

- 解冻最后4个Transformer block；
- 其余设置与S2保持一致；
- 将S3作为独立实验记录，不把“解冻2～4层”混写为同一配置。



### 10.4 结果记录


| 实验     | 训练范围             | Clean AUC | JPEG | Blur | Resize | Noise | Color | Crop | Robust AUC | Final Score |
| ------ | ---------------- | --------- | ---- | ---- | ------ | ----- | ----- | ---- | ---------- | ----------- |
| S1     | Linear head      |           |      |      |        |       |       |      |            |             |
| S2     | Head + 最后2个Block |           |      |      |        |       |       |      |            |             |
| S3（可选） | Head + 最后4个Block |           |      |      |        |       |       |      |            |             |


除汇总表外，保存每个Validation样本的ID、真实标签、logit、概率与评测条件，供后续OpenCLIP/RGPA互补性分析使用。

### 10.5 Stage 1退出条件

- S1能够完整训练、保存并恢复checkpoint；
- Clean与已有Robust评测函数均能正常运行；
- S1与S2完成相同数据划分下的公平比较；
- 训练结果可复现，且未发现明显数据泄漏；
- 输出完整结果表与逐样本预测；
- 确定进入正式数据实验时采用linear probe、最后2层微调或可选最后4层微调。

---



## 11. 分阶段执行计划


| 优先级    | 阶段       | 必须完成的任务                                                                      | 数据集与退出条件                                                                  |
| ------ | -------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **P0** | Stage 1  | 输入图像按概率保持clean或施加一种退化，训练OpenCLIP-H空间分支；比较S1 linear probe、S2最后2层微调和可选S3最后4层微调 | CIFAKE先验证流程，再迁移SID-Set正式训练。复用已有Robust评测函数，输出完整结果表、逐样本spatial logit和最佳空间分支 |
| **P1** | Stage 2  | 使用与Stage 1相同的数据划分和概率性单退化分布，训练RGPA           | CIFAKE只跑通RGPA链路，SID-Set正式评估。输出相同样本上的forensic logit，完成格式、压缩与分辨率捷径检查   |
| **P2** | Stage 3A | 冻结前两阶段候选，标准化两类logit并验证加权logit融合                                              | 统计预测相关性、错误重合、OpenCLIP错误样本纠正率和各条件AUC；只有Final Score稳定提升或错误分歧明显才继续复杂融合       |
| **P2** | Stage 3B | 对最佳空间与低层特征分别LayerNorm/Projection后进行feature concat，并与最佳logit融合公平比较            | WildFake未见生成器子集验证；只有提升Final Score且不显著损害最差退化条件才作为最终模型                      |
| **P2** | 最终校准     | 对最终候选使用独立calibration split做temperature scaling；报告ECE/Brier和FP/FN分析           | calibration split不参与模型选择；输出最终鲁棒性表格和错误分析                                   |


最终路线固定为：Stage 1训练OpenCLIP-H空间分支，Stage 2训练RGPA，Stage 3融合两个独立分支。OpenCLIP-H空间单塔是最低交付版本；标准化加权logit融合负责验证决策互补，feature concat仅作为时间充裕时的延伸。只有带来可复现收益的增量才进入最终模型。

---



## 12. 参考文献

- Raising the bar of AI-generated image detection with CLIP (Cozzolino et al., CVPR 2024)
- SAFE (KDD 2025)：模型预处理阶段crop优于强制缩放的insight（不替代评估阶段的Resize退化测试）
- DDA (NeurIPS 2025)：真实图与合成图后处理对齐、避免频率偏置的insight
- AIDE / A Sanity Check for AI-generated Image Detection (ICLR 2025)：DCT频率评分选择局部patch、SRM残差取证与OpenCLIP全局语义融合
