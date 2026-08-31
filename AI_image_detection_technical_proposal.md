# AI-Generated Image Detection — Technical Proposal (v14)

> TikTok TechJam Track 5: build a detector that separates AI-generated images from real photographs, stays robust under real-world post-processing (compression, resize, blur, color shift, …), and generalizes to generators not seen in training.

This document covers **two independently trained branches** and their fusion:

- **Spatial branch** — OpenCLIP ViT-H/14, global semantics and structure.
- **Forensic branch** — Residual-Guided Patch Aggregation (RGPA), local pixel-level traces from a frozen high-pass residual.

---

## 1. Problem

Generative models can produce photorealistic images at very low cost, which creates platform risk around misinformation, identity misuse, e-commerce fraud, and copyright.

Two difficulties dominate:

- **Generalization.** A new generator (diffusion, GAN, next month’s model) leaves a fingerprint that need not match the training set.
- **Robustness.** Images on a real sharing path are compressed, cropped, blurred, and recolored. Those operations destroy part of the signal a detector relies on.

---

## 2. Constraints

| Constraint | Detail |
| --- | --- |
| Time | Delivery in a few days (hackathon pace) |
| Local compute | Sufficient |
| Parameter cap | Combined models **< 2B** |
| Engineering rule | Finish the OpenCLIP-H spatial baseline first, then train a light RGPA forensic branch, then prove complementarity with logit fusion. Keep an increment only if it raises the official Final Score. |

---

## 3. Core idea: spatial semantics and local residuals

The **spatial branch** takes a normal RGB image and learns content, structure, and high-level visual representation. The **forensic branch** extracts an SRM-inspired high-pass residual on the **full image**, encodes residual patches with a shared CNN, and aggregates high- and low-residual patches with bidirectional soft weights.

The two branches are trained separately. Stage 3 tests whether their logits are complementary.

**The OpenCLIP-H spatial tower is the starting point and the fallback.** RGPA is an optional local-forensics increment. If fusion does not stably improve Final Score, the submitted model stays spatial-only.

```text
shared RGB image
    ├→ official OpenCLIP normalize → OpenCLIP-H → spatial logit
    └→ full-image SRM-inspired residual → RGPA → forensic logit
                              ↓
         standardized weighted logit fusion → calibrated P(AI)
```

The branches share probabilistic single-degradation training and 224×224 geometry. They do **not** share channel normalization.

---

## 4. Model selection

### 4.1 Candidate backbones

| Model | Vision parameters | Role |
| --- | --- | --- |
| OpenCLIP ViT-L/14 | ~304M | Spatial fallback if H does not fit |
| **OpenCLIP ViT-H/14** | **Vision tower ~632M; full CLIP ~986M — count the loaded checkpoint** | **Primary spatial tower.** Checkpoint `laion2b_s32b_b79k` |
| **RGPA forensic branch** | **Tens of thousands; count loaded modules** | **SRM-inspired residual + shared patch encoder + bidirectional soft aggregation** |

Decision: **start from OpenCLIP-H, not from a two-branch system.** Train RGPA after the spatial tower is in place. Fuse standardized logits only after the forensic branch has independent value. Feature concat is not a minimum deliverable.

### 4.2 Parameter budget

| Combination | Parameters |
| --- | --- |
| Single L-scale tower | ~300M |
| **OpenCLIP-H spatial + RGPA forensic** | **~632M plus a tiny forensic head. Count loaded modules only; do not count an unused text tower.** |

---

## 5. Spatial branch (OpenCLIP-H)

The spatial branch is a global semantic detector. It is Stage 1 of the roadmap and the lowest-risk shippable model.

### 5.1 Data flow

```text
RGB image
  ↓
keep clean, or apply one official degradation
  ↓
OpenCLIP geometry (Resize 224 + CenterCrop 224)
  ↓
official OpenCLIP channel normalize
  ↓
ViT-H/14 vision tower
  ↓
classification head
  ↓
spatial logit
```

### 5.2 Why this backbone

OpenCLIP ViT-H/14 is pretrained at web scale on natural images. Frozen or lightly unfrozen CLIP features already separate many real vs. synthetic pairs, which is why a linear probe is the first experiment. The spatial tower is **not** designed to read pixel-level forensic traces; those are the forensic branch’s job.

### 5.3 Freeze and light unfreeze

- Freeze the vision backbone first. Train only a linear head (**linear probe**).
- Once that baseline is stable, unfreeze the last **2** transformer blocks plus the head. Keep the rest frozen.
- Optionally unfreeze the last **4** blocks, but only if unfreeze-2 beats the probe with a stable Final Score gain. Record unfreeze-2 and unfreeze-4 as separate runs.
- Goal: avoid overfitting a relatively small train set to seen generators, and keep pretrained generalization.

### 5.4 Stage 1 experiments (CIFAKE smoke, then SID-Set / WildFake)

CIFAKE is 32×32. Upsampling to 224×224 does not restore missing high-frequency detail, so CIFAKE scores are **not** an upper bound for the real system. Use CIFAKE only to prove the train / eval / checkpoint path, then move to SID-Set.

#### Split (CIFAKE smoke)

| Dataset | Split | Real (0) | AI (1) | Total |
| --- | --- | ---: | ---: | ---: |
| CIFAKE | Train | 4,000 | 4,000 | 8,000 |
| CIFAKE | Validation | 1,000 | 1,000 | 2,000 |

Fix the seed and persist sample IDs. Train and validation must not share duplicates or near-duplicates. Later forensic and fusion stages reuse this split during smoke tests.

#### S1 — Linear probe

- Freeze the entire OpenCLIP vision backbone.
- Train only the linear binary head.
- Use the shared clean-keep + probabilistic single-degradation recipe in §7.2.
- Real and AI images use the same degradation sampler.
- Pick the checkpoint by validation Final Score.

Purpose: test whether pretrained OpenCLIP features already contain a real vs. AI margin, at low cost and with few free variables.

#### S2 — Unfreeze last 2 blocks

- Unfreeze the last 2 transformer blocks.
- Keep the rest of the backbone frozen.
- Train the unfrozen blocks and the head.
- Same split, same geometry, same degradation distribution as S1.

Purpose: test whether a small task adaptation helps, and whether Robust AUC drops because of that adaptation.

#### S3 — Unfreeze last 4 blocks (optional)

Start only if S2 beats S1 stably. Same recipe as S2, recorded as its own experiment. Do not collapse “unfreeze 2–4” into one config.

#### Result log

| Run | Trainable scope | Clean AUC | JPEG | Blur | Resize | Noise | Color | Crop | Robust AUC | Final Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | Linear head |  |  |  |  |  |  |  |  |  |
| S2 | Head + last 2 blocks |  |  |  |  |  |  |  |  |  |
| S3 (optional) | Head + last 4 blocks |  |  |  |  |  |  |  |  |  |

Also save, for every validation sample: ID, label, logit, probability, and evaluation condition. Those per-sample spatial logits are the frozen input to Stage 3 fusion.

#### Spatial-branch exit criteria

- S1 trains, saves, and reloads a checkpoint.
- Clean and the existing robustness evaluator both run.
- S1 and S2 are compared on the same split.
- Results are reproducible, with no obvious leakage.
- The team decides whether official-data training uses the probe, last-2 unfreeze, or optional last-4 unfreeze.

---

## 6. Forensic branch (RGPA)

RGPA is Stage 2. It answers: after a **fixed full-image high-pass residual**, can shared patch encoding plus bidirectional high/low residual aggregation give a **reproducible gain that is independent of OpenCLIP-H**?

```text
OpenCLIP-H spatial tower
    ↓
RGPA forensic tower
    ↓
OpenCLIP-H + RGPA standardized weighted logit fusion
```

### 6.1 Input

- Input is a **224×224 RGB** image.
- Patch size is **32×32**, so **7×7 = 49** patches.
- Spatial and forensic branches share probabilistic single-degradation training and the same geometry.
- They do **not** share channel normalization: OpenCLIP uses the official CLIP normalize; RGPA uses pixel-scale RGB.

Fixed 224×224 is an engineering choice under the time budget. It can discard forensic traces that exist only at native resolution. Native-resolution or multi-scale patches are **not** on the current mainline.

### 6.2 Data flow

```text
RGB image
  ↓
probabilistic single degradation
  ↓
one-shot geometry to 224×224
  ↓
full-image SRM-inspired frozen high-pass residual
  ↓
split the residual into 49 patches of 32×32
  ↓
shared light CNN, one code per patch
  ↓
in-image standardized residual energy
  ↓
bidirectional high / low residual soft aggregation
  ↓
forensic logit
```

The high-pass filter **must run on the full image before the split**. Per-patch convolution padding would invent artificial patch borders.

### 6.3 Bidirectional soft aggregation

For patch \(i\), compute residual energy and standardize it **inside the same image**:

$$
a_i=E_{\mathrm{residual},i}
\qquad
\hat{a}_i=\frac{a_i-\mu_a}{\sigma_a+\epsilon}
$$

High- and low-residual weights:

$$
w_i^{high}=\mathrm{softmax}\left(\frac{\hat{a}_i}{\tau}\right)
\qquad
w_i^{low}=\mathrm{softmax}\left(-\frac{\hat{a}_i}{\tau}\right)
$$

Aggregate patch features:

$$
z^{high}=\sum_i w_i^{high}z_i
\qquad
z^{low}=\sum_i w_i^{low}z_i
$$

$$
z_{\mathrm{forensic}}
=\mathrm{concat}\left(z^{high},z^{low}\right)
$$

In-image standardization makes the weights describe **relative** residual strength in the current photo, so absolute energy scales that change with image and degradation matter less. When patch residuals are nearly uniform, high and low aggregates can look similar. That is a mechanism boundary: log the weight gap when interpreting results.

Soft weights keep **all** patches; nothing is hard-dropped.

### 6.4 Training

RGPA is trained from scratch. The SRM-inspired high-pass kernel stays **frozen**. Training uses the **same split** and **same probabilistic single-degradation distribution** as the spatial branch. Input is 224×224 pixel-scale RGB.

CIFAKE only proves that the RGPA train loop runs. Official forensic numbers need SID-Set (or similar) at a resolution that still has local traces, after format / compression / resolution alignment so the branch cannot cheat on source shortcuts.

### 6.5 Forensic experiments and exit

| Step | Experiment | Exit |
| --- | --- | --- |
| 1 | RGPA alone | Report Clean AUC, per-degradation AUC, Robust AUC, and Final Score on the **same validation samples** as OpenCLIP-H. Keep the branch only if it has independent value and the worst degradation does not collapse. |
| 2 | OpenCLIP-H + RGPA logit fusion | Standardize both logits, search a few fixed weights. Keep fusion only if Final Score rises stably. |

If RGPA has no independent value, skip fusion. If fusion has no gain, ship the OpenCLIP-H spatial tower.

Stage 2 must save per-sample forensic logits aligned with the spatial validation IDs.

### 6.6 Risks specific to the forensic branch

- JPEG, blur, resize, and noise **directly** change a high-pass residual. Robust AUC on the official 15-condition grid is mandatory; Clean AUC alone is not enough to keep RGPA.
- Real and AI images must share the same codec, geometry, and degradation sampler.
- Fixed 224×224 can drop native-resolution traces.
- When high and low weights are close, bidirectional aggregation can collapse into near-duplicate features.

### 6.7 Method statement

> Inspired by AIDE’s local-forensics plus global-semantics fusion, RGPA first extracts an SRM-inspired frozen high-pass residual on the full image, encodes residual patches with a shared CNN, and aggregates them with bidirectional soft weights from in-image standardized residual energy. All patches remain in the representation. Train RGPA first, then test complementarity with OpenCLIP-H by logit fusion.

---

## 7. Shared training

Both branches use the same split, the same official single-degradation family, and the same clean-keep probability. Only the encoder and the channel normalize differ.

### 7.1 Official evaluation transforms

These must all appear at eval. Do not drop any cell because a different crop or resize is preferred at train time.

| Transform | Parameters | Real-world analogue |
| --- | --- | --- |
| JPEG | quality = 90 / 70 / 50 / 30 | Social re-encode, messaging |
| Gaussian blur | σ = 0.5 / 1.0 / 2.0 | Defocus, screenshot smoothing |
| **Resize (down then up)** | **0.5× / 0.25×, then back to original size** | **Thumbnails, CDN resize — official; do not replace with crop** |
| Gaussian noise | σ = 0.02 / 0.05 / 0.10 | Low-light sensor grain |
| Color jitter | brightness / contrast / saturation ±20% | Filter apps, auto-enhance |
| Center crop | 80% | Avatar crop, reframing |

### 7.2 Probabilistic single-degradation training

The official Final test applies **one** degradation per image, never a stack. Both branches therefore train with:

$$
x' =
\begin{cases}
x, & r < p_{clean} \\
T_k(x), & r \ge p_{clean}
\end{cases}
\qquad r \sim U(0,1)
$$

\(T_k\) is drawn from the official single-degradation set. Each sample gets at most one transform. \(p_{clean}\) is a hyperparameter chosen on validation. Real and AI images use the same sampler.

$$
p_{degradation}=1-p_{clean}
$$

The official score weights clean and robust equally, so the main training config is:

$$
p_{degradation}=0.5,\qquad p_{clean}=0.5
$$

Training loss is not AUC, so keep three checks:

| Config | Clean probability | Degradation probability | Role |
| --- | ---: | ---: | --- |
| P30 | 0.7 | 0.3 | Clean-leaning control |
| **P50** | **0.5** | **0.5** | **Main config; matches the 0.5 / 0.5 Final Score** |
| P70 | 0.3 | 0.7 | Robust-leaning control |

Hold every other setting fixed. Choose \(p_{clean}\) by validation Final Score, and inspect Clean AUC, Robust AUC, and the worst cell.

### 7.3 Probability calibration

- Hold out a **calibration split** that is not used for training. Isolate it by generator, source photo, and image family.
- Spatial and forensic logits can live on different scales. Before fusion, standardize with validation statistics, then search a few **fixed** weights.
- Fuse **logits**, not probabilities.
- After fusion, fit temperature scaling on the calibration split only.
- The calibration split is not used for model selection, early stopping, or hyperparameter search.
- Report ECE and Brier score.

### 7.4 Leakage control

Split by **generator, source photograph, and image family**. A real photo and any synthetic or edited descendant of it must not appear on both sides of train / test. Otherwise AUC is inflated and not trustworthy.

---

## 8. Fusion (Stage 3)

### 8.1 Mainline: standardized weighted logits

```text
Stage 1: OpenCLIP-H spatial branch → spatial logit
Stage 2: RGPA forensic branch     → forensic logit
Stage 3: standardized spatial logit + forensic logit
         → weighted logit fusion
         → optional feature concat
         → calibrated P(AI)
```

Each stage must be reproducible before the next starts.

| Experiment | Purpose |
| --- | --- |
| Stage 1: OpenCLIP-H spatial tower | Pick the best spatial branch under probabilistic single-degradation training; export spatial logits |
| Stage 2: RGPA forensic tower | Train RGPA; export forensic logits on the same samples |
| Stage 3A: standardized weighted logit fusion | Cheap test of decision-level complementarity; avoids a fake fusion caused by mismatched logit scales |
| Stage 3B: LayerNorm / projection + feature concat | Only after 3A shows complementarity; tests deeper fusion before the head |

**Defense narrative.** Under one shared degradation distribution, first obtain the OpenCLIP-H spatial branch, then train RGPA, then ask whether semantic evidence and local forensic evidence complement each other. A module enters the final model only if it raises Final Score and does not clearly hurt the worst degradation.

### 8.2 Not on the mainline

Cross-attention and feature concat add alignment and joint-training cost. The mainline is standardized weighted logit fusion. Feature concat is considered only if fusion already looks stably complementary and time remains.

---

## 9. Datasets

| Dataset | Scale (verify on disk) | Character | Use |
| --- | --- | --- | --- |
| **CIFAKE** | 120k, 32×32, one generator (SD 1.4) | Too small to test high-frequency forensics or high-res robustness | **Day-1 smoke only**: loader, falling loss, output schema. **Not** a validity test |
| **SID_Set** | Official card ~300k; Real / Full Synthetic / Tampered | Social-looking, high fidelity | Main official training (Real + Full Synthetic). Count classes from metadata; do not assume three equal 100k buckets |
| **WildFake** | Large, hierarchical | GAN → diffusion; Civitai / Midjourney and similar | Cross-generator generalization (hold out generators unseen in training) |

---

## 10. Evaluation

### 10.1 Robustness grid (full official parameter set)

| Condition | Note |
| --- | --- |
| Clean | Unprocessed image |
| JPEG q=90 / 70 / 50 / 30 | All four, not only q=30 |
| Blur σ=0.5 / 1.0 / 2.0 | All three |
| Resize 0.5× / 0.25× (down then up) | Official required cell |
| Gaussian noise σ=0.02 / 0.05 / 0.10 | All three |
| Color jitter ±20% | — |
| Crop 80% | — |
| Unseen generator | WildFake subset held out of training |

### 10.2 Metrics

- **Primary:** ROC AUC (threshold-free, robust to class imbalance).
- Calibration: ECE, Brier score (§7.3).

### 10.3 Official Final Score

```
Final Score = 0.50 × AUC_clean + 0.50 × AUC_robust
```

- `AUC_clean`: ROC AUC on the clean test set.
- `AUC_robust`: ROC AUC on the official robustness set.
- Each term is 50%.

The team may also report per-condition AUC, mean-degradation AUC, worst-cell AUC, ECE, and Brier for diagnosis. Those extras must not replace or be renamed into the official Final Score.

### 10.4 Single vs stacked degradations

- **Official grid:** one degradation per sample, covering every parameter in §10.1.
- **Optional research grid:** JPEG+crop and similar stacks, labeled as a team extension.
- Model selection uses official Final Score first, then per-cell and worst-cell AUC, so an average cannot hide a collapse on one transform.

---

## 11. Trade-offs

- **Robustness vs clean accuracy.** Heavy augmentation can lower Clean AUC a little while raising Robust AUC. Use ablations, not a prior that “it is always worth it.”
- **Generalization vs specialization.** A detector tuned to one generator scores high there and drops on a new one. That is expected.
- **Complexity vs feasibility.** RGPA and feature concat both cost implementation time. This proposal trains RGPA, then judges complementarity with logit fusion only.
- **Residual signal vs robustness.** An SRM-inspired residual can be obvious on clean pixels and then change under JPEG, blur, resize, and noise. RGPA must survive the official grid; Clean AUC is not a keep criterion.

---

## 12. Phased plan

| Priority | Stage | Must finish | Data and exit |
| --- | --- | --- | --- |
| **P0** | Stage 1 — **spatial** | Keep the image clean with probability \(p_{clean}\), else apply one official degradation. Train OpenCLIP-H. Compare S1 probe, S2 last-2 unfreeze, optional S3 last-4 | CIFAKE for the pipeline, then SID-Set. Reuse the robustness evaluator. Emit the full table, per-sample spatial logits, and the chosen spatial tower |
| **P1** | Stage 2 — **forensic** | Same split and same degradation distribution. Train RGPA | CIFAKE only to run the loop; SID-Set for real numbers. Emit forensic logits on the same samples. Check format / compression / resolution shortcuts |
| **P2** | Stage 3A | Freeze both candidates. Standardize logits and test weighted fusion | Correlation, error overlap, OpenCLIP-error correction rate, per-cell AUC. Continue to deeper fusion only if Final Score rises or errors clearly diverge |
| **P2** | Stage 3B | LayerNorm / project the best spatial and forensic features, concat, compare fairly with best logit fusion | WildFake unseen-generator subset. Keep as the final model only if Final Score rises and the worst cell does not collapse |
| **P2** | Final calibration | Temperature-scale the final candidate on a held-out calibration split. Report ECE / Brier and FP / FN | Calibration split is not used for selection. Emit the final robustness table and error analysis |

Fixed route: Stage 1 trains the OpenCLIP-H **spatial** branch, Stage 2 trains the RGPA **forensic** branch, Stage 3 fuses the two. The spatial tower is the minimum delivery. Standardized weighted logit fusion is the complementarity test. Feature concat is an extension if time remains. Only a reproducible gain enters the final model.

---

## 13. References

- Raising the bar of AI-generated image detection with CLIP (Cozzolino et al., CVPR 2024)
- SAFE (KDD 2025): crop-before-resize in preprocessing (does **not** replace the official Resize degradation at eval)
- DDA (NeurIPS 2025): align real vs. synthetic post-processing to avoid frequency shortcuts
- AIDE / A Sanity Check for AI-generated Image Detection (ICLR 2025): DCT patch selection, SRM residual forensics, fusion with OpenCLIP global semantics
