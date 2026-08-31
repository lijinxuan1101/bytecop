# 5. Error Analysis Note

Submitted detector: OpenCLIP ViT-H/14 spatial model. Errors are counted on the 5,000-image WildFake val slice (2,500 real / 2,500 AI) at threshold 0.5. Fake = AI-generated. A **false positive** flags a real photo as AI; a **false negative** lets an AI image through as real.

| | Clean |
| --- | ---: |
| False positives | 6 |
| False negatives | 298 |
| Precision @ 0.5 | 0.997 |
| Recall @ 0.5 | 0.881 |

## False positives — rare, high-confidence when they happen

On clean images the model almost never accuses a real photo: **6 FP out of 2,500 reals (0.24%)**. Those six are not borderline — the strongest has P(AI) = 0.91. CLIP’s global semantics can still read a real scene as generated when lighting, texture, or composition looks synthetic.

Heavy resharing inflates this class. Gaussian noise at 10% raises FP to **64**; blur σ=2.0 raises it to **43**. Additive grain and strong blur are the transforms that make real photos look least photographic to the tower.

## False negatives — the dominant error

**298 of 2,500 AI images** are missed on clean (11.9%). The worst misses are not near the threshold: several score P(AI) ≈ 0. Photorealistic generators (especially later diffusion / Midjourney-style images in WildFake) land on the same semantic manifold as real photos, and a spatial CLIP tower has little local forensic signal to fall back on.

Noise and blur also drive FN up (437 and 429). That matches the official grid: Noise 10% is the worst cell (AUC 0.9707). The model still ranks well; at 0.5 it simply stays on the “call it real” side of the logit.

## Where errors grow

Same 5k slice, threshold 0.5.

| Condition | FP | FN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| Clean | 6 | 298 | 0.9973 | 0.8808 |
| Color | 11 | 302 | 0.9950 | 0.8792 |
| JPEG 30 | 10 | 360 | 0.9953 | 0.8560 |
| Crop 80% | 16 | 352 | 0.9926 | 0.8592 |
| Resize ¼ | 21 | 365 | 0.9903 | 0.8540 |
| Blur 2.0 | 43 | 429 | 0.9797 | 0.8284 |
| **Noise 10%** | **64** | **437** | **0.9699** | **0.8252** |

JPEG / color / crop barely move FP. Noise and strong blur move both FP and FN.

## Trade-offs in the proposed approach

**Threshold 0.5 is conservative on purpose.** Precision 0.997 vs recall 0.881. For a platform, accusing a real photo is worse than missing some AI. Lowering the threshold would cut FN and raise FP; we did not retune it on this val slice.

- **Spatial-only vs a forensic second tower.** RGPA matches Spatial on SID, then drops to official 0.906 on WildFake. Gated fusion still mixes in ~25% RGPA and loses most on noise. The submitted model is Spatial alone — fewer FPs under noise than the fused system, at the cost of no local-residual fallback on photoreal fakes.
- **Ranking vs operating point.** Official AUC 0.9924 looks almost saturated; Acc 0.9315 does not, because FN dominate at 0.5. AUC describes order; the product call is the 0.5 cut.
- **Robustness training vs extra eval enhancement.** Single-degradation training (30% clean) stays. MBE at eval hurts blur and crop and is not used. We accept a small clean/robust AUC gap (0.0084) rather than a preprocess the tower never saw.
- **Capacity vs speed.** Truncating ViT-H to 8 layers keeps clean AUC ~0.99 but collapses robust AUC to ~0.79. The 632M vision tower is what holds the grid; a smaller forensic substitute only sped the pipeline 22% and lost 0.087 AUC.
