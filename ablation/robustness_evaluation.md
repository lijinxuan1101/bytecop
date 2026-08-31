# 4. Robustness Evaluation Summary

Submitted model: OpenCLIP ViT-H/14 spatial tower. Protocol: official 15-condition grid on a 5,000-image WildFake val slice (50/50 real / AI). Acc / F1 use threshold 0.5. **Robust** is the mean of the 14 transformed conditions. **Official** = 0.50 × clean + 0.50 × robust.

## Clean vs transformed

| | AUC | AP | Acc | F1 | EER |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 0.9966 | 0.9968 | 0.9392 | 0.9354 | 0.0298 |
| Robust (14 transforms) | 0.9882 | 0.9893 | 0.9238 | 0.9183 | 0.0579 |
| **Official** | **0.9924** | **0.9931** | **0.9315** | **0.9269** | **0.0438** |

AUC drops **0.0084** from clean to the mean of transformed images. Ranking (AUC / AP) stays high; Acc / F1 drop more because the 0.5 threshold is conservative (clean precision 0.997, recall 0.881).

```
AUC  0.88                         1.00
     |                              |
Clean| ████████████████████████████ | 0.9966
Robust| ███████████████████████████ | 0.9882
Official| ███████████████████████████ | 0.9924
```

Bar length is linear on [0.88, 1.00].

## Per transform (vs clean)

Each row is one official degradation, applied alone. Δ is versus clean AUC.

| Condition | AUC | Δ vs clean | Acc |
| --- | ---: | ---: | ---: |
| Clean | 0.9966 | — | 0.9392 |
| JPEG 90 | 0.9964 | −0.0002 | 0.9356 |
| JPEG 70 | 0.9955 | −0.0011 | 0.9258 |
| JPEG 50 | 0.9938 | −0.0028 | 0.9306 |
| JPEG 30 | 0.9902 | −0.0064 | 0.9260 |
| Blur 0.5 | 0.9958 | −0.0008 | 0.9370 |
| Blur 1.0 | 0.9871 | −0.0095 | 0.9242 |
| Blur 2.0 | 0.9766 | −0.0200 | 0.9056 |
| Resize ½ | 0.9939 | −0.0027 | 0.9352 |
| Resize ¼ | 0.9871 | −0.0095 | 0.9228 |
| Noise 2% | 0.9811 | −0.0155 | 0.9186 |
| Noise 5% | 0.9816 | −0.0150 | 0.9080 |
| **Noise 10% (worst)** | **0.9707** | **−0.0259** | **0.8998** |
| Color | 0.9938 | −0.0028 | 0.9374 |
| Crop 80% | 0.9914 | −0.0052 | 0.9264 |

Mild JPEG, light blur, color jitter, and 80% crop stay within ~0.005 AUC of clean. The grid is hardest on Gaussian noise and strong blur. No cell collapses: worst-condition AUC is 0.9707.
