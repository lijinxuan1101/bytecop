# Stage 1 — Results Summary

Every experiment was trained on **CIFAKE (10,000 sampled)** and evaluated on the same held-out `test/` split using the official 15-condition robustness matrix.

## Summary

| Experiment | AUC_clean | AUC_robust | Final Score | Worst Condition AUC |
|---|---:|---:|---:|---:|
| `s1_linear_probe` | 0.9958 | 0.9542 | 0.9750 | 0.8315 |
| `s2_unfreeze2` | 0.9982 | 0.9561 | 0.9772 | 0.8353 |
| `s3_unfreeze4` | 0.9988 | 0.9449 | 0.9719 | 0.8117 |

## Per-Condition AUC

| Condition | `s1_linear_probe` | `s2_unfreeze2` | `s3_unfreeze4` |
|---|---:|---:|---:|
| clean | 0.9958 | 0.9982 | 0.9988 |
| jpeg_q90 | 0.9951 | 0.9978 | 0.9985 |
| jpeg_q70 | 0.9952 | 0.9977 | 0.9981 |
| jpeg_q50 | 0.9837 | 0.9872 | 0.9858 |
| jpeg_q30 | 0.9684 | 0.9751 | 0.9689 |
| blur_s0.5 | 0.9876 | 0.9933 | 0.9917 |
| blur_s1.0 | 0.9590 | 0.9682 | 0.9574 |
| blur_s2.0 | 0.8796 | 0.8868 | 0.8578 |
| resize_0.5 | 0.9561 | 0.9629 | 0.9486 |
| resize_0.25 | 0.8315 | 0.8353 | 0.8117 |
| noise_s0.02 | 0.9793 | 0.9748 | 0.9647 |
| noise_s0.05 | 0.9527 | 0.9404 | 0.9142 |
| noise_s0.10 | 0.9005 | 0.8872 | 0.8628 |
| color_jitter | 0.9934 | 0.9957 | 0.9963 |
| center_crop_80 | 0.9773 | 0.9828 | 0.9722 |

## Config Snapshot

Each run's exact config is copied to `runs/stage1/<name>/config.yaml`.
