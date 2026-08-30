# experiments/

两塔分开训，结果写在 `runs/`，不写在这个目录。

| 目录 | 模型 | 启动 |
|---|---|---|
| [`spatial_tower/`](spatial_tower/README.md) | OpenCLIP ViT-H/14 | `bash experiments/spatial_tower/run.sh` |
| [`forensic_tower/`](forensic_tower/run.sh) | RGPA | `bash experiments/forensic_tower/run.sh` |

共用代码：`common/`（DDP、checkpoint、scheduler、AMP）。配置在各塔的 `configs/*.yaml`。
