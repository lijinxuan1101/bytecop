# Spatial 推理后段

交付模型只保留 **OpenCLIP-H spatial tower**（WildFake `best.pt`）。RGPA / 门控融合留下做消融，不进 demo。

可视化前端只依赖这一层：Python 直接 import，或 HTTP。

## 默认权重

`runs/spatial_tower/spatial_tower_wildfake/best.pt`（step 12000，和 `runs/fusion/fusion_wildfake/clip_h.pt` 是同一份硬链）

加载时 **不读** 3.9G DFN-5B，只建 ViT-H 结构再灌 `best.pt`。

## 给可视化用的字段

| 字段 | 含义 |
| --- | --- |
| `image_path` | 路径；上传图时是文件名 |
| `pred` | P(AI)，0–1，官方交付字段 |
| `logit` | 温度缩放前的原始 logit，方便画分布 |
| `label` | `fake` if `pred >= 0.5` else `real` |

官方脚本仍只写 `image_path` + `pred`。

## Python（前端进程里直接调）

```python
from serve.spatial_backend import SpatialDetector

det = SpatialDetector()                    # 第一次会占一张 GPU
det.score_path("photo.jpg")
det.score_dir("demo_images/")
det.score_pil(pil_image, path="upload.png")
```

`SpatialDetector` 建一次、反复打分。不要每个请求重新 `CLIPTower(...)`。

## HTTP

```bash
source ~/techjam/venv/bin/activate
pip install fastapi uvicorn python-multipart   # 只需装一次
CUDA_VISIBLE_DEVICES=1 uvicorn serve.app:app --host 0.0.0.0 --port 8008
```

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/health` | 探活 + ckpt / device |
| GET | `/v1/model` | 同上 |
| POST | `/v1/score` | `multipart/form-data` 字段名 `file`，单张图 |
| POST | `/v1/score-dir` | JSON `{"directory": "/abs/path"}` |

单张上传返回：

```json
{"image_path": "shot.png", "pred": 0.923001, "logit": 2.48, "label": "fake"}
```

## 官方目录推理

```bash
python infer.py --input /path/to/images --output predictions.json
```
