# serving/ — 高吞吐推理服务

单塔 CLIP ViT-H/14 检测器的生产化推理层。模型权重与 `serve/` 完全相同,
**AUC 不变**;全部收益来自工程优化。

## 实测结果（A40 单卡,真实 WildFake 图片,平均 44 KB）

### 推理核心（`bench_core.py`,不含 HTTP 层）

| 配置 | 吞吐 | avg_batch | GPU ms/img |
|---|---:|---:|---:|
| fp32 + batch=1 ← **原 `serve/` 的配置** | **31.4 img/s** | 1.0 | 31.83 |
| bf16 + batch=1 | 87.9 img/s | 1.0 | 11.34 |
| bf16 + batch=32 | 200.4 img/s | 31.75 | 4.94 |
| **bf16 + batch=64** | **209.0 img/s** | 62.55 | 4.73 |

**6.7× 吞吐提升,模型一个字节未改。**

### 端到端 HTTP（`bench.py`）

单进程 uvicorn 约 72–78 img/s。**瓶颈在 Python Web 层,不在模型**:
单张请求 45.6 ms 中,解码 2.2 ms + GPU 10.8 ms,其余约 32 ms 是
FastAPI multipart 解析与 asyncio 开销。

> ⚠️ `uvicorn --workers N` 在本服务上**不可用**:它通过 fork 派生 worker,
> 而 CUDA context 无法跨 fork 继承。横向扩展必须起 N 个独立进程
> (各自 `CUDA_VISIBLE_DEVICES` 绑一张卡),前置 nginx / 负载均衡。

## 四项优化

| # | 优化 | 位置 | 实测收益 |
|---|---|---|---|
| 1 | **bf16 权重** | `detector.py` | 31.83 → 11.34 ms/img（2.8×） |
| 2 | **微批处理 + 贪婪排空** | `batcher.py` | 11.34 → 4.73 ms/img（2.4×） |
| 3 | **uint8 H2D + GPU 端归一化** | `detector.py:infer` | PCIe 流量降为 1/4 |
| 4 | **内容哈希去重** | `cache.py` | 命中即 O(1) 返回 |

### 关于 #2 的一个坑

首版实现 `avg_batch` 恒为 1.0,吞吐退化到 bs=1 水平。原因是闭环客户端下
形成了自锁死循环:batch=1 → GPU 11 ms/张 → 请求间隔 ~11 ms → 超过 10 ms
攒批窗口 → 每个窗口只收到 1 个请求。

修法是**先非阻塞排空队列中已有的全部请求,再开始计时**
(`batcher.py::_collect`),并把窗口放宽到 30 ms。修复后 `avg_batch` 达到 62.55。

## 用法

```bash
# 启动（一张卡一个进程）
CUDA_VISIBLE_DEVICES=0 DTYPE=bf16 MAX_BATCH=64 MAX_WAIT_MS=30 CACHE=1 \
  python -m uvicorn serving.app:app --host 0.0.0.0 --port 8080

# 单张打分
curl -X POST -F "file=@image.jpg" http://127.0.0.1:8080/score
# -> {"pred":1.46e-06,"logit":-13.44,"cached":false,"decode_ms":2.22,"latency_ms":45.57}

curl http://127.0.0.1:8080/stats     # batch 大小、GPU 耗时、缓存命中率、队列深度

# 压测
python serving/bench.py --n 1000 --concurrency 128 --images <dir> --dup 0.5
python serving/bench_core.py --images <dir> --n 2000 --dtype bf16 --max-batch 64
```

环境变量:`DTYPE`(bf16/fp16/fp32)、`MAX_BATCH`、`MAX_WAIT_MS`、
`DECODE_THREADS`、`CACHE`、`CKPT`。**fp32 与 batch=1 保留下来做对照实验。**

## 架构

```
HTTP 请求 → 哈希缓存查询（命中直接返回）
              ↓ 未命中
         解码线程池（PIL 解码时释放 GIL）
              ↓ uint8 [224,224,3]
         微批处理器（攒满 64 或等满 30 ms）
              ↓ [B,224,224,3] uint8 → GPU
         GPU：归一化 + ViT-H/14 前向（bf16）
              ↓ 按索引分发回各请求的 Future
         温度校准 → 概率
```

队列有容量上限(默认 512),满时返回 **503** 而非无限堆积。

## 容量规划

- 单卡 A40:209 img/s,显存峰值 **1.9 GB**(bs=64)→ 不需要 A40,8 GB 卡即可
- 解码配比:**3.4 个 CPU 核 / GPU**(1024² 大图)或约 0.5 核(44 KB 小图)
- 成本:**1.26 GPU-小时 / 百万张**
- 冷启动:12.1 s（加载 2.4 GB checkpoint）
