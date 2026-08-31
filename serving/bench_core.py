"""Inference-core throughput: decode pool + micro-batcher + GPU, without the HTTP layer.

Separates two very different ceilings:
  * what the detection pipeline can do
  * what a single-process Python web server can push through it
"""
from __future__ import annotations
import argparse, asyncio, glob, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serving.batcher import MicroBatcher
from serving.detector import Detector, decode_resize

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True); ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--dtype", default="bf16"); ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--max-wait-ms", type=float, default=30.0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--inflight", type=int, default=256)
    a = ap.parse_args()
    blobs = [Path(p).read_bytes() for p in sorted(glob.glob(a.images + "/*"))]
    blobs = [blobs[i % len(blobs)] for i in range(a.n)]
    det = Detector(dtype=a.dtype)
    b = MicroBatcher(det, max_batch=a.max_batch, max_wait_ms=a.max_wait_ms); b.start()
    pool = ThreadPoolExecutor(max_workers=a.threads)
    loop = asyncio.get_running_loop(); sem = asyncio.Semaphore(a.inflight); lat = []
    async def one(raw):
        async with sem:
            t0 = time.perf_counter()
            arr = await loop.run_in_executor(pool, decode_resize, raw)
            await b.submit(arr)
            lat.append((time.perf_counter() - t0) * 1000)
    await asyncio.gather(*(one(x) for x in blobs[:64]))          # warm
    lat.clear(); t0 = time.perf_counter()
    await asyncio.gather(*(one(x) for x in blobs))
    wall = time.perf_counter() - t0
    lat.sort(); q = lambda p: lat[int(len(lat) * p)]
    s = b.snapshot()
    print(f"dtype={a.dtype} max_batch={a.max_batch} wait={a.max_wait_ms}ms threads={a.threads}")
    print(f"  吞吐 {len(lat)/wall:7.1f} img/s   p50={q(.5):6.1f}ms p95={q(.95):6.1f}ms "
          f"p99={q(.99):6.1f}ms   avg_batch={s['avg_batch_size']}  gpu_ms/img={s['gpu_ms_per_img']}")
    await b.stop()
asyncio.run(main())
