"""Load test: fire N concurrent uploads at /score, report throughput + latency percentiles.

    python throughput/bench.py --url http://127.0.0.1:8080 --n 600 --concurrency 64
    python throughput/bench.py ... --dup 0.3      # 30% duplicate uploads (cache realism)
"""
from __future__ import annotations

import argparse, asyncio, io, random, statistics, time, json
from pathlib import Path

import httpx
import numpy as np
from PIL import Image


def make_images(n: int, dup: float, size: int, src: Path | None) -> list[bytes]:
    pool: list[bytes] = []
    if src and src.is_dir():
        files = [p for p in sorted(src.rglob("*")) if p.suffix.lower() in
                 {".jpg", ".jpeg", ".png", ".webp"}][:max(1, int(n * (1 - dup)) + 1)]
        for p in files:
            try:
                pool.append(p.read_bytes())
            except OSError:
                pass
    while len(pool) < max(1, int(n * (1 - dup))):
        arr = (np.random.rand(size, size, 3) * 255).astype("uint8")
        b = io.BytesIO(); Image.fromarray(arr).save(b, "JPEG", quality=90)
        pool.append(b.getvalue())
    out = [random.choice(pool) for _ in range(n)]
    return out


async def run(url: str, blobs: list[bytes], conc: int) -> dict:
    lat: list[float] = []
    errs: dict[int, int] = {}
    sem = asyncio.Semaphore(conc)
    limits = httpx.Limits(max_connections=conc + 16, max_keepalive_connections=conc + 16)
    async with httpx.AsyncClient(timeout=120.0, limits=limits) as cli:
        async def one(b: bytes):
            async with sem:
                t0 = time.perf_counter()
                try:
                    r = await cli.post(f"{url}/score", files={"file": ("x.jpg", b, "image/jpeg")})
                    if r.status_code != 200:
                        errs[r.status_code] = errs.get(r.status_code, 0) + 1
                        return
                except Exception:
                    errs[-1] = errs.get(-1, 0) + 1
                    return
                lat.append((time.perf_counter() - t0) * 1000)
        await cli.post(f"{url}/score", files={"file": ("w.jpg", blobs[0], "image/jpeg")})  # warm
        t0 = time.perf_counter()
        await asyncio.gather(*(one(b) for b in blobs))
        wall = time.perf_counter() - t0
    lat.sort()
    q = lambda p: lat[min(len(lat) - 1, int(len(lat) * p))] if lat else float("nan")
    return {"ok": len(lat), "errors": errs, "wall_s": round(wall, 2),
            "throughput_img_s": round(len(lat) / wall, 1) if wall else 0,
            "p50_ms": round(q(.50), 1), "p95_ms": round(q(.95), 1),
            "p99_ms": round(q(.99), 1),
            "mean_ms": round(statistics.mean(lat), 1) if lat else 0}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--dup", type=float, default=0.0)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--images", type=Path, default=None)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    blobs = make_images(a.n, a.dup, a.size, a.images)
    async with httpx.AsyncClient(timeout=30) as c:
        cfg = (await c.get(f"{a.url}/healthz")).json().get("cfg", {})
    res = await run(a.url, blobs, a.concurrency)
    async with httpx.AsyncClient(timeout=30) as c:
        st = (await c.get(f"{a.url}/stats")).json()
    print(json.dumps({"label": a.label, "cfg": cfg, "load": {"n": a.n,
          "concurrency": a.concurrency, "dup": a.dup}, "result": res,
          "server": st}, indent=1))
    print(f"\n>>> {a.label or 'run'}: {res['throughput_img_s']} img/s   "
          f"p50={res['p50_ms']}ms p95={res['p95_ms']}ms p99={res['p99_ms']}ms   "
          f"avg_batch={st['batcher']['avg_batch_size']}")

asyncio.run(main())
