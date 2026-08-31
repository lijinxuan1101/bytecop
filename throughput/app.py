"""High-throughput AIGC detection service.

    uvicorn throughput.app:app --host 0.0.0.0 --port 8080

Env knobs (used by bench.py to A/B the optimizations):
    DTYPE=fp32|bf16     MAX_BATCH=64     MAX_WAIT_MS=30
    DECODE_THREADS=16   CACHE=1|0        CKPT=<path>
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from throughput.batcher import MicroBatcher, Overloaded
from throughput.cache import HashCache
from throughput.detector import Detector, decode_resize

CFG = {
    "dtype": os.getenv("DTYPE", "fp32"),
    "max_batch": int(os.getenv("MAX_BATCH", "32")),
    "max_wait_ms": float(os.getenv("MAX_WAIT_MS", "10")),
    "decode_threads": int(os.getenv("DECODE_THREADS", "16")),
    "cache": os.getenv("CACHE", "1") == "1",
    "ckpt": os.getenv("CKPT", "runs/spatial_tower/spatial_tower_wildfake/best.pt"),
}
STATE: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    t0 = time.perf_counter()
    det = Detector(Path(CFG["ckpt"]), dtype=CFG["dtype"])
    STATE["detector"] = det
    STATE["pool"] = ThreadPoolExecutor(max_workers=CFG["decode_threads"])
    STATE["cache"] = HashCache() if CFG["cache"] else None
    b = MicroBatcher(det, max_batch=CFG["max_batch"], max_wait_ms=CFG["max_wait_ms"])
    b.start()
    STATE["batcher"] = b
    STATE["boot_s"] = round(time.perf_counter() - t0, 2)
    print(f"[throughput] ready in {STATE['boot_s']}s  cfg={CFG}", flush=True)
    yield
    await b.stop()
    STATE["pool"].shutdown(wait=False)


app = FastAPI(title="AIGC detector", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True, "cfg": CFG, "boot_s": STATE.get("boot_s")}


@app.get("/stats")
async def stats() -> dict:
    c = STATE["cache"]
    return {"cfg": CFG, "batcher": STATE["batcher"].snapshot(),
            "cache": c.snapshot() if c else None}


@app.post("/score")
async def score(file: UploadFile = File(...)) -> JSONResponse:
    raw = await file.read()
    t0 = time.perf_counter()
    cache = STATE["cache"]
    if cache is not None:
        k = cache.key(raw)
        hit = cache.get(k)
        if hit is not None:
            det = STATE["detector"]
            return JSONResponse({"pred": float(det.probs(__import__("numpy").array([hit]))[0]),
                                 "logit": hit, "cached": True,
                                 "latency_ms": round((time.perf_counter() - t0) * 1000, 2)})
    loop = asyncio.get_running_loop()
    try:
        arr = await loop.run_in_executor(STATE["pool"], decode_resize, raw)
    except Exception as exc:                                  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"decode failed: {exc}") from exc
    decode_ms = (time.perf_counter() - t0) * 1000
    try:
        logit = await STATE["batcher"].submit(arr)
    except Overloaded:
        raise HTTPException(status_code=503, detail="overloaded") from None
    if cache is not None:
        cache.put(k, logit)
    import numpy as np
    return JSONResponse({
        "pred": float(STATE["detector"].probs(np.array([logit]))[0]),
        "logit": logit, "cached": False,
        "decode_ms": round(decode_ms, 2),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    })
