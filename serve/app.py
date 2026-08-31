"""HTTP backend for the visualization layer (spatial tower only).

Start once; the 2.4G checkpoint stays in GPU memory.

    source ~/techjam/venv/bin/activate
    pip install fastapi uvicorn python-multipart   # if missing
    uvicorn serve.app:app --host 0.0.0.0 --port 8008

Do not bind GPU 0 on this box if vLLM is using it:
    CUDA_VISIBLE_DEVICES=1 uvicorn serve.app:app --host 0.0.0.0 --port 8008
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from serve.spatial_backend import DEFAULT_CKPT, SpatialDetector

_detector: SpatialDetector | None = None


def get_detector() -> SpatialDetector:
    if _detector is None:
        raise RuntimeError("detector not loaded")
    return _detector


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _detector
    _detector = SpatialDetector(ckpt=DEFAULT_CKPT)
    yield
    _detector = None


app = FastAPI(
    title="TraceLens: Robust AI Image Detection Beyond Redistribution",
    description="P(AI-generated) from the OpenCLIP-H spatial tower.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"ok": True, **get_detector().info()}


@app.get("/v1/model")
def model_info() -> dict:
    return get_detector().info()


@app.post("/v1/score")
async def score_upload(file: UploadFile = File(...)) -> dict:
    """Score one uploaded image. Used by the visualization UI."""
    try:
        image = Image.open(BytesIO(await file.read()))
        image.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"unreadable image: {exc}") from exc
    return get_detector().score_pil(image, path=file.filename)


class ScoreDirBody(BaseModel):
    directory: str


@app.post("/v1/score-dir")
def score_dir(body: ScoreDirBody) -> dict:
    """Score every image under ``directory``."""
    root = Path(body.directory)
    if not root.is_dir():
        raise HTTPException(status_code=400, detail=f"not a directory: {root}")
    records = get_detector().score_dir(root, show_progress=False)
    return {"n": len(records), "records": records}
