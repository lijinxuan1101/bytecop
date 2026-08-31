"""Spatial-only AIGC detector — the model we ship, and the viz backend.

Loads OpenCLIP ViT-H/14 (WildFake ``best.pt``) once, then scores images.
``pred`` is P(AI-generated). Official contest JSON is ``image_path`` + ``pred``;
the extra fields are for the visualization layer.

    from serve.spatial_backend import SpatialDetector

    det = SpatialDetector()
    det.score_path("demo.jpg")
    det.score_dir(Path("images/"))
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.special import expit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from models.clip_tower import CLIPTower

_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

DEFAULT_CKPT = Path("runs/spatial_tower/spatial_tower_wildfake/best.pt")
DEFAULT_UNFREEZE_BLOCKS = 2
DEFAULT_PROJ_DIM = 512
DEFAULT_DROPOUT = 0.1
IMG_SIZE = 224
THRESHOLD = 0.5

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def clip_transform(*, img_size: int = IMG_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
    ])


class _PathDataset(Dataset):
    def __init__(self, paths: list[Path], transform: T.Compose) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        path = self.paths[idx]
        with Image.open(path) as img:
            tensor = self.transform(img.convert("RGB"))
        return tensor, str(path.resolve())


def list_images(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMG_EXTENSIONS
    )


def _record(*, path: str | None, logit: float, temperature: float) -> dict:
    pred = float(expit(logit / max(temperature, 1e-8)))
    return {
        "image_path": path,
        "pred": round(pred, 6),
        "logit": round(float(logit), 6),
        "label": "fake" if pred >= THRESHOLD else "real",
    }


class SpatialDetector:
    """Frozen CLIP-H spatial tower. Construct once, reuse for every request."""

    def __init__(
        self,
        *,
        ckpt: str | Path = DEFAULT_CKPT,
        device: str | torch.device | None = None,
        temperature: float = 1.0,
        unfreeze_blocks: int = DEFAULT_UNFREEZE_BLOCKS,
        proj_dim: int = DEFAULT_PROJ_DIM,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        self.ckpt = Path(ckpt).resolve()
        if not self.ckpt.is_file():
            raise FileNotFoundError(f"spatial checkpoint not found: {self.ckpt}")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.temperature = float(temperature)
        self.transform = clip_transform()
        self.model = self._load(
            self.ckpt,
            unfreeze_blocks=unfreeze_blocks,
            proj_dim=proj_dim,
            dropout=dropout,
        )

    def _load(
        self,
        ckpt: Path,
        *,
        unfreeze_blocks: int,
        proj_dim: int,
        dropout: float,
    ) -> nn.Module:
        model = CLIPTower(
            unfreeze_blocks=unfreeze_blocks,
            proj_dim=proj_dim,
            dropout=dropout,
            load_weights=False,
        )
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        model.to(self.device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    def info(self) -> dict:
        return {
            "model": "spatial_clip_h",
            "ckpt": str(self.ckpt),
            "device": str(self.device),
            "img_size": IMG_SIZE,
            "threshold": THRESHOLD,
            "temperature": self.temperature,
        }

    @torch.no_grad()
    def score_pil(self, image: Image.Image, *, path: str | None = None) -> dict:
        """Score one RGB PIL image (visualization upload path)."""
        return self.score_pils([image], names=[path])[0]

    @torch.no_grad()
    def score_pils(
        self,
        images: list[Image.Image],
        *,
        names: list[str | None] | None = None,
    ) -> list[dict]:
        """Score a small in-memory batch (robustness sweep in the viz app)."""
        if not images:
            return []
        if names is None:
            names = [None] * len(images)
        tensors = torch.stack([
            self.transform(img.convert("RGB")) for img in images
        ]).to(self.device)
        logits = np.asarray(self.model(tensors).cpu().numpy()).reshape(-1)
        return [
            _record(path=name, logit=float(logit), temperature=self.temperature)
            for name, logit in zip(names, logits)
        ]

    def score_path(self, path: str | Path) -> dict:
        path = Path(path)
        with Image.open(path) as img:
            return self.score_pil(img, path=str(path.resolve()))

    @torch.no_grad()
    def score_paths(
        self,
        paths: list[Path],
        *,
        batch_size: int = 32,
        workers: int = 4,
        show_progress: bool = False,
    ) -> list[dict]:
        if not paths:
            return []
        ds = _PathDataset(paths, self.transform)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=self.device.type == "cuda",
        )
        records: list[dict] = []
        for tensors, batch_paths in tqdm(loader, desc="spatial", disable=not show_progress):
            logits = self.model(tensors.to(self.device)).cpu().numpy()
            for path, logit in zip(batch_paths, np.asarray(logits).reshape(-1)):
                records.append(
                    _record(path=path, logit=float(logit), temperature=self.temperature)
                )
        return records

    def score_dir(
        self,
        root: str | Path,
        *,
        batch_size: int = 32,
        workers: int = 4,
        show_progress: bool = True,
    ) -> list[dict]:
        root = Path(root)
        if not root.is_dir():
            raise NotADirectoryError(f"{root} is not a directory")
        return self.score_paths(
            list_images(root),
            batch_size=batch_size,
            workers=workers,
            show_progress=show_progress,
        )


def official_records(records: list[dict]) -> list[dict]:
    """Contest JSON: only ``image_path`` and ``pred``."""
    return [{"image_path": r["image_path"], "pred": r["pred"]} for r in records]
