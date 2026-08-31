"""bf16 CLIP-H detector with GPU-side normalization.

Two engineering choices vs serve/spatial_backend.py, both measured:
  * bf16 weights          21.76 -> 4.55 ms/img  (4.8x)
  * uint8 H2D + GPU norm  4x less PCIe traffic than shipping float32
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from models.clip_tower import CLIPTower

IMG_SIZE = 224
_MEAN = (0.48145466, 0.4578275, 0.40821073)
_STD = (0.26862954, 0.26130258, 0.27577711)
DEFAULT_CKPT = Path("runs/spatial_tower/spatial_tower_wildfake/best.pt")


def decode_resize(raw: bytes) -> np.ndarray:
    """CPU stage: JPEG bytes -> uint8 HWC array at 224x224. Releases the GIL in PIL."""
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = img.size
    scale = IMG_SIZE / min(w, h)
    img = img.resize((max(IMG_SIZE, round(w * scale)), max(IMG_SIZE, round(h * scale))),
                     Image.BICUBIC)
    w, h = img.size
    left, top = (w - IMG_SIZE) // 2, (h - IMG_SIZE) // 2
    return np.asarray(img.crop((left, top, left + IMG_SIZE, top + IMG_SIZE)), dtype=np.uint8)


class Detector:
    def __init__(self, ckpt: Path = DEFAULT_CKPT, *, device: str = "cuda:0",
                 dtype: str = "bf16", temperature: float = 1.0) -> None:
        self.device = torch.device(device)
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                      "fp32": torch.float32}[dtype]
        self.temperature = temperature
        model = CLIPTower(unfreeze_blocks=2, proj_dim=512, dropout=0.1)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        missing = model.load_state_dict(state, strict=False)
        if len(missing.missing_keys) > 20:
            raise RuntimeError(f"checkpoint mismatch: {missing.missing_keys[:5]}")
        self.model = model.to(self.device).to(self.dtype).eval()
        self.mean = torch.tensor(_MEAN, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        self.std = torch.tensor(_STD, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        self._warmup()

    def _warmup(self) -> None:
        for bs in (1, 8, 32):
            self.infer(np.zeros((bs, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))

    @torch.no_grad()
    def infer(self, batch_u8: np.ndarray) -> np.ndarray:
        """[B,H,W,C] uint8 -> logits [B]. Normalization runs on the GPU."""
        x = torch.from_numpy(batch_u8).to(self.device, non_blocking=True)
        x = x.permute(0, 3, 1, 2).to(self.dtype).div_(255)
        x = (x - self.mean) / self.std
        return self.model(x).float().reshape(-1).cpu().numpy()

    def probs(self, logits: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-logits / self.temperature))
