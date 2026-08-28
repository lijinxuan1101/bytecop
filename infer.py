"""Inference script: score all images in a directory and write results to JSON.

Output format (as required by the official deliverables):
    [
        {"image_path": "/abs/path/to/img.jpg", "pred": 0.92},
        ...
    ]

``pred`` is the calibrated probability that the image is AI-generated (1 = AI,
0 = real).

Usage
-----
    # Single tower
    python infer.py --backbone clip_h --ckpt runs/clip_h/best.pt \
        --input /path/to/images \
        --output predictions.json \
        [--calibrator runs/clip_h/calibrator.pkl]

    # Dual-tower (logit average)
    python infer.py --backbone dual \
        --clip-ckpt runs/clip_h/best.pt \
        --dino-ckpt runs/dino_h/best.pt \
        --input /path/to/images \
        --output predictions.json \
        [--calibrator runs/dual/calibrator.pkl]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm


_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ------------------------------------------------------------------
# Dataset for a flat image directory (no labels)
# ------------------------------------------------------------------

class _ImageFolder(Dataset):
    def __init__(self, root: Path, transform: T.Compose) -> None:
        self.paths = sorted(
            p for p in root.rglob("*")
            if p.suffix.lower() in _IMG_EXTENSIONS
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        path = self.paths[idx]
        with Image.open(path) as img:
            tensor = self.transform(img.convert("RGB"))
        return tensor, str(path)


# ------------------------------------------------------------------
# Pre-processing
# ------------------------------------------------------------------

def _clip_transform() -> T.Compose:
    return T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)),
    ])


def _dino_transform() -> T.Compose:
    return T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------

def _load_single_tower(backbone: str, ckpt: Path, device: torch.device) -> nn.Module:
    if backbone == "clip_h":
        from models.clip_tower import CLIPTower
        model = CLIPTower(unfreeze_blocks=4)
    elif backbone == "dino_h":
        from models.dino_tower import DINOTower
        model = DINOTower(unfreeze_blocks=4)
    else:
        raise ValueError(f"Unknown backbone: {backbone!r}")

    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


# ------------------------------------------------------------------
# Inference runners
# ------------------------------------------------------------------

@torch.no_grad()
def _run_single(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler,
) -> list[dict]:
    records = []
    for tensors, paths in tqdm(loader, desc="Inference"):
        logits = model(tensors.to(device)).cpu().numpy()
        if scaler is not None:
            probs = scaler.predict_proba(logits)
        else:
            import numpy as np
            from scipy.special import expit
            probs = expit(logits)
        for path, prob in zip(paths, probs):
            records.append({"image_path": path, "pred": round(float(prob), 6)})
    return records


@torch.no_grad()
def _run_dual(
    clip_model: nn.Module,
    dino_model: nn.Module,
    clip_loader: DataLoader,
    dino_loader: DataLoader,
    device: torch.device,
    scaler,
) -> list[dict]:
    """Run both towers and fuse their logits via average."""
    import numpy as np
    from scipy.special import expit

    clip_records: dict[str, float] = {}
    for tensors, paths in tqdm(clip_loader, desc="CLIP inference"):
        logits = clip_model(tensors.to(device)).cpu().numpy()
        for path, logit in zip(paths, logits):
            clip_records[path] = float(logit)

    dino_records: dict[str, float] = {}
    for tensors, paths in tqdm(dino_loader, desc="DINO inference"):
        logits = dino_model(tensors.to(device)).cpu().numpy()
        for path, logit in zip(paths, logits):
            dino_records[path] = float(logit)

    records = []
    all_paths = sorted(clip_records.keys())
    fused_logits = np.array(
        [(clip_records[p] + dino_records[p]) / 2.0 for p in all_paths]
    )
    if scaler is not None:
        probs = scaler.predict_proba(fused_logits)
    else:
        probs = expit(fused_logits)

    for path, prob in zip(all_paths, probs):
        records.append({"image_path": path, "pred": round(float(prob), 6)})
    return records


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def infer(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"{input_dir} is not a directory.")

    scaler = None
    if args.calibrator:
        with open(args.calibrator, "rb") as f:
            scaler = pickle.load(f)
        print(f"Loaded temperature scaler (T={scaler.temperature_:.4f})")

    if args.backbone == "dual":
        clip_model = _load_single_tower("clip_h", Path(args.clip_ckpt), device)
        dino_model = _load_single_tower("dino_h", Path(args.dino_ckpt), device)
        clip_ds = _ImageFolder(input_dir, _clip_transform())
        dino_ds = _ImageFolder(input_dir, _dino_transform())
        clip_loader = DataLoader(clip_ds, batch_size=args.batch_size,
                                 num_workers=args.workers, pin_memory=True)
        dino_loader = DataLoader(dino_ds, batch_size=args.batch_size,
                                 num_workers=args.workers, pin_memory=True)
        records = _run_dual(clip_model, dino_model, clip_loader, dino_loader, device, scaler)
    else:
        model = _load_single_tower(args.backbone, Path(args.ckpt), device)
        transform = _clip_transform() if args.backbone == "clip_h" else _dino_transform()
        ds = _ImageFolder(input_dir, transform)
        loader = DataLoader(ds, batch_size=args.batch_size,
                            num_workers=args.workers, pin_memory=True)
        records = _run_single(model, loader, device, scaler)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nSaved {len(records)} predictions to {out_path}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AIGC detection inference on an image directory.")
    parser.add_argument("--backbone", choices=["clip_h", "dino_h", "dual"], required=True)
    parser.add_argument("--input", required=True, help="Directory of images to score.")
    parser.add_argument("--output", required=True, help="Output JSON file path.")
    parser.add_argument("--ckpt", default=None, help="Checkpoint for single-tower mode.")
    parser.add_argument("--clip-ckpt", default=None, dest="clip_ckpt", help="CLIP checkpoint for dual mode.")
    parser.add_argument("--dino-ckpt", default=None, dest="dino_ckpt", help="DINO checkpoint for dual mode.")
    parser.add_argument("--calibrator", default=None, help="Path to calibrator.pkl (optional).")
    parser.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    return parser.parse_args()


if __name__ == "__main__":
    infer(_parse_args())
