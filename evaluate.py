"""Official robustness evaluation matrix for a trained AIGC detection model.

Runs the complete official evaluation:
    - Clean (no transform)
    - JPEG compression: q in {90, 70, 50, 30}
    - Gaussian blur: σ in {0.5, 1.0, 2.0}
    - Resize (down→up): scale in {0.5, 0.25}
    - Gaussian noise: σ in {0.02, 0.05, 0.10}
    - Color jitter: strength = 0.2
    - Center crop: fraction = 0.8

Final Score = 0.50 × AUC_clean + 0.50 × AUC_robust
where AUC_robust is the mean over all degraded conditions.

Usage
-----
    python evaluate.py --backbone clip_h --ckpt runs/clip_h/best.pt \
        --data /path/to/dataset/test \
        --calibrator runs/clip_h/calibrator.pkl \
        --output runs/clip_h/eval_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from data.transforms import (
    TransformName,
    apply_transform,
)
from data.dataset import AIGCDataset


# ------------------------------------------------------------------
# Official evaluation matrix
# ------------------------------------------------------------------

EVAL_CONDITIONS: list[tuple[str, TransformName | None, float | None]] = [
    ("clean",             None,                None  ),
    ("jpeg_q90",          "jpeg_compression",  90    ),
    ("jpeg_q70",          "jpeg_compression",  70    ),
    ("jpeg_q50",          "jpeg_compression",  50    ),
    ("jpeg_q30",          "jpeg_compression",  30    ),
    ("blur_s0.5",         "gaussian_blur",     0.5   ),
    ("blur_s1.0",         "gaussian_blur",     1.0   ),
    ("blur_s2.0",         "gaussian_blur",     2.0   ),
    ("resize_0.5",        "resize",            0.5   ),
    ("resize_0.25",       "resize",            0.25  ),
    ("noise_s0.02",       "gaussian_noise",    0.02  ),
    ("noise_s0.05",       "gaussian_noise",    0.05  ),
    ("noise_s0.10",       "gaussian_noise",    0.10  ),
    ("color_jitter",      "color_jitter",      0.2   ),
    ("center_crop_80",    "center_crop",       0.8   ),
]


# ------------------------------------------------------------------
# Dataset wrapper that applies a single evaluation transform
# ------------------------------------------------------------------

class _TransformedDataset(Dataset):
    def __init__(
        self,
        base: AIGCDataset,
        pil_transform: callable | None,
        tensor_transform: T.Compose,
    ) -> None:
        self.base = base
        self.pil_transform = pil_transform
        self.tensor_transform = tensor_transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        from PIL import Image as PILImage
        path, label = self.base.samples[idx]
        with PILImage.open(path) as img:
            image = img.convert("RGB")
        if self.pil_transform is not None:
            image = self.pil_transform(image)
        return self.tensor_transform(image), label


# ------------------------------------------------------------------
# Pre-processing
# ------------------------------------------------------------------

def _tensor_transform(backbone: str) -> T.Compose:
    if backbone == "clip_h":
        return T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711)),
        ])
    return T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])


# ------------------------------------------------------------------
# Inference helpers
# ------------------------------------------------------------------

def _build_model(backbone: str, ckpt: Path, device: torch.device) -> nn.Module:
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


@torch.no_grad()
def _run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    all_logits, all_labels = [], []
    for images, labels in tqdm(loader, leave=False):
        logits = model(images.to(device))
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    return (
        torch.cat(all_logits).numpy(),
        torch.cat(all_labels).numpy(),
    )


# ------------------------------------------------------------------
# Main evaluation function
# ------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = _build_model(args.backbone, Path(args.ckpt), device)

    scaler = None
    if args.calibrator:
        with open(args.calibrator, "rb") as f:
            scaler = pickle.load(f)
        print(f"Loaded temperature scaler (T={scaler.temperature_:.4f})")

    t_transform = _tensor_transform(args.backbone)
    base_ds = AIGCDataset(Path(args.data))
    print(f"Test set: {len(base_ds)} samples  {base_ds.class_counts()}")

    results: dict[str, dict] = {}
    auc_robust_list: list[float] = []

    for condition_name, transform_name, value in EVAL_CONDITIONS:
        if transform_name is None:
            pil_fn = None
        else:
            def pil_fn(img, tn=transform_name, v=value):
                return apply_transform(img, tn, value=v)

        ds = _TransformedDataset(base_ds, pil_fn, t_transform)
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )

        logits, labels = _run_inference(model, loader, device)
        if scaler is not None:
            probs = scaler.predict_proba(logits)
        else:
            probs = torch.sigmoid(torch.from_numpy(logits)).numpy()

        auc = float(roc_auc_score(labels, probs))
        results[condition_name] = {"auc": auc}

        marker = "(clean)" if condition_name == "clean" else ""
        print(f"  {condition_name:<20s}  AUC={auc:.4f}  {marker}")

        if condition_name != "clean":
            auc_robust_list.append(auc)

    auc_clean = results["clean"]["auc"]
    auc_robust = float(np.mean(auc_robust_list))
    final_score = 0.5 * auc_clean + 0.5 * auc_robust
    worst_auc = float(np.min(auc_robust_list))

    summary = {
        "auc_clean": auc_clean,
        "auc_robust": auc_robust,
        "final_score": final_score,
        "worst_condition_auc": worst_auc,
        "conditions": results,
    }

    print(f"\n{'─'*50}")
    print(f"AUC_clean  = {auc_clean:.4f}")
    print(f"AUC_robust = {auc_robust:.4f}  (mean over {len(auc_robust_list)} conditions)")
    print(f"Final Score = 0.50×{auc_clean:.4f} + 0.50×{auc_robust:.4f} = {final_score:.4f}")
    print(f"Worst condition AUC = {worst_auc:.4f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {out_path}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AIGC detection model on official robustness matrix.")
    parser.add_argument("--backbone", choices=["clip_h", "dino_h"], required=True)
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint (best.pt).")
    parser.add_argument("--data", required=True, help="Test dataset root (real/ and fake/ sub-folders).")
    parser.add_argument("--calibrator", default=None, help="Path to calibrator.pkl (optional).")
    parser.add_argument("--output", default=None, help="Path to save JSON results.")
    parser.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(_parse_args())
