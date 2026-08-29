"""Smoke test — verify end-to-end pipeline connectivity without real data.

Creates a tiny synthetic dataset in a temp directory, runs 2 training epochs,
evaluates on val, and checks calibration.  No internet access required.

Usage
-----
    python scripts/smoke_test.py
    python scripts/smoke_test.py --backbone dino_h
    python scripts/smoke_test.py --config configs/smoke.yaml
"""

from __future__ import annotations

import argparse
import pickle
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import yaml

# Make sure project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms as T

from calibration.temperature_scaling import TemperatureScaler
from data.dataset import AIGCDataset
from data.transforms import build_train_augment


# ---------------------------------------------------------------------------
# Synthetic dataset helpers
# ---------------------------------------------------------------------------

def _make_synthetic_split(root: Path, n_per_class: int, img_size: int) -> None:
    """Create random RGB images under root/real/ and root/fake/."""
    rng = random.Random(42)
    for label in ("real", "fake"):
        folder = root / label
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            pixels = bytes(rng.randint(0, 255) for _ in range(img_size * img_size * 3))
            img = Image.frombytes("RGB", (img_size, img_size), pixels)
            img.save(folder / f"{i:04d}.png")


def _build_synthetic_dataset(tmp_root: Path, cfg: dict) -> Path:
    """Build train / val / calibration splits and return the dataset root."""
    n = cfg.get("num_synthetic_samples", 20)
    img_size = cfg.get("img_size", 224)
    for split in ("train", "val", "calibration"):
        _make_synthetic_split(tmp_root / split, n_per_class=n, img_size=img_size)
    return tmp_root


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def _preprocess(backbone: str, img_size: int = 224) -> T.Compose:
    if backbone == "clip_h":
        mean = (0.48145466, 0.4578275, 0.40821073)
        std  = (0.26862954, 0.26130258, 0.27577711)
    else:
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def _build_model(backbone: str, cfg: dict, device: torch.device) -> nn.Module:
    if backbone == "clip_h":
        from models.clip_tower import CLIPTower
        model = CLIPTower(
            unfreeze_blocks=cfg.get("unfreeze_blocks", 2),
            proj_dim=cfg.get("proj_dim", 256),
            dropout=cfg.get("dropout", 0.1),
        )
    elif backbone == "dino_h":
        from models.dino_tower import DINOTower
        model = DINOTower(
            unfreeze_blocks=cfg.get("unfreeze_blocks", 2),
            proj_dim=cfg.get("proj_dim", 256),
            dropout=cfg.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone!r}")
    return model.to(device)


# ---------------------------------------------------------------------------
# Eval helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    logits_np = torch.cat(all_logits).numpy()
    labels_np = torch.cat(all_labels).numpy()
    probs_np  = torch.sigmoid(torch.from_numpy(logits_np)).numpy()
    auc = roc_auc_score(labels_np, probs_np) if len(set(labels_np)) > 1 else float("nan")
    return {"auc": auc, "logits": logits_np, "labels": labels_np}


# ---------------------------------------------------------------------------
# Main smoke test
# ---------------------------------------------------------------------------

def smoke_test(backbone: str, cfg: dict) -> None:
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f" Smoke Test — backbone={backbone}  device={device}")
    print(f"{'='*60}")

    # 1. Synthetic data
    print("\n[1/5] Building synthetic dataset …")
    tmp_dir = Path(tempfile.mkdtemp(prefix="smoke_"))
    try:
        data_root = _build_synthetic_dataset(tmp_dir, cfg)
        print(f"      Synthetic images at: {tmp_dir}")

        # 2. Build model
        print("\n[2/5] Loading model …")
        model = _build_model(backbone, cfg, device)
        counts = model.param_count()
        print(f"      Total params   : {counts['total']:,}")
        print(f"      Trainable params: {counts['trainable']:,}")

        # 3. Build datasets
        print("\n[3/5] Building DataLoaders …")
        tensor_tfm = _preprocess(backbone, cfg.get("img_size", 224))
        pil_aug = build_train_augment(clean_prob=cfg.get("clean_prob", 0.3))

        def train_tfm(img: Image.Image) -> torch.Tensor:
            return tensor_tfm(pil_aug(img))

        batch_size = cfg.get("batch_size", 4)
        train_ds = AIGCDataset(data_root / "train",   transform=train_tfm)
        val_ds   = AIGCDataset(data_root / "val",     transform=tensor_tfm)
        cal_ds   = AIGCDataset(data_root / "calibration", transform=tensor_tfm)
        print(f"      Train: {len(train_ds)}  Val: {len(val_ds)}  Cal: {len(cal_ds)}")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
        cal_loader   = DataLoader(cal_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

        # 4. Mini training loop
        print(f"\n[4/5] Training {cfg.get('epochs', 2)} epochs …")
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            model.trainable_parameters(),
            lr=cfg.get("lr", 1e-4),
            weight_decay=cfg.get("weight_decay", 0.01),
        )

        for epoch in range(1, cfg.get("epochs", 2) + 1):
            model.train()
            ep_loss = 0.0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.float().to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
            val_m = _evaluate(model, val_loader, device)
            print(f"      Epoch {epoch}: train_loss={ep_loss/len(train_loader):.4f}  val_auc={val_m['auc']:.4f}")

        # 5. Calibration
        print("\n[5/5] Temperature scaling …")
        cal_m = _evaluate(model, cal_loader, device)
        scaler = TemperatureScaler()
        scaler.fit(cal_m["logits"], cal_m["labels"])
        metrics = scaler.calibration_metrics(cal_m["logits"], cal_m["labels"])
        print(f"      Temperature : {metrics['temperature']:.4f}")
        print(f"      ECE         : {metrics['ece']:.4f}")
        print(f"      Brier Score : {metrics['brier']:.4f}")

        elapsed = time.time() - t_start
        print(f"\n{'='*60}")
        print(f" PASSED  ({elapsed:.1f}s) — all pipeline stages completed successfully.")
        print(f"{'='*60}\n")

    except Exception as exc:
        print(f"\n FAILED: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end pipeline smoke test.")
    parser.add_argument("--backbone", choices=["clip_h", "dino_h"], default="clip_h")
    parser.add_argument("--config", default="configs/smoke.yaml",
                        help="YAML config file (default: configs/smoke.yaml)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}")
        sys.exit(1)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["backbone"] = args.backbone
    smoke_test(args.backbone, cfg)
