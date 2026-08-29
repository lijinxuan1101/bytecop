"""Train a single CLIP-H or DINOv3-H+ tower for AIGC detection.

Usage
-----
    python train.py --backbone clip_h --data /path/to/dataset --output runs/clip_h
    python train.py --backbone dino_h --data /path/to/dataset --output runs/dino_h

Dataset layout expected at ``--data``:
    <root>/
        train/
            real/   ... real images
            fake/   ... AI-generated images
        val/
            real/
            fake/
        calibration/        (independent split, NOT used for early stopping)
            real/
            fake/

After training, a checkpoint ``best.pt`` and a temperature-scaled calibrator
``calibrator.pkl`` are saved under ``--output``.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm

from calibration.temperature_scaling import TemperatureScaler
from data.transforms import build_train_augment
from data.dataset import AIGCDataset


# ------------------------------------------------------------------
# Pre-processing pipelines
# ------------------------------------------------------------------

def _clip_preprocess(img_size: int = 224) -> T.Compose:
    """Standard CLIP normalisation (mean/std from OpenAI)."""
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)),
    ])


def _dino_preprocess(img_size: int = 224) -> T.Compose:
    """Standard ImageNet normalisation used by DINOv2/v3."""
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_model(backbone: str, device: torch.device) -> nn.Module:
    if backbone == "clip_h":
        from models.clip_tower import CLIPTower
        model = CLIPTower(unfreeze_blocks=4, proj_dim=512, dropout=0.1)
    elif backbone == "dino_h":
        from models.dino_tower import DINOTower
        model = DINOTower(unfreeze_blocks=4, proj_dim=512, dropout=0.1)
    else:
        raise ValueError(f"Unknown backbone: {backbone!r}. Choose 'clip_h' or 'dino_h'.")
    return model.to(device)


def _preprocess_fn(backbone: str) -> T.Compose:
    if backbone == "clip_h":
        return _clip_preprocess()
    return _dino_preprocess()


def _build_dataset(split_dir: Path, backbone: str, *, augment: bool) -> AIGCDataset:
    """Build dataset with appropriate pre-processing.

    When ``augment=True`` (training split) the official single-transform policy
    is applied before tensor normalisation.  Validation/calibration use clean
    images only.
    """
    tensor_transform = _preprocess_fn(backbone)

    if augment:
        pil_augment = build_train_augment(clean_prob=0.3)

        def transform(img):
            img = pil_augment(img)
            return tensor_transform(img)
    else:
        transform = tensor_transform

    return AIGCDataset(split_dir, transform=transform)


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        all_logits.append(logits.cpu())
        all_labels.append(labels)

    logits_np = torch.cat(all_logits).numpy()
    labels_np = torch.cat(all_labels).numpy()
    probs_np = torch.sigmoid(torch.from_numpy(logits_np)).numpy()

    auc = roc_auc_score(labels_np, probs_np)
    loss = float(nn.functional.binary_cross_entropy_with_logits(
        torch.from_numpy(logits_np),
        torch.from_numpy(labels_np).float(),
    ))
    return {"auc": auc, "loss": loss, "logits": logits_np, "labels": labels_np}


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_root = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    tb_dir = output_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"TensorBoard: tensorboard --logdir {tb_dir}")

    # Build datasets
    train_ds = _build_dataset(data_root / "train", args.backbone, augment=True)
    val_ds = _build_dataset(data_root / "val", args.backbone, augment=False)
    cal_split = data_root / "calibration"
    has_cal = cal_split.is_dir()
    if has_cal:
        cal_ds = _build_dataset(cal_split, args.backbone, augment=False)

    print(f"Train: {len(train_ds)} samples  {train_ds.class_counts()}")
    print(f"Val:   {len(val_ds)} samples  {val_ds.class_counts()}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    if has_cal:
        cal_loader = DataLoader(
            cal_ds, batch_size=args.batch_size * 2, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )

    # Model
    model = _build_model(args.backbone, device)
    counts = model.param_count()
    print(f"Params — total: {counts['total']:,}  trainable: {counts['trainable']:,}")

    # Class-balanced loss weight
    class_counts = train_ds.class_counts()
    n_real = class_counts.get(0, 1)
    n_fake = class_counts.get(1, 1)
    pos_weight = torch.tensor([n_real / n_fake], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_auc = 0.0
    history = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()

            writer.add_scalar("train/step_loss", loss.item(), global_step)
            global_step += 1

        scheduler.step()

        train_loss = running_loss / len(train_loader)
        val_metrics = _evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        # TensorBoard — per-epoch scalars
        writer.add_scalars("loss", {"train": train_loss, "val": val_metrics["loss"]}, epoch)
        writer.add_scalar("val/auc",  val_metrics["auc"], epoch)
        writer.add_scalar("train/lr", lr_now, epoch)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_auc": val_metrics["auc"],
            "lr": lr_now,
            "elapsed_s": round(elapsed, 1),
        }
        history.append(record)
        print(
            f"Epoch {epoch:3d} | train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_auc={val_metrics['auc']:.4f} "
            f"({elapsed:.0f}s)"
        )

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save(model.state_dict(), output_dir / "best.pt")
            print(f"  ↑ Saved best model (val_auc={best_auc:.4f})")

    writer.add_hparams(
        hparam_dict={
            "backbone": args.backbone,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "weight_decay": args.weight_decay,
        },
        metric_dict={"hparam/best_val_auc": best_auc},
    )
    writer.close()
    print(f"\nBest val AUC: {best_auc:.4f}")

    # Save training history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Temperature scaling on calibration split
    if has_cal:
        print("\nFitting temperature scaling on calibration split …")
        model.load_state_dict(torch.load(output_dir / "best.pt", map_location=device, weights_only=True))
        cal_metrics = _evaluate(model, cal_loader, device)

        scaler = TemperatureScaler()
        scaler.fit(cal_metrics["logits"], cal_metrics["labels"])
        cal_quality = scaler.calibration_metrics(cal_metrics["logits"], cal_metrics["labels"])
        print(f"  Temperature: {cal_quality['temperature']:.4f}")
        print(f"  ECE: {cal_quality['ece']:.4f}")
        print(f"  Brier: {cal_quality['brier']:.4f}")

        with open(output_dir / "calibrator.pkl", "wb") as f:
            pickle.dump(scaler, f)
        with open(output_dir / "calibration_metrics.json", "w") as f:
            json.dump(cal_quality, f, indent=2)
        print(f"Calibrator saved to {output_dir / 'calibrator.pkl'}")

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a single AIGC detection tower.")
    parser.add_argument("--backbone", choices=["clip_h", "dino_h"], required=True)
    parser.add_argument("--data", required=True, help="Dataset root with train/val/calibration splits.")
    parser.add_argument("--output", required=True, help="Output directory for checkpoints.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01, dest="weight_decay")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    return parser.parse_args()


if __name__ == "__main__":
    train(_parse_args())
