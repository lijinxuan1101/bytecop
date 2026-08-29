"""Stage 1 training script — CIFAKE baseline for OpenCLIP single tower.

Reads a YAML config from ``configs/`` and trains a single experiment (S1/S2/S3).
Uses parameter-group optimization so the head and (optionally unfrozen) backbone
blocks can have different learning rates.

Usage
-----
    python experiments/stage1/train.py --config experiments/stage1/configs/s1_linear_probe.yaml
    python experiments/stage1/train.py --config <path> --data <alt-data-root>
    python experiments/stage1/train.py --config <path> --output <alt-output-dir>

Output layout: runs/stage1/<name>/
    best.pt                 (best checkpoint by val AUC)
    calibrator.pkl          (temperature-scaled calibrator)
    history.json            (per-epoch train/val metrics)
    calibration_metrics.json
    tensorboard/            (TensorBoard event files)
    config.yaml             (frozen copy of the config used for this run)
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm

# Add project root to sys.path so we can import shared modules
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from calibration.temperature_scaling import TemperatureScaler  # noqa: E402
from data.dataset import AIGCDataset  # noqa: E402
from data.transforms import build_train_augment  # noqa: E402
from models.clip_tower import CLIPTower  # noqa: E402


# ------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------

_REQUIRED_KEYS = {
    "name", "backbone",
    "unfreeze_blocks", "proj_dim", "dropout",
    "dataset", "img_size", "augment",
    "epochs", "batch_size",
    "head_lr", "backbone_lr", "weight_decay", "grad_clip",
    "workers",
}


def _load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    missing = _REQUIRED_KEYS - set(cfg)
    if missing:
        sys.exit(f"Config {path} missing keys: {sorted(missing)}")
    if cfg["backbone"] != "clip_h":
        sys.exit(f"Stage 1 only supports backbone=clip_h, got {cfg['backbone']!r}")
    return cfg


# ------------------------------------------------------------------
# Pre-processing
# ------------------------------------------------------------------

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


def _tensor_transform(img_size: int) -> T.Compose:
    """Standard CLIP normalisation. CIFAKE (32x32) is upsampled here."""
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
    ])


def _train_transform(img_size: int, augment: bool, clean_prob: float = 0.3) -> callable:
    tensor_tfm = _tensor_transform(img_size)
    if not augment:
        return tensor_tfm
    if not 0.0 <= clean_prob <= 1.0:
        raise ValueError(f"clean_prob must be in [0, 1], got {clean_prob}")
    pil_aug = build_train_augment(clean_prob=clean_prob)

    def _t(img):
        return tensor_tfm(pil_aug(img))
    return _t


# ------------------------------------------------------------------
# Model + optimizer
# ------------------------------------------------------------------

def _build_model(cfg: dict, device: torch.device) -> CLIPTower:
    model = CLIPTower(
        unfreeze_blocks=cfg["unfreeze_blocks"],
        proj_dim=cfg["proj_dim"],
        dropout=cfg["dropout"],
    )
    return model.to(device)


def _param_groups(model: CLIPTower, cfg: dict) -> list[dict]:
    """Split trainable params into backbone vs head groups with distinct lrs."""
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("backbone.")
    ]

    groups = [{"params": head_params, "lr": cfg["head_lr"], "name": "head"}]
    if backbone_params:
        groups.append({
            "params": backbone_params,
            "lr": cfg["backbone_lr"],
            "name": "backbone",
        })
    return groups


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

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
    auc = float(roc_auc_score(labels_np, probs_np))
    loss = float(nn.functional.binary_cross_entropy_with_logits(
        torch.from_numpy(logits_np),
        torch.from_numpy(labels_np).float(),
    ))
    return {"auc": auc, "loss": loss, "logits": logits_np, "labels": labels_np}


# ------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    cfg = _load_config(Path(args.config))
    if args.clean_prob is not None:
        cfg["clean_prob"] = args.clean_prob
    cfg.setdefault("clean_prob", 0.3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Freeze a copy of the config next to the outputs
    shutil.copy(args.config, output_dir / "config.yaml")

    print(f"[stage1/{cfg['name']}]  device={device}")
    print(f"  data   : {data_root}")
    print(f"  output : {output_dir}")
    print(f"  augment: {cfg['augment']}  clean_prob: {cfg['clean_prob']}")

    # --------------------------------------------------------------
    # Data
    # --------------------------------------------------------------
    train_tfm = _train_transform(
        cfg["img_size"], cfg["augment"], cfg["clean_prob"]
    )
    eval_tfm  = _tensor_transform(cfg["img_size"])

    train_ds = AIGCDataset(data_root / "train",       transform=train_tfm)
    val_ds   = AIGCDataset(data_root / "val",         transform=eval_tfm)
    cal_split = data_root / "calibration"
    has_cal = cal_split.is_dir()
    cal_ds = AIGCDataset(cal_split, transform=eval_tfm) if has_cal else None

    print(f"  train  : {len(train_ds)} samples  {train_ds.class_counts()}")
    print(f"  val    : {len(val_ds)} samples  {val_ds.class_counts()}")
    if cal_ds is not None:
        print(f"  cal    : {len(cal_ds)} samples  {cal_ds.class_counts()}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
        num_workers=cfg["workers"], pin_memory=True,
    )
    cal_loader = DataLoader(
        cal_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
        num_workers=cfg["workers"], pin_memory=True,
    ) if cal_ds is not None else None

    # --------------------------------------------------------------
    # Model + optimizer + scheduler + loss
    # --------------------------------------------------------------
    model = _build_model(cfg, device)
    counts = model.param_count()
    print(f"  params : total={counts['total']:,}  trainable={counts['trainable']:,}")

    groups = _param_groups(model, cfg)
    for g in groups:
        print(f"    group[{g['name']}]: lr={g['lr']:.1e}  n={sum(p.numel() for p in g['params']):,}")

    optimizer = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    class_counts = train_ds.class_counts()
    n_real = class_counts.get(0, 1)
    n_fake = class_counts.get(1, 1)
    pos_weight = torch.tensor([n_real / n_fake], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --------------------------------------------------------------
    # TensorBoard
    # --------------------------------------------------------------
    tb_dir = output_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"  tb     : tensorboard --logdir {tb_dir}")

    # --------------------------------------------------------------
    # Training loop
    # --------------------------------------------------------------
    best_auc = 0.0
    history: list[dict] = []
    global_step = 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for images, labels in tqdm(
            train_loader, desc=f"epoch {epoch}/{cfg['epochs']}", leave=False,
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()

            running_loss += loss.item()
            writer.add_scalar("train/step_loss", loss.item(), global_step)
            global_step += 1

        scheduler.step()

        train_loss = running_loss / len(train_loader)
        val_metrics = _evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        lr_head = optimizer.param_groups[0]["lr"]

        writer.add_scalars("loss", {"train": train_loss, "val": val_metrics["loss"]}, epoch)
        writer.add_scalar("val/auc",  val_metrics["auc"], epoch)
        writer.add_scalar("train/lr_head", lr_head, epoch)
        if len(optimizer.param_groups) > 1:
            writer.add_scalar("train/lr_backbone", optimizer.param_groups[1]["lr"], epoch)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_auc": val_metrics["auc"],
            "lr_head": lr_head,
            "elapsed_s": round(elapsed, 1),
        }
        history.append(record)
        print(
            f"  epoch {epoch:3d} | train={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_auc={val_metrics['auc']:.4f} "
            f"({elapsed:.0f}s)"
        )

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save(model.state_dict(), output_dir / "best.pt")
            print(f"    saved best.pt (val_auc={best_auc:.4f})")

    writer.add_hparams(
        hparam_dict={
            "name": cfg["name"],
            "unfreeze_blocks": cfg["unfreeze_blocks"],
            "head_lr": cfg["head_lr"],
            "backbone_lr": cfg["backbone_lr"],
            "batch_size": cfg["batch_size"],
            "epochs": cfg["epochs"],
            "augment": cfg["augment"],
            "clean_prob": cfg["clean_prob"],
        },
        metric_dict={"hparam/best_val_auc": best_auc},
    )
    writer.close()

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  best val AUC: {best_auc:.4f}")

    # --------------------------------------------------------------
    # Temperature scaling on calibration split
    # --------------------------------------------------------------
    if cal_loader is None:
        print("  no calibration split found, skipping temperature scaling.")
        return

    print("\n  fitting temperature scaling on calibration split ...")
    model.load_state_dict(torch.load(
        output_dir / "best.pt", map_location=device, weights_only=True,
    ))
    cal_metrics = _evaluate(model, cal_loader, device)
    scaler = TemperatureScaler()
    scaler.fit(cal_metrics["logits"], cal_metrics["labels"])
    quality = scaler.calibration_metrics(cal_metrics["logits"], cal_metrics["labels"])
    print(f"    T   = {quality['temperature']:.4f}")
    print(f"    ECE = {quality['ece']:.4f}")
    print(f"    Bri = {quality['brier']:.4f}")

    with open(output_dir / "calibrator.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(output_dir / "calibration_metrics.json", "w") as f:
        json.dump(quality, f, indent=2)
    print(f"  saved calibrator.pkl")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1 training (CIFAKE baseline).")
    p.add_argument("--config", required=True, help="Path to a stage 1 YAML config.")
    p.add_argument("--data", default="data/datasets/CIFAKE_images",
                   help="Dataset root with train/val/test/calibration subfolders.")
    p.add_argument("--output", default=None,
                   help="Output directory. Defaults to runs/stage1/<config.name>/.")
    p.add_argument(
        "--clean-prob", type=float, default=None,
        help="Probability of clean training samples; overrides config clean_prob.",
    )
    args = p.parse_args()

    if args.output is None:
        cfg_name = yaml.safe_load(open(args.config))["name"]
        args.output = f"runs/stage1/{cfg_name}"
    return args


if __name__ == "__main__":
    train(_parse_args())
