"""Stage 2 training — RGPA forensic branch.

Reads a YAML config and trains one RGPA experiment. Input is pixel-scale RGB
(no CLIP / ImageNet normalize). Training uses the official probabilistic
single-degradation policy. SRM-inspired high-pass kernels stay frozen.

Usage
-----
    python experiments/stage2/train.py \\
        --config experiments/stage2/configs/rgpa_p50.yaml
    python experiments/stage2/train.py --config <path> --data <alt-data-root>

Output layout: runs/stage2/<name>/
    best.pt
    calibrator.pkl
    history.json
    calibration_metrics.json
    val_predictions.json     (paths, labels, forensic logits for Stage 3)
    aggregation_stats.json   (high/low weight divergence)
    tensorboard/
    config.yaml
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from calibration.temperature_scaling import TemperatureScaler  # noqa: E402
from data.dataset import AIGCDataset  # noqa: E402
from data.transforms import build_train_augment  # noqa: E402
from models.rgpa import RGPA  # noqa: E402


_REQUIRED_KEYS = {
    "name", "backbone",
    "embed_dim", "dropout", "tau",
    "dataset", "img_size", "augment", "clean_prob",
    "epochs", "batch_size", "lr", "weight_decay", "grad_clip",
    "workers",
}


def _load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    missing = _REQUIRED_KEYS - set(cfg)
    if missing:
        sys.exit(f"Config {path} missing keys: {sorted(missing)}")
    if cfg["backbone"] != "rgpa":
        sys.exit(f"Stage 2 only supports backbone=rgpa, got {cfg['backbone']!r}")
    return cfg


def _tensor_transform(img_size: int) -> T.Compose:
    """Geometry only. Forensic branch uses pixel-scale RGB in [0, 1]."""
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
    ])


def _train_transform(img_size: int, augment: bool, clean_prob: float) -> callable:
    tensor_tfm = _tensor_transform(img_size)
    if not augment:
        return tensor_tfm
    pil_aug = build_train_augment(clean_prob=clean_prob)

    def _t(img):
        return tensor_tfm(pil_aug(img))
    return _t


def _build_model(cfg: dict, device: torch.device) -> nn.Module:
    model = RGPA(
        embed_dim=cfg["embed_dim"],
        dropout=cfg["dropout"],
        tau=cfg["tau"],
    )
    return model.to(device)


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
    probs_np = torch.sigmoid(torch.from_numpy(logits_np)).numpy()
    auc = float(roc_auc_score(labels_np, probs_np))
    loss = float(nn.functional.binary_cross_entropy_with_logits(
        torch.from_numpy(logits_np),
        torch.from_numpy(labels_np).float(),
    ))
    return {"auc": auc, "loss": loss, "logits": logits_np, "labels": labels_np}


class _PathDataset(Dataset):
    """Wrap AIGCDataset so a DataLoader can also yield image paths."""

    def __init__(self, base: AIGCDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image, label = self.base[idx]
        return image, label, str(self.base.samples[idx][0])


@torch.no_grad()
def _dump_val_predictions(
    model: nn.Module,
    val_ds: AIGCDataset,
    device: torch.device,
    output_dir: Path,
    *,
    batch_size: int,
    workers: int,
    dump_weights: bool,
) -> None:
    """Write forensic logits (and RGPA aggregation stats) on the val split."""
    model.eval()
    loader = DataLoader(
        _PathDataset(val_ds),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    records: list[dict] = []
    weight_batches: list[tuple[torch.Tensor, torch.Tensor]] = []

    for images, labels, paths in loader:
        images = images.to(device, non_blocking=True)
        if dump_weights:
            z_forensic, w_high, w_low = model.forward_features(images)
            logits = model.head(z_forensic).squeeze(1)
            weight_batches.append((w_high.cpu(), w_low.cpu()))
        else:
            logits = model(images)
        for path, label, logit in zip(paths, labels.tolist(), logits.cpu().tolist()):
            records.append({
                "image_path": path,
                "label": int(label),
                "logit": round(float(logit), 6),
            })

    with open(output_dir / "val_predictions.json", "w") as f:
        json.dump(records, f, indent=2)

    if dump_weights and weight_batches:
        w_high = torch.cat([w[0] for w in weight_batches])
        w_low = torch.cat([w[1] for w in weight_batches])
        stats = RGPA.aggregation_stats(w_high, w_low)
        stats["n"] = int(w_high.shape[0])
        with open(output_dir / "aggregation_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        print(
            f"    aggregation  L1(high,low)={stats['mean_l1_high_low']:.4f}  "
            f"H_high={stats['mean_entropy_high']:.3f}  "
            f"H_low={stats['mean_entropy_low']:.3f}"
        )


def train(args: argparse.Namespace) -> None:
    cfg = _load_config(Path(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, output_dir / "config.yaml")

    print(f"[stage2/{cfg['name']}]  device={device}  backbone={cfg['backbone']}")
    print(f"  data   : {data_root}")
    print(f"  output : {output_dir}")
    print(f"  mix    : clean_prob={cfg['clean_prob']}  augment={cfg['augment']}")

    train_tfm = _train_transform(cfg["img_size"], cfg["augment"], cfg["clean_prob"])
    eval_tfm = _tensor_transform(cfg["img_size"])

    train_ds = AIGCDataset(data_root / "train", transform=train_tfm)
    val_ds = AIGCDataset(data_root / "val", transform=eval_tfm)
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

    model = _build_model(cfg, device)
    counts = model.param_count()
    print(f"  params : total={counts['total']:,}  trainable={counts['trainable']:,}")

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"],
    )

    class_counts = train_ds.class_counts()
    n_real = class_counts.get(0, 1)
    n_fake = class_counts.get(1, 1)
    pos_weight = torch.tensor([n_real / n_fake], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    tb_dir = output_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"  tb     : tensorboard --logdir {tb_dir}")

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
        lr_now = optimizer.param_groups[0]["lr"]

        writer.add_scalars(
            "loss", {"train": train_loss, "val": val_metrics["loss"]}, epoch,
        )
        writer.add_scalar("val/auc", val_metrics["auc"], epoch)
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
            "backbone": cfg["backbone"],
            "clean_prob": cfg["clean_prob"],
            "lr": cfg["lr"],
            "batch_size": cfg["batch_size"],
            "epochs": cfg["epochs"],
        },
        metric_dict={"hparam/best_val_auc": best_auc},
    )
    writer.close()

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  best val AUC: {best_auc:.4f}")

    model.load_state_dict(torch.load(
        output_dir / "best.pt", map_location=device, weights_only=True,
    ))

    print("  dumping val forensic logits ...")
    _dump_val_predictions(
        model, val_ds, device, output_dir,
        batch_size=cfg["batch_size"] * 2,
        workers=cfg["workers"],
        dump_weights=True,
    )
    print("  saved val_predictions.json")

    if cal_loader is None:
        print("  no calibration split found, skipping temperature scaling.")
        return

    print("\n  fitting temperature scaling on calibration split ...")
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
    print("  saved calibrator.pkl")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2 training (RGPA).")
    p.add_argument("--config", required=True, help="Path to a stage 2 YAML config.")
    p.add_argument(
        "--data",
        default="data/datasets/SID_Set_images",
        help="Dataset root with train/val/calibration subfolders.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output directory. Defaults to runs/stage2/<config.name>/.",
    )
    args = p.parse_args()

    if args.output is None:
        cfg_name = yaml.safe_load(open(args.config))["name"]
        args.output = f"runs/stage2/{cfg_name}"
    return args


if __name__ == "__main__":
    train(_parse_args())
