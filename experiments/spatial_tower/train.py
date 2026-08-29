"""Train the OpenCLIP-H spatial tower.

Single GPU:
    python experiments/spatial_tower/train.py \\
        --config experiments/spatial_tower/configs/spatial_tower.yaml

Multi GPU (torchrun):
    torchrun --nproc_per_node=4 experiments/spatial_tower/train.py \\
        --config experiments/spatial_tower/configs/spatial_tower.yaml

``batch_size`` in the config is per GPU. Effective global batch is
``batch_size × world_size × grad_accum_steps``.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from calibration.temperature_scaling import TemperatureScaler  # noqa: E402
from data.dataset import AIGCDataset  # noqa: E402
from data.transforms import build_train_augment  # noqa: E402
from experiments.common.amp import autocast_ctx, bf16_enabled  # noqa: E402
from experiments.common.checkpoint import MetricMonitor  # noqa: E402
from experiments.common.distributed import (  # noqa: E402
    barrier,
    cleanup,
    init_distributed,
    reduce_mean,
    unwrap,
    wrap_ddp,
)
from experiments.common.schedulers import build_warmup_cosine  # noqa: E402
from models.clip_tower import CLIPTower  # noqa: E402


_REQUIRED_KEYS = {
    "name", "backbone",
    "unfreeze_blocks", "proj_dim", "dropout",
    "dataset", "img_size", "augment",
    "epochs", "batch_size",
    "head_lr", "backbone_lr", "weight_decay", "grad_clip",
    "workers",
}

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    missing = _REQUIRED_KEYS - set(cfg)
    if missing:
        sys.exit(f"Config {path} missing keys: {sorted(missing)}")
    if cfg["backbone"] != "clip_h":
        sys.exit(f"Spatial tower only supports backbone=clip_h, got {cfg['backbone']!r}")
    cfg.setdefault("ddp", True)
    cfg.setdefault("num_gpus", 4)
    cfg.setdefault("find_unused_parameters", False)
    cfg.setdefault("scheduler", "cosine")
    cfg.setdefault("warmup_ratio", 0.05)
    cfg.setdefault("grad_accum_steps", 1)
    cfg.setdefault("bf16", False)
    cfg.setdefault("grad_checkpointing", False)
    cfg.setdefault("monitor", "val_auc")
    cfg.setdefault("clean_prob", 0.3)
    if cfg["grad_accum_steps"] < 1:
        sys.exit(f"grad_accum_steps must be >= 1, got {cfg['grad_accum_steps']}")
    return cfg


def _tensor_transform(img_size: int) -> T.Compose:
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


def _build_model(cfg: dict, device: torch.device) -> CLIPTower:
    model = CLIPTower(
        unfreeze_blocks=cfg["unfreeze_blocks"],
        proj_dim=cfg["proj_dim"],
        dropout=cfg["dropout"],
    )
    if cfg["grad_checkpointing"]:
        model.set_grad_checkpointing(True)
    return model.to(device)


def _param_groups(model: CLIPTower, cfg: dict) -> list[dict]:
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


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_bf16: bool,
) -> dict:
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        with autocast_ctx(use_bf16):
            logits = model(images.to(device))
        all_logits.append(logits.float().cpu())
        all_labels.append(labels)
    logits_np = torch.cat(all_logits).numpy()
    labels_np = torch.cat(all_labels).numpy()
    auc = float(roc_auc_score(labels_np, torch.sigmoid(torch.from_numpy(logits_np)).numpy()))
    loss = float(nn.functional.binary_cross_entropy_with_logits(
        torch.from_numpy(logits_np),
        torch.from_numpy(labels_np).float(),
    ))
    return {"auc": auc, "loss": loss, "logits": logits_np, "labels": labels_np}


def train(args: argparse.Namespace) -> None:
    cfg = _load_config(Path(args.config))
    if args.clean_prob is not None:
        cfg["clean_prob"] = args.clean_prob
    dist_info = init_distributed()
    device = dist_info.device
    use_bf16 = bf16_enabled(cfg, device)
    accum = cfg["grad_accum_steps"]

    data_root = Path(args.data)
    output_dir = Path(args.output)
    if dist_info.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.config, output_dir / "config.yaml")
    barrier(dist_info)

    global_batch = cfg["batch_size"] * dist_info.world_size * accum
    if dist_info.is_main:
        print(f"[spatial_tower/{cfg['name']}]  device={device}  "
              f"world_size={dist_info.world_size}  ddp={dist_info.enabled}")
        print(f"  data   : {data_root}")
        print(f"  output : {output_dir}")
        print(f"  batch  : {cfg['batch_size']} per GPU  accum={accum}  "
              f"(global {global_batch})")
        print(f"  amp    : bf16={use_bf16}  grad_checkpointing={cfg['grad_checkpointing']}")
        print(f"  augment: {cfg['augment']}  clean_prob: {cfg['clean_prob']}")

    train_tfm = _train_transform(cfg["img_size"], cfg["augment"], cfg["clean_prob"])
    eval_tfm = _tensor_transform(cfg["img_size"])

    train_ds = AIGCDataset(data_root / "train", transform=train_tfm)
    val_ds = AIGCDataset(data_root / "val", transform=eval_tfm)
    cal_split = data_root / "calibration"
    cal_ds = AIGCDataset(cal_split, transform=eval_tfm) if cal_split.is_dir() else None

    if dist_info.is_main:
        print(f"  train  : {len(train_ds)} samples  {train_ds.class_counts()}")
        print(f"  val    : {len(val_ds)} samples  {val_ds.class_counts()}")
        if cal_ds is not None:
            print(f"  cal    : {len(cal_ds)} samples  {cal_ds.class_counts()}")

    train_sampler = (
        DistributedSampler(train_ds, shuffle=True) if dist_info.enabled else None
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"],
        shuffle=train_sampler is None,
        sampler=train_sampler,
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

    raw_model = _build_model(cfg, device)
    if dist_info.is_main:
        counts = raw_model.param_count()
        print(f"  params : total={counts['total']:,}  trainable={counts['trainable']:,}")

    groups = _param_groups(raw_model, cfg)
    if dist_info.is_main:
        for g in groups:
            print(f"    group[{g['name']}]: lr={g['lr']:.1e}  "
                  f"n={sum(p.numel() for p in g['params']):,}")

    model = wrap_ddp(
        raw_model, dist_info,
        find_unused_parameters=cfg["find_unused_parameters"],
    )
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    total_steps = cfg["epochs"] * steps_per_epoch
    scheduler, warmup_steps = build_warmup_cosine(
        optimizer, total_steps=total_steps, warmup_ratio=cfg["warmup_ratio"],
    )
    if dist_info.is_main:
        print(f"  sched  : {cfg['scheduler']}  warmup_ratio={cfg['warmup_ratio']}  "
              f"warmup_steps={warmup_steps}/{total_steps}")
        print(f"  monitor: {cfg['monitor']} → {output_dir / 'best.pt'}")

    class_counts = train_ds.class_counts()
    pos_weight = torch.tensor(
        [class_counts.get(0, 1) / class_counts.get(1, 1)], device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard")) if dist_info.is_main else None
    monitor = MetricMonitor(output_dir, metric=cfg["monitor"]) if dist_info.is_main else None
    history: list[dict] = []
    global_step = 0

    try:
        for epoch in range(1, cfg["epochs"] + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            t0 = time.time()
            running_loss = 0.0
            optimizer.zero_grad(set_to_none=True)

            iterator = tqdm(
                train_loader, desc=f"epoch {epoch}/{cfg['epochs']}", leave=False,
                disable=not dist_info.is_main,
            )
            for micro_idx, (images, labels) in enumerate(iterator, start=1):
                images = images.to(device, non_blocking=True)
                labels = labels.float().to(device, non_blocking=True)
                with autocast_ctx(use_bf16):
                    loss = criterion(model(images), labels) / accum
                loss.backward()
                running_loss += loss.item() * accum

                stepped = micro_idx % accum == 0 or micro_idx == len(train_loader)
                if stepped:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    if writer is not None:
                        writer.add_scalar("train/step_loss", loss.item() * accum, global_step)
                        writer.add_scalar(
                            "train/lr_head", optimizer.param_groups[0]["lr"], global_step,
                        )
                    global_step += 1

            train_loss = reduce_mean(running_loss / len(train_loader), dist_info)

            if dist_info.is_main:
                val_metrics = _evaluate(unwrap(model), val_loader, device, use_bf16=use_bf16)
                elapsed = time.time() - t0
                lr_head = optimizer.param_groups[0]["lr"]
                record = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_metrics["loss"],
                    "val_auc": val_metrics["auc"],
                    "lr_head": lr_head,
                    "elapsed_s": round(elapsed, 1),
                }
                writer.add_scalars(
                    "loss", {"train": train_loss, "val": val_metrics["loss"]}, epoch,
                )
                writer.add_scalar("val/auc", val_metrics["auc"], epoch)
                writer.add_scalar("train/lr_head", lr_head, epoch)
                history.append(record)
                is_best = monitor.update(epoch, record, unwrap(model))
                mark = "  *best*" if is_best else ""
                print(
                    f"  epoch {epoch:3d} | train={train_loss:.4f} "
                    f"val_loss={val_metrics['loss']:.4f} val_auc={val_metrics['auc']:.4f} "
                    f"({elapsed:.0f}s){mark}"
                )
                if is_best:
                    print(f"    saved best.pt (val_auc={monitor.best:.4f})")
            barrier(dist_info)

        if dist_info.is_main:
            if writer is not None:
                writer.close()
            with open(output_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)
            print(f"\n  best val AUC: {monitor.best:.4f}  (epoch {monitor.best_epoch})")

            if cal_loader is None:
                print("  no calibration split found, skipping temperature scaling.")
                return

            print("\n  fitting temperature scaling on calibration split ...")
            unwrap(model).load_state_dict(torch.load(
                output_dir / "best.pt", map_location=device, weights_only=True,
            ))
            cal_metrics = _evaluate(unwrap(model), cal_loader, device, use_bf16=use_bf16)
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
        barrier(dist_info)
    finally:
        cleanup(dist_info)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the OpenCLIP-H spatial tower.")
    p.add_argument("--config", required=True)
    p.add_argument("--data", default="data/datasets/SID_Set_images")
    p.add_argument("--output", default=None)
    p.add_argument("--clean-prob", type=float, default=None)
    args = p.parse_args()
    if args.output is None:
        cfg_name = yaml.safe_load(open(args.config))["name"]
        args.output = f"runs/spatial_tower/{cfg_name}"
    return args


if __name__ == "__main__":
    train(_parse_args())
