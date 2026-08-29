"""Distributed/single-GPU training entry point for AIGC detection."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm

from calibration.temperature_scaling import TemperatureScaler
from data.dataset import AIGCDataset
from data.transforms import build_train_augment


def distributed_setup():
    is_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_ddp:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        return True, rank, local_rank, world
    return False, 0, 0, 1


def cleanup(is_ddp):
    if is_ddp:
        dist.destroy_process_group()


def _clip_preprocess(img_size=224):
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(img_size),
        T.ToTensor(), T.Normalize((0.48145466, 0.4578275, 0.40821073),
                                   (0.26862954, 0.26130258, 0.27577711)),
    ])


def _dino_preprocess(img_size=224):
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(img_size),
        T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def _build_model(backbone, device):
    if backbone == "clip_h":
        from models.clip_tower import CLIPTower
        model = CLIPTower(unfreeze_blocks=4, proj_dim=512, dropout=0.1)
    elif backbone == "dino_h":
        from models.dino_tower import DINOTower
        model = DINOTower(unfreeze_blocks=4, proj_dim=512, dropout=0.1)
    else:
        raise ValueError(backbone)
    return model.to(device)


def _dataset(root, backbone, augment):
    base = _clip_preprocess() if backbone == "clip_h" else _dino_preprocess()
    if augment:
        aug = build_train_augment(clean_prob=0.3)
        transform = lambda img: base(aug(img))
    else:
        transform = base
    return AIGCDataset(root, transform=transform)


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    logits, labels = [], []
    for images, y in loader:
        logits.append(model(images.to(device, non_blocking=True)).cpu())
        labels.append(y)
    x = torch.cat(logits).numpy()
    y = torch.cat(labels).numpy()
    p = torch.sigmoid(torch.from_numpy(x)).numpy()
    return {
        "auc": float(roc_auc_score(y, p)),
        "loss": float(nn.functional.binary_cross_entropy_with_logits(
            torch.from_numpy(x), torch.from_numpy(y).float())),
        "logits": x, "labels": y,
    }


def train(args):
    is_ddp, rank, local_rank, world = distributed_setup()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    master = rank == 0
    if master:
        print(f"Device: {device} | world_size: {world}")

    root, out = Path(args.data), Path(args.output)
    if master:
        out.mkdir(parents=True, exist_ok=True)
    if is_ddp:
        dist.barrier()

    train_ds = _dataset(root / "train", args.backbone, True)
    val_ds = _dataset(root / "val", args.backbone, False)
    cal_ds = _dataset(root / "calibration", args.backbone, False) if (root / "calibration").is_dir() else None
    if master:
        print(f"Train: {len(train_ds)} {train_ds.class_counts()}")
        print(f"Val:   {len(val_ds)} {val_ds.class_counts()}")

    train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True) if is_ddp else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=train_sampler is None, sampler=train_sampler,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    cal_loader = DataLoader(cal_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=args.workers, pin_memory=True) if cal_ds else None

    model = _build_model(args.backbone, device)
    if master:
        print(f"Params: {model.param_count()}")
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    raw_model = model.module if is_ddp else model

    counts = train_ds.class_counts()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        [counts.get(0, 1) / counts.get(1, 1)], device=device))
    optimizer = torch.optim.AdamW(raw_model.trainable_parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    writer = SummaryWriter(str(out / "tensorboard")) if master else None
    best_auc, history, global_step = 0.0, [], 0

    for epoch in range(1, args.epochs + 1):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        model.train(); start = time.time(); running = 0.0
        it = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False) if master else train_loader
        for images, labels in it:
            images = images.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
            optimizer.step(); running += loss.item()
            if master:
                writer.add_scalar("train/step_loss", loss.item(), global_step)
                global_step += 1
        if is_ddp:
            dist.barrier()
        scheduler.step()
        if master:
            vm = _evaluate(raw_model, val_loader, device)
            elapsed = time.time() - start
            train_loss = running / max(1, len(train_loader))
            lr = scheduler.get_last_lr()[0]
            writer.add_scalars("loss", {"train": train_loss, "val": vm["loss"]}, epoch)
            writer.add_scalar("val/auc", vm["auc"], epoch)
            writer.add_scalar("train/lr", lr, epoch)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": vm["loss"],
                            "val_auc": vm["auc"], "lr": lr, "elapsed_s": round(elapsed, 1)})
            print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} val_loss={vm['loss']:.4f} val_auc={vm['auc']:.4f} ({elapsed:.0f}s)")
            if vm["auc"] > best_auc:
                best_auc = vm["auc"]
                torch.save(raw_model.state_dict(), out / "best.pt")
                print(f"  Saved best model (val_auc={best_auc:.4f})")
        if is_ddp:
            dist.barrier()

    if master:
        writer.close()
        print(f"\nBest val AUC: {best_auc:.4f}")
        (out / "history.json").write_text(json.dumps(history, indent=2))
        if cal_loader:
            print("Fitting temperature scaling on calibration split ...")
            cal = _evaluate(raw_model, cal_loader, device)
            scaler = TemperatureScaler(); scaler.fit(cal["logits"], cal["labels"])
            quality = scaler.calibration_metrics(cal["logits"], cal["labels"])
            print(f"Temperature: {quality['temperature']:.4f}  ECE: {quality['ece']:.4f}  Brier: {quality['brier']:.4f}")
            with open(out / "calibrator.pkl", "wb") as f: pickle.dump(scaler, f)
            (out / "calibration_metrics.json").write_text(json.dumps(quality, indent=2))
    if is_ddp:
        dist.barrier()
    cleanup(is_ddp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=["clip_h", "dino_h"], required=True)
    p.add_argument("--data", required=True); p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=10); p.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--weight-decay", type=float, default=0.01, dest="weight_decay")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    train(p.parse_args())


if __name__ == "__main__": main()
