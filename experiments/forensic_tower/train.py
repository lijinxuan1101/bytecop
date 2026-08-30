"""Train the RGPA forensic tower.

Input is pixel-scale RGB in ``[0, 1]`` (no OpenCLIP / ImageNet normalize).

Single GPU:
    python experiments/forensic_tower/train.py \\
        --config experiments/forensic_tower/configs/forensic_tower.yaml

Multi GPU (torchrun):
    torchrun --nproc_per_node=2 experiments/forensic_tower/train.py \\
        --config experiments/forensic_tower/configs/forensic_tower.yaml

``batch_size`` in the config is per GPU.
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
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from calibration.temperature_scaling import TemperatureScaler  # noqa: E402
from data.dataset import AIGCDataset, FlaggedAugmentDataset  # noqa: E402
from experiments.common.amp import autocast_ctx, bf16_enabled  # noqa: E402
from experiments.common.checkpoint import (  # noqa: E402
    LAST_NAME,
    MetricMonitor,
    last_positive_lr,
    load_checkpoint,
    resolve_resume_path,
    save_last,
)
from experiments.common.distributed import (  # noqa: E402
    DistInfo,
    barrier,
    broadcast_bool,
    broadcast_module,
    broadcast_object,
    cleanup,
    eval_sampler,
    gather_cat,
    init_distributed,
    reduce_mean,
    reduce_sum,
    unwrap,
    wrap_ddp,
)
from experiments.common.schedulers import build_warmup_cosine  # noqa: E402
from models.rgpa import RGPA  # noqa: E402


_REQUIRED_KEYS = {
    "name", "backbone",
    "embed_dim", "dropout", "tau",
    "dataset", "img_size", "augment", "clean_prob",
    "epochs", "batch_size", "lr", "weight_decay", "grad_clip",
    "workers",
}

_ALLOWED_NORMALIZE = {None, False, "none", "pixel"}


def _load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    missing = _REQUIRED_KEYS - set(cfg)
    if missing:
        sys.exit(f"Config {path} missing keys: {sorted(missing)}")
    if cfg["backbone"] != "rgpa":
        sys.exit(f"Forensic tower only supports backbone=rgpa, got {cfg['backbone']!r}")
    if cfg.get("normalize", "none") not in _ALLOWED_NORMALIZE:
        sys.exit(
            "RGPA must use pixel-scale RGB without OpenCLIP / ImageNet normalize "
            f"(got normalize={cfg.get('normalize')!r})"
        )
    cfg.setdefault("ddp", True)
    cfg.setdefault("num_gpus", 2)
    cfg.setdefault("find_unused_parameters", False)
    cfg.setdefault("scheduler", "cosine")
    cfg.setdefault("warmup_ratio", 0.05)
    cfg.setdefault("bf16", False)
    cfg.setdefault("patch_size", 32)
    cfg.setdefault("normalize", "none")
    cfg.setdefault("monitor", "val_auc")
    cfg.setdefault("early_stop_patience", 0)
    return cfg


def _tensor_transform(img_size: int) -> T.Compose:
    """Pixel-scale RGB in [0, 1]. Do not apply OpenCLIP normalize."""
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
    ])


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    dist_info: DistInfo | None = None,
    *,
    use_bf16: bool,
) -> dict:
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        images, labels = batch[0], batch[1]
        with autocast_ctx(use_bf16):
            logits = model(images.to(device))
        all_logits.append(logits.float())
        all_labels.append(labels.to(device))
    local_logits = torch.cat(all_logits)
    local_labels = torch.cat(all_labels).float()
    if dist_info is not None and dist_info.enabled:
        logits = gather_cat(local_logits, dist_info)
        labels = gather_cat(local_labels, dist_info)
    else:
        logits = local_logits
        labels = local_labels
    logits_np = logits.cpu().numpy()
    labels_np = labels.cpu().numpy()
    auc = float(roc_auc_score(labels_np, torch.sigmoid(logits.cpu()).numpy()))
    loss = float(nn.functional.binary_cross_entropy_with_logits(logits.cpu(), labels.cpu()))
    return {"auc": auc, "loss": loss, "logits": logits_np, "labels": labels_np}


def _split_plain_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    is_clean: torch.Tensor,
) -> tuple[float, int, float, int]:
    """Plain BCE sums for clean vs robust samples (no pos_weight)."""
    per = nn.functional.binary_cross_entropy_with_logits(
        logits.float(), labels, reduction="none",
    )
    clean = is_clean.bool()
    robust = ~clean
    clean_sum = float(per[clean].sum()) if clean.any() else 0.0
    robust_sum = float(per[robust].sum()) if robust.any() else 0.0
    return clean_sum, int(clean.sum()), robust_sum, int(robust.sum())


def _safe_mean(total: float, count: float) -> float | None:
    return None if count <= 0 else total / count


class _PathDataset(Dataset):
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
    use_bf16: bool,
) -> None:
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
        with autocast_ctx(use_bf16):
            z_forensic, w_high, w_low = model.forward_features(images)
            logits = model.head(z_forensic).squeeze(1)
        weight_batches.append((w_high.float().cpu(), w_low.float().cpu()))
        for path, label, logit in zip(paths, labels.tolist(), logits.float().cpu().tolist()):
            records.append({
                "image_path": path,
                "label": int(label),
                "logit": round(float(logit), 6),
            })

    with open(output_dir / "val_predictions.json", "w") as f:
        json.dump(records, f, indent=2)

    if weight_batches:
        w_high = torch.cat([w[0] for w in weight_batches])
        w_low = torch.cat([w[1] for w in weight_batches])
        stats = RGPA.aggregation_stats(w_high, w_low)
        stats["n"] = int(w_high.shape[0])
        with open(output_dir / "aggregation_stats.json", "w") as f:
            json.dump(stats, f, indent=2)


def train(args: argparse.Namespace) -> None:
    cfg = _load_config(Path(args.config))
    dist_info = init_distributed()
    device = dist_info.device
    use_bf16 = bf16_enabled(cfg, device)

    data_root = Path(args.data)
    output_dir = Path(args.output)
    if dist_info.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.config, output_dir / "config.yaml")
    barrier(dist_info)

    if dist_info.is_main:
        print(f"[forensic_tower/{cfg['name']}]  device={device}  "
              f"world_size={dist_info.world_size}  ddp={dist_info.enabled}")
        print(f"  data   : {data_root}")
        print(f"  output : {output_dir}")
        print(f"  batch  : {cfg['batch_size']} per GPU  "
              f"(global {cfg['batch_size'] * dist_info.world_size})")
        print(f"  input  : pixel RGB  normalize={cfg['normalize']}  "
              f"patch_size={cfg['patch_size']}")
        print(f"  amp    : bf16={use_bf16}")
        print(f"  mix    : clean_prob={cfg['clean_prob']}  augment={cfg['augment']}")

    eval_tfm = _tensor_transform(cfg["img_size"])
    train_ds = FlaggedAugmentDataset(
        data_root / "train",
        tensor_transform=eval_tfm,
        augment=cfg["augment"],
        clean_prob=cfg["clean_prob"],
    )
    val_ds = AIGCDataset(data_root / "val", transform=eval_tfm)
    val_robust_ds = FlaggedAugmentDataset(
        data_root / "val",
        tensor_transform=eval_tfm,
        augment=True,
        clean_prob=0.0,
    )
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
        sampler=eval_sampler(len(val_ds), dist_info),
        num_workers=cfg["workers"], pin_memory=True,
    )
    val_robust_loader = DataLoader(
        val_robust_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
        sampler=eval_sampler(len(val_robust_ds), dist_info),
        num_workers=cfg["workers"], pin_memory=True,
    )
    cal_loader = DataLoader(
        cal_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
        num_workers=cfg["workers"], pin_memory=True,
    ) if cal_ds is not None else None

    raw_model = RGPA(
        embed_dim=cfg["embed_dim"],
        dropout=cfg["dropout"],
        tau=cfg["tau"],
        img_size=cfg["img_size"],
        patch_size=cfg["patch_size"],
    ).to(device)
    if dist_info.is_main:
        counts = raw_model.param_count()
        print(f"  params : total={counts['total']:,}  trainable={counts['trainable']:,}")
        if dist_info.enabled:
            print("  load   : rank 0 reads ckpt, then broadcast")

    start_epoch = 1
    end_epoch = cfg["epochs"]
    history: list[dict] = []
    global_step = 0
    opt_lr = cfg["lr"]
    constant_lr = False
    resume_payload: dict | None = None

    if args.resume is not None and dist_info.is_main:
        ckpt_path = resolve_resume_path(args.resume)
        resume_blob = load_checkpoint(ckpt_path, map_location=device)
        raw_model.load_state_dict(resume_blob["model"])
        if resume_blob.get("history"):
            history = list(resume_blob["history"])
        elif (output_dir / "history.json").is_file():
            with open(output_dir / "history.json") as f:
                history = json.load(f)
        if resume_blob.get("epoch") is not None:
            start_epoch = int(resume_blob["epoch"]) + 1
        elif history:
            start_epoch = int(history[-1]["epoch"]) + 1
        if resume_blob.get("global_step") is not None:
            global_step = int(resume_blob["global_step"])
        else:
            global_step = (start_epoch - 1) * len(train_loader)
        full_resume = resume_blob.get("optimizer") is not None
        if not full_resume:
            inherit_lr = last_positive_lr(history)
            if inherit_lr is None:
                inherit_lr = cfg["lr"]
            opt_lr = inherit_lr
            constant_lr = True
        resume_payload = {
            "ckpt_path": str(ckpt_path),
            "kind": "full (model+optim+sched)" if full_resume else "weights-only",
            "history": history,
            "start_epoch": start_epoch,
            "global_step": global_step,
            "constant_lr": constant_lr,
            "opt_lr": opt_lr,
            "optimizer": resume_blob.get("optimizer"),
            "scheduler": resume_blob.get("scheduler"),
            "monitor": resume_blob.get("monitor"),
        }

    broadcast_module(raw_model, dist_info)
    if args.resume is not None:
        resume_payload = broadcast_object(resume_payload, dist_info)
        history = list(resume_payload["history"])
        start_epoch = int(resume_payload["start_epoch"])
        global_step = int(resume_payload["global_step"])
        constant_lr = bool(resume_payload["constant_lr"])
        opt_lr = float(resume_payload["opt_lr"])
        if args.extra_epochs is not None:
            end_epoch = start_epoch + args.extra_epochs - 1
        if start_epoch > end_epoch:
            sys.exit(
                f"Nothing to train: start_epoch={start_epoch} > end_epoch={end_epoch}. "
                "Pass --extra-epochs N to continue past the original schedule."
            )
        if dist_info.is_main:
            print(f"  resume : {resume_payload['ckpt_path']}  ({resume_payload['kind']})")
            print(f"  epochs : {start_epoch} → {end_epoch}")
            if constant_lr:
                print(
                    f"  note   : last.pt was not saved; Adam moments reset. "
                    f"Using constant lr={opt_lr:.3e} (cosine already finished at 0)."
                )

    optimizer = torch.optim.AdamW(
        raw_model.trainable_parameters(),
        lr=opt_lr,
        weight_decay=cfg["weight_decay"],
    )
    if constant_lr:
        scheduler = LambdaLR(optimizer, lambda _: 1.0)
        warmup_steps = 0
        total_steps = args.extra_epochs or 0
    else:
        total_steps = cfg["epochs"] * len(train_loader)
        scheduler, warmup_steps = build_warmup_cosine(
            optimizer, total_steps=total_steps, warmup_ratio=cfg["warmup_ratio"],
        )
        if resume_payload is not None and resume_payload.get("optimizer") is not None:
            optimizer.load_state_dict(resume_payload["optimizer"])
            if resume_payload.get("scheduler") is not None:
                scheduler.load_state_dict(resume_payload["scheduler"])

    if dist_info.is_main:
        print(f"  sched  : {'constant' if constant_lr else cfg['scheduler']}  "
              f"warmup_ratio={0 if constant_lr else cfg['warmup_ratio']}  "
              f"warmup_steps={warmup_steps}/{total_steps}")
        print(f"  monitor: {cfg['monitor']}  early_stop_patience={cfg['early_stop_patience']}")
    model = wrap_ddp(
        raw_model, dist_info,
        find_unused_parameters=cfg["find_unused_parameters"],
    )

    class_counts = train_ds.class_counts()
    pos_weight = torch.tensor(
        [class_counts.get(0, 1) / class_counts.get(1, 1)], device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard")) if dist_info.is_main else None
    monitor = (
        MetricMonitor(
            output_dir,
            metric=cfg["monitor"],
            patience=cfg["early_stop_patience"],
        ) if dist_info.is_main else None
    )
    if dist_info.is_main and monitor is not None:
        if resume_payload is not None and resume_payload.get("monitor"):
            monitor.load_state_dict(resume_payload["monitor"])
        elif args.resume is not None:
            monitor.restore_from_run(output_dir, history)
    stopped_early = False

    try:
        for epoch in range(start_epoch, end_epoch + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            t0 = time.time()
            running_loss = 0.0
            clean_sum = robust_sum = 0.0
            clean_n = robust_n = 0

            iterator = tqdm(
                train_loader, desc=f"epoch {epoch}/{end_epoch}", leave=False,
                disable=not dist_info.is_main,
            )
            for images, labels, is_clean in iterator:
                images = images.to(device, non_blocking=True)
                labels = labels.float().to(device, non_blocking=True)
                is_clean = is_clean.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast_ctx(use_bf16):
                    logits = model(images)
                    loss = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()
                c_sum, c_n, r_sum, r_n = _split_plain_bce(logits.detach(), labels, is_clean)
                clean_sum += c_sum
                clean_n += c_n
                robust_sum += r_sum
                robust_n += r_n
                if writer is not None:
                    writer.add_scalar("train/step_loss", loss.item(), global_step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                global_step += 1

            train_loss = reduce_mean(running_loss / len(train_loader), dist_info)
            train_loss_clean = _safe_mean(
                reduce_sum(clean_sum, dist_info), reduce_sum(float(clean_n), dist_info),
            )
            train_loss_robust = _safe_mean(
                reduce_sum(robust_sum, dist_info), reduce_sum(float(robust_n), dist_info),
            )

            should_stop = False
            val_metrics = _evaluate(
                unwrap(model), val_loader, device, dist_info, use_bf16=use_bf16,
            )
            val_robust = _evaluate(
                unwrap(model), val_robust_loader, device, dist_info, use_bf16=use_bf16,
            )
            if dist_info.is_main:
                elapsed = time.time() - t0
                lr_now = optimizer.param_groups[0]["lr"]
                record = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_loss_clean": train_loss_clean,
                    "train_loss_robust": train_loss_robust,
                    "val_loss": val_metrics["loss"],
                    "val_loss_clean": val_metrics["loss"],
                    "val_loss_robust": val_robust["loss"],
                    "val_auc": val_metrics["auc"],
                    "val_auc_clean": val_metrics["auc"],
                    "val_auc_robust": val_robust["auc"],
                    "lr": lr_now,
                    "elapsed_s": round(elapsed, 1),
                }
                writer.add_scalars(
                    "loss", {"train": train_loss, "val": val_metrics["loss"]}, epoch,
                )
                if train_loss_clean is not None:
                    writer.add_scalar("train/loss_clean", train_loss_clean, epoch)
                if train_loss_robust is not None:
                    writer.add_scalar("train/loss_robust", train_loss_robust, epoch)
                writer.add_scalar("val/auc", val_metrics["auc"], epoch)
                writer.add_scalar("val/auc_robust", val_robust["auc"], epoch)
                writer.add_scalar("val/loss", val_metrics["loss"], epoch)
                writer.add_scalar("val/loss_clean", val_metrics["loss"], epoch)
                writer.add_scalar("val/loss_robust", val_robust["loss"], epoch)
                writer.add_scalars(
                    "loss_clean",
                    {"train": train_loss_clean or 0.0, "val": val_metrics["loss"]},
                    epoch,
                )
                writer.add_scalars(
                    "loss_robust",
                    {"train": train_loss_robust or 0.0, "val": val_robust["loss"]},
                    epoch,
                )
                writer.add_scalar("train/lr", lr_now, epoch)
                history.append(record)
                is_best = monitor.update(epoch, record, unwrap(model))
                mark = "  *best*" if is_best else ""
                def _fmt(v: float | None) -> str:
                    return "  n/a" if v is None else f"{v:.4f}"
                print(
                    f"  epoch {epoch:3d} | train={train_loss:.4f} "
                    f"(clean={_fmt(train_loss_clean)} robust={_fmt(train_loss_robust)}) "
                    f"val_clean={val_metrics['loss']:.4f}/{val_metrics['auc']:.4f} "
                    f"val_robust={val_robust['loss']:.4f}/{val_robust['auc']:.4f} "
                    f"({elapsed:.0f}s){mark}"
                )
                if is_best:
                    print(f"    saved best.pt (val_auc={monitor.best:.4f})")
                save_last(
                    output_dir / LAST_NAME,
                    model=unwrap(model),
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    history=history,
                    monitor=monitor,
                )
                should_stop = monitor.should_stop()
                if should_stop:
                    print(
                        f"  early stop: {cfg['monitor']} did not improve for "
                        f"{cfg['early_stop_patience']} epochs "
                        f"(best={monitor.best:.4f} @ epoch {monitor.best_epoch})"
                    )
            stopped_early = broadcast_bool(should_stop, dist_info)
            barrier(dist_info)
            if stopped_early:
                break

        if dist_info.is_main:
            if writer is not None:
                writer.close()
            with open(output_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)
            print(f"\n  best val AUC: {monitor.best:.4f}  (epoch {monitor.best_epoch})")

            unwrap(model).load_state_dict(torch.load(
                output_dir / "best.pt", map_location=device, weights_only=True,
            ))
            print("  dumping val forensic logits ...")
            _dump_val_predictions(
                unwrap(model), val_ds, device, output_dir,
                batch_size=cfg["batch_size"] * 2,
                workers=cfg["workers"],
                use_bf16=use_bf16,
            )

            if cal_loader is None:
                print("  no calibration split found, skipping temperature scaling.")
            else:
                print("\n  fitting temperature scaling on calibration split ...")
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
    p = argparse.ArgumentParser(description="Train the RGPA forensic tower.")
    p.add_argument("--config", required=True)
    p.add_argument("--data", default="data/datasets/SID_Set_images")
    p.add_argument("--output", default=None)
    p.add_argument(
        "--resume",
        default=None,
        help="last.pt / best.pt, or a run directory containing them.",
    )
    p.add_argument(
        "--extra-epochs",
        type=int,
        default=None,
        help="Train this many more epochs after the resume point.",
    )
    args = p.parse_args()
    if args.output is None:
        cfg_name = yaml.safe_load(open(args.config))["name"]
        args.output = f"runs/forensic_tower/{cfg_name}"
    return args


if __name__ == "__main__":
    train(_parse_args())
