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
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
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
    move_optimizer_state,
    resolve_resume_path,
    save_last,
)
from experiments.common.distributed import (  # noqa: E402
    DistInfo,
    barrier,
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


def _build_model(cfg: dict, device: torch.device, *, load_weights: bool) -> CLIPTower:
    model = CLIPTower(
        unfreeze_blocks=cfg["unfreeze_blocks"],
        proj_dim=cfg["proj_dim"],
        dropout=cfg["dropout"],
        load_weights=load_weights,
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

    raw_model = _build_model(cfg, device, load_weights=dist_info.is_main)
    if dist_info.is_main:
        counts = raw_model.param_count()
        print(f"  params : total={counts['total']:,}  trainable={counts['trainable']:,}")
        if dist_info.enabled:
            print("  load   : rank 0 reads CLIP / ckpt, then broadcast")

    start_epoch = 1
    end_epoch = cfg["epochs"]
    history: list[dict] = []
    global_step = 0
    constant_lr = False
    inherit_lr = None
    resume_payload: dict | None = None

    if args.resume is not None and dist_info.is_main:
        ckpt_path = resolve_resume_path(args.resume)
        resume_blob = load_checkpoint(ckpt_path, map_location="cpu")
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
        elif (output_dir / "best_metrics.json").is_file():
            with open(output_dir / "best_metrics.json") as f:
                start_epoch = int(json.load(f).get("epoch", 0)) + 1
        steps_per_epoch = math.ceil(len(train_loader) / accum)
        if resume_blob.get("global_step") is not None:
            global_step = int(resume_blob["global_step"])
        else:
            global_step = (start_epoch - 1) * steps_per_epoch
        full_resume = resume_blob.get("optimizer") is not None
        if not full_resume:
            inherit_lr = last_positive_lr(history)
            constant_lr = True
        resume_payload = {
            "ckpt_path": str(ckpt_path),
            "kind": "full (model+optim+sched)" if full_resume else "weights-only",
            "history": history,
            "start_epoch": start_epoch,
            "global_step": global_step,
            "constant_lr": constant_lr,
            "inherit_lr": inherit_lr,
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
        inherit_lr = resume_payload["inherit_lr"]
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
                    f"Using constant lr={inherit_lr or cfg['head_lr']:.3e}."
                )

    groups = _param_groups(raw_model, cfg)
    if inherit_lr is not None:
        for group in groups:
            group["lr"] = inherit_lr if group["name"] == "head" else inherit_lr * (
                cfg["backbone_lr"] / cfg["head_lr"]
            )
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
    if constant_lr:
        scheduler = LambdaLR(optimizer, lambda _: 1.0)
        warmup_steps = 0
        total_steps = args.extra_epochs or 0
    else:
        total_steps = cfg["epochs"] * steps_per_epoch
        scheduler, warmup_steps = build_warmup_cosine(
            optimizer, total_steps=total_steps, warmup_ratio=cfg["warmup_ratio"],
        )
        if resume_payload is not None and resume_payload.get("optimizer") is not None:
            optimizer.load_state_dict(resume_payload["optimizer"])
            move_optimizer_state(optimizer, device)
            if resume_payload.get("scheduler") is not None:
                scheduler.load_state_dict(resume_payload["scheduler"])
    if dist_info.is_main:
        print(f"  sched  : {'constant' if constant_lr else cfg['scheduler']}  "
              f"warmup_ratio={0 if constant_lr else cfg['warmup_ratio']}  "
              f"warmup_steps={warmup_steps}/{total_steps}")
        print(f"  monitor: {cfg['monitor']} → {output_dir / 'best.pt'}")

    class_counts = train_ds.class_counts()
    pos_weight = torch.tensor(
        [class_counts.get(0, 1) / class_counts.get(1, 1)], device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard")) if dist_info.is_main else None
    monitor = MetricMonitor(output_dir, metric=cfg["monitor"]) if dist_info.is_main else None
    if dist_info.is_main and monitor is not None:
        if resume_payload is not None and resume_payload.get("monitor"):
            monitor.load_state_dict(resume_payload["monitor"])
        elif args.resume is not None:
            monitor.restore_from_run(output_dir, history)

    try:
        for epoch in range(start_epoch, end_epoch + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            t0 = time.time()
            running_loss = 0.0
            clean_sum = robust_sum = 0.0
            clean_n = robust_n = 0
            optimizer.zero_grad(set_to_none=True)

            iterator = tqdm(
                train_loader, desc=f"epoch {epoch}/{end_epoch}", leave=False,
                disable=not dist_info.is_main,
            )
            for micro_idx, (images, labels, is_clean) in enumerate(iterator, start=1):
                images = images.to(device, non_blocking=True)
                labels = labels.float().to(device, non_blocking=True)
                is_clean = is_clean.to(device, non_blocking=True)
                with autocast_ctx(use_bf16):
                    logits = model(images)
                    loss = criterion(logits, labels) / accum
                loss.backward()
                running_loss += loss.item() * accum
                c_sum, c_n, r_sum, r_n = _split_plain_bce(logits.detach(), labels, is_clean)
                clean_sum += c_sum
                clean_n += c_n
                robust_sum += r_sum
                robust_n += r_n

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
            train_loss_clean = _safe_mean(
                reduce_sum(clean_sum, dist_info), reduce_sum(float(clean_n), dist_info),
            )
            train_loss_robust = _safe_mean(
                reduce_sum(robust_sum, dist_info), reduce_sum(float(robust_n), dist_info),
            )

            # All ranks run val so nobody sits on an NCCL barrier for 10+ minutes.
            val_metrics = _evaluate(
                unwrap(model), val_loader, device, dist_info, use_bf16=use_bf16,
            )
            val_robust = _evaluate(
                unwrap(model), val_robust_loader, device, dist_info, use_bf16=use_bf16,
            )
            if dist_info.is_main:
                elapsed = time.time() - t0
                lr_head = optimizer.param_groups[0]["lr"]
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
                    "lr_head": lr_head,
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
                writer.add_scalar("train/lr_head", lr_head, epoch)
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
        args.output = f"runs/spatial_tower/{cfg_name}"
    return args


if __name__ == "__main__":
    train(_parse_args())
