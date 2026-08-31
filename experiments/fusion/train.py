"""Train GatedFusion on frozen CLIP-H + RGPA logits.

Pipeline:
  1. Sample a type-balanced 50k subset (train/val split) if missing.
  2. Extract tower logits once (train + val_clean + val_robust).
  3. Train the tiny MLP on the cached tensors for many epochs.

Towers are never updated. The gate trains on rank 0 only.
"""

from __future__ import annotations

import argparse
import json
import os
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
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from calibration.temperature_scaling import TemperatureScaler  # noqa: E402
from data.dataset import AIGCDataset  # noqa: E402
from data.prepare_fusion_subset import prepare as prepare_subset  # noqa: E402
from data.transforms import apply_train_policy  # noqa: E402
from experiments.common.amp import autocast_ctx, bf16_enabled  # noqa: E402
from experiments.common.distributed import (  # noqa: E402
    DistInfo,
    barrier,
    broadcast_module,
    cleanup,
    eval_sampler,
    gather_cat,
    init_distributed,
)
from models.clip_tower import CLIPTower  # noqa: E402
from models.gated_fusion import GatedFusion  # noqa: E402
from models.rgpa import RGPA  # noqa: E402


_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_CACHE_NAMES = {
    "train": "logits_train.pt",
    "val_clean": "logits_val_clean.pt",
    "val_robust": "logits_val_robust.pt",
}


class DualViewDataset(Dataset):
    """One PIL augment, then CLIP-normalize and RGPA pixel tensors."""

    def __init__(
        self,
        root: str | Path,
        *,
        img_size: int,
        augment: bool,
        clean_prob: float,
    ) -> None:
        self.base = AIGCDataset(root, transform=None)
        self.augment = augment
        self.clean_prob = clean_prob
        self.clip_tfm = T.Compose([
            T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
        ])
        self.rgpa_tfm = T.Compose([
            T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image, label = self.base[idx]
        is_clean = True
        if self.augment:
            image, is_clean = apply_train_policy(image, clean_prob=self.clean_prob)
        return self.clip_tfm(image), self.rgpa_tfm(image), label, int(is_clean)


def _load_cfg(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("hidden_dim", 8)
    cfg.setdefault("entropy_coef", 0.0)
    cfg.setdefault("epochs", 1)
    cfg.setdefault("lr", 1e-3)
    cfg.setdefault("weight_decay", 0.0)
    cfg.setdefault("gate_batch_size", 4096)
    cfg.setdefault("extract_batch_size", 16)
    cfg.setdefault("n_subset", 50_000)
    cfg.setdefault("n_val", 5_000)
    cfg.setdefault("seed", 42)
    cfg.setdefault("subset", "data/datasets/WildFake_fusion_50k")
    cfg.setdefault("workers", 8)
    cfg.setdefault("bf16", False)
    cfg.setdefault("augment", True)
    cfg.setdefault("clean_prob", 0.3)
    cfg.setdefault("img_size", 224)
    return cfg


def _load_towers(cfg: dict, dist_info: DistInfo) -> tuple[nn.Module, nn.Module]:
    clip_kw = dict(cfg["clip"])
    clip_kw["load_weights"] = False
    clip = CLIPTower(**clip_kw)
    rgpa = RGPA(**cfg["rgpa"])
    if dist_info.is_main:
        clip.load_state_dict(
            torch.load(cfg["clip_ckpt"], map_location="cpu", weights_only=True),
            strict=False,
        )
        rgpa.load_state_dict(
            torch.load(cfg["rgpa_ckpt"], map_location="cpu", weights_only=True),
            strict=False,
        )
    clip.to(dist_info.device)
    rgpa.to(dist_info.device)
    broadcast_module(clip, dist_info)
    broadcast_module(rgpa, dist_info)
    clip.eval()
    rgpa.eval()
    for p in clip.parameters():
        p.requires_grad = False
    for p in rgpa.parameters():
        p.requires_grad = False
    return clip, rgpa


@torch.no_grad()
def _extract_logits(
    clip: nn.Module,
    rgpa: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_bf16: bool,
    max_samples: int | None,
    desc: str,
    show_progress: bool = True,
) -> dict[str, torch.Tensor]:
    clips, rgpas, labels, flags = [], [], [], []
    n = 0
    clip.eval()
    rgpa.eval()
    for clip_x, rgpa_x, y, is_clean in tqdm(
        loader, desc=desc, disable=not show_progress,
    ):
        clip_x = clip_x.to(device, non_blocking=True)
        rgpa_x = rgpa_x.to(device, non_blocking=True)
        with autocast_ctx(use_bf16):
            c = clip(clip_x)
            r = rgpa(rgpa_x)
        clips.append(c.float().cpu())
        rgpas.append(r.float().cpu())
        labels.append(y.float().cpu())
        flags.append(is_clean.float().cpu())
        n += y.shape[0]
        if max_samples is not None and n >= max_samples:
            break
    if not clips:
        empty = torch.zeros(0, dtype=torch.float32)
        payload = {"clip": empty, "rgpa": empty.clone(), "label": empty.clone(), "is_clean": empty.clone()}
    else:
        payload = {
            "clip": torch.cat(clips),
            "rgpa": torch.cat(rgpas),
            "label": torch.cat(labels),
            "is_clean": torch.cat(flags),
        }
        if max_samples is not None:
            payload = {k: v[:max_samples] for k, v in payload.items()}
    return payload


def _gather_payload(
    payload: dict[str, torch.Tensor],
    dist_info: DistInfo,
    *,
    max_samples: int | None = None,
) -> dict[str, torch.Tensor]:
    gathered = {
        key: gather_cat(value, dist_info).cpu()
        for key, value in payload.items()
    }
    if max_samples is not None:
        gathered = {k: v[:max_samples] for k, v in gathered.items()}
    return gathered


def _save_cache(path: Path, payload: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _stage_ckpt(src: str | Path, dest: Path) -> Path:
    """Hard-link (or copy) a tower checkpoint next to fusion.pt."""
    src = Path(src).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)
    return dest


def _load_cache(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)


def _standardize(logits: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (logits - mean) / max(std, 1e-4)


def _auc(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(roc_auc_score(labels, 1.0 / (1.0 + np.exp(-logits))))


def _eval_gate(
    gate: GatedFusion,
    clip_z: torch.Tensor,
    rgpa_z: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    gate.eval()
    fused_chunks, gate_chunks = [], []
    ds = TensorDataset(clip_z, rgpa_z)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for c, r in loader:
            fused, weights = gate(c.to(device), r.to(device))
            fused_chunks.append(fused.cpu())
            gate_chunks.append(weights.cpu())
    fused_np = torch.cat(fused_chunks).numpy()
    labels_np = labels.numpy()
    return _auc(fused_np, labels_np), fused_np, torch.cat(gate_chunks).numpy()


def _make_val_loader(
    ds: Dataset,
    batch_size: int,
    workers: int,
    dist_info: DistInfo,
) -> DataLoader:
    sampler = eval_sampler(len(ds), dist_info)
    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )


def train(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config)
    dist_info = init_distributed()
    output_dir = Path(args.output)
    cache_dir = output_dir / "logits_cache"
    data_root = Path(args.data)
    device = dist_info.device
    use_bf16 = bf16_enabled(cfg, device)
    if dist_info.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.config, output_dir / "config.yaml")
        print(
            f"[fusion/{cfg['name']}]  device={device}  world={dist_info.world_size}  "
            f"clip={cfg['clip_ckpt']}  rgpa={cfg['rgpa_ckpt']}"
        )
    barrier(dist_info)

    try:
        _train_body(
            args, cfg, dist_info, output_dir, cache_dir, data_root,
            device, use_bf16,
        )
    finally:
        cleanup(dist_info)




def _train_body(
    args: argparse.Namespace,
    cfg: dict,
    dist_info: DistInfo,
    output_dir: Path,
    cache_dir: Path,
    data_root: Path,
    device: torch.device,
    use_bf16: bool,
) -> None:
    subset_root = Path(cfg.get("subset") or "data/datasets/WildFake_fusion_50k")
    n_subset = int(cfg.get("n_subset") or cfg.get("max_train_samples") or 50_000)
    n_val = int(cfg.get("n_val") or 5_000)
    seed = int(cfg.get("seed") or 42)
    summary_path = subset_root / "manifest.json"
    need_subset = True
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text())
        need_subset = existing.get("val_source") != "wildfake_images_val"
    if dist_info.is_main and need_subset:
        print(
            f"  subset : train n={n_subset} from {data_root}/train, "
            f"val n={n_val} from {data_root}/val → {subset_root}"
        )
        summary = prepare_subset(
            data_root, subset_root, n=n_subset, n_val=n_val, seed=seed,
            keep_train=True,
        )
        print(f"           train={summary['train']} val={summary['val']}")
    barrier(dist_info)

    clip, rgpa = _load_towers(cfg, dist_info)
    img_size = cfg["img_size"]
    extract_bs = cfg["extract_batch_size"]
    workers = cfg["workers"]

    specs = [
        ("train", DualViewDataset(
            subset_root / "train", img_size=img_size,
            augment=cfg["augment"], clean_prob=cfg["clean_prob"],
        )),
        ("val_clean", DualViewDataset(
            subset_root / "val", img_size=img_size,
            augment=False, clean_prob=1.0,
        )),
        ("val_robust", DualViewDataset(
            subset_root / "val", img_size=img_size,
            augment=True, clean_prob=0.0,
        )),
    ]
    for name, ds in specs:
        path = cache_dir / _CACHE_NAMES[name]
        if path.is_file():
            n_cached = int(torch.load(path, map_location="cpu", weights_only=True)["clip"].shape[0])
            if n_cached == len(ds):
                if dist_info.is_main:
                    print(f"  extract: reuse {path.name}  n={n_cached}")
                continue
            if dist_info.is_main:
                print(f"  extract: stale {path.name} n={n_cached} != {len(ds)}, redo")
                path.unlink()
            barrier(dist_info)
        if dist_info.is_main:
            print(f"  extract: {name}  n={len(ds)}")
        payload = _extract_logits(
            clip, rgpa,
            _make_val_loader(ds, extract_bs, workers, dist_info),
            device,
            use_bf16=use_bf16, max_samples=None, desc=f"extract {name}",
            show_progress=dist_info.is_main,
        )
        payload = _gather_payload(payload, dist_info)
        if dist_info.is_main:
            _save_cache(path, payload)
            print(f"    {name} logits: {payload['clip'].shape[0]}")
        barrier(dist_info)

    del clip, rgpa
    if device.type == "cuda":
        torch.cuda.empty_cache()
    barrier(dist_info)
    if dist_info.is_main:
        _train_gate(
            cfg, cache_dir, output_dir, subset_root, device,
        )
    barrier(dist_info)


def _train_gate(
    cfg: dict,
    cache_dir: Path,
    output_dir: Path,
    subset_root: Path,
    device: torch.device,
) -> None:
    train_pack = _load_cache(cache_dir / _CACHE_NAMES["train"])
    val_clean = _load_cache(cache_dir / _CACHE_NAMES["val_clean"])
    val_robust = _load_cache(cache_dir / _CACHE_NAMES["val_robust"])
    clip_mean = float(train_pack["clip"].mean())
    clip_std = float(train_pack["clip"].std().clamp(min=1e-4))
    rgpa_mean = float(train_pack["rgpa"].mean())
    rgpa_std = float(train_pack["rgpa"].std().clamp(min=1e-4))
    stats = {
        "clip_mean": clip_mean, "clip_std": clip_std,
        "rgpa_mean": rgpa_mean, "rgpa_std": rgpa_std,
    }
    print(
        f"  stats  : clip μ={clip_mean:.4f} σ={clip_std:.4f}  "
        f"rgpa μ={rgpa_mean:.4f} σ={rgpa_std:.4f}"
    )
    print("  mlp    : training GatedFusion on cached logits")

    def zpack(pack: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            "clip": _standardize(pack["clip"], clip_mean, clip_std),
            "rgpa": _standardize(pack["rgpa"], rgpa_mean, rgpa_std),
            "label": pack["label"],
        }

    train_z = zpack(train_pack)
    val_c_z = zpack(val_clean)
    val_r_z = zpack(val_robust)
    gate = GatedFusion(hidden_dim=cfg["hidden_dim"]).to(device)
    optimizer = torch.optim.AdamW(
        gate.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )
    criterion = nn.BCEWithLogitsLoss()
    entropy_coef = float(cfg["entropy_coef"])
    gate_bs = cfg["gate_batch_size"]
    train_loader = DataLoader(
        TensorDataset(train_z["clip"], train_z["rgpa"], train_z["label"]),
        batch_size=gate_bs, shuffle=True,
    )
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    history: list[dict] = []
    best_official = -1.0
    best_state = None
    t0 = time.time()
    n_epochs = int(cfg["epochs"])
    for epoch in range(1, n_epochs + 1):
        gate.train()
        running = 0.0
        n_batches = 0
        for clip_z, rgpa_z, y in train_loader:
            clip_z = clip_z.to(device)
            rgpa_z = rgpa_z.to(device)
            y = y.to(device)
            fused, weights = gate(clip_z, rgpa_z)
            loss = criterion(fused, y)
            if entropy_coef > 0:
                entropy = -(weights * (weights.clamp(min=1e-8).log())).sum(dim=-1).mean()
                loss = loss - entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1
        train_loss = running / max(n_batches, 1)
        auc_c, fused_c, w_c = _eval_gate(
            gate, val_c_z["clip"], val_c_z["rgpa"], val_c_z["label"], device, gate_bs,
        )
        auc_r, _, w_r = _eval_gate(
            gate, val_r_z["clip"], val_r_z["rgpa"], val_r_z["label"], device, gate_bs,
        )
        official = 0.5 * auc_c + 0.5 * auc_r
        w_mean = 0.5 * (w_c.mean(axis=0) + w_r.mean(axis=0))
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auc_clean": auc_c,
            "val_auc_robust": auc_r,
            "official": official,
            "gate_w_spatial": float(w_mean[0]),
            "gate_w_forensic": float(w_mean[1]),
            "elapsed_s": round(time.time() - t0, 1),
        }
        history.append(record)
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/auc_clean", auc_c, epoch)
        writer.add_scalar("val/auc_robust", auc_r, epoch)
        writer.add_scalar("val/official", official, epoch)
        writer.add_scalar("gate/w_spatial", float(w_mean[0]), epoch)
        writer.add_scalar("gate/w_forensic", float(w_mean[1]), epoch)
        mark = ""
        if official > best_official:
            best_official = official
            best_state = {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()}
            mark = "  *best*"
        print(
            f"  epoch {epoch:3d} | loss={train_loss:.4f}  "
            f"clean={auc_c:.4f} robust={auc_r:.4f} official={official:.4f}  "
            f"w=[{w_mean[0]:.2f} clip, {w_mean[1]:.2f} rgpa]{mark}"
        )

    assert best_state is not None
    gate.load_state_dict(best_state)
    _, fused_c, _ = _eval_gate(
        gate, val_c_z["clip"], val_c_z["rgpa"], val_c_z["label"], device, gate_bs,
    )
    scaler = TemperatureScaler()
    scaler.fit(fused_c, val_c_z["label"].numpy())
    quality = scaler.calibration_metrics(fused_c, val_c_z["label"].numpy())
    print(f"  temperature T={quality['temperature']:.4f}  ECE={quality['ece']:.4f}")
    clip_local = _stage_ckpt(cfg["clip_ckpt"], output_dir / "clip_h.pt")
    rgpa_local = _stage_ckpt(cfg["rgpa_ckpt"], output_dir / "rgpa.pt")
    print(f"  staged : {clip_local.name}  {rgpa_local.name}")
    torch.save({
        "gate": best_state,
        "hidden_dim": cfg["hidden_dim"],
        **stats,
        "temperature": quality["temperature"],
        "clip_ckpt": str(clip_local),
        "rgpa_ckpt": str(rgpa_local),
        "best_official": best_official,
        "subset": str(subset_root),
    }, output_dir / "fusion.pt")
    with open(output_dir / "calibrator.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(output_dir / "best_metrics.json", "w") as f:
        json.dump({
            **history[max(range(len(history)), key=lambda i: history[i]["official"])],
            "temperature": quality["temperature"],
            "ece": quality["ece"],
            **stats,
        }, f, indent=2)
    writer.close()
    print(f"  wrote {output_dir / 'fusion.pt'}  official={best_official:.4f}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train gated fusion on frozen tower logits.")
    p.add_argument("--config", required=True)
    p.add_argument("--data", default="data/datasets/WildFake_images")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    if args.output is None:
        cfg_name = yaml.safe_load(open(args.config))["name"]
        args.output = f"runs/fusion/{cfg_name}"
    return args


if __name__ == "__main__":
    train(_parse_args())
