"""Spatial 15-condition ablation: CLIP preprocess vs MBE then CLIP.

Same val slice as fusion eval (``WildFake_fusion_50k/val``, 5k from official
WildFake val). MBE runs after the official degradation, before OpenCLIP resize.

    CUDA_VISIBLE_DEVICES=2,3,4 torchrun --standalone --nproc_per_node=3 \\
        experiments/spatial_tower/eval_mbe.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from data.dataset import AIGCDataset  # noqa: E402
from data.mbe import enhance  # noqa: E402
from data.transforms import apply_transform  # noqa: E402
from evaluate import EVAL_CONDITIONS  # noqa: E402
from experiments.common.amp import autocast_ctx  # noqa: E402
from experiments.common.distributed import (  # noqa: E402
    barrier,
    cleanup,
    eval_sampler,
    gather_cat,
    init_distributed,
)
from models.clip_tower import CLIPTower  # noqa: E402

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

DEFAULT_CKPT = Path("runs/spatial_tower/spatial_tower_wildfake/best.pt")
DEFAULT_VAL = Path("data/datasets/WildFake_fusion_50k/val")
DEFAULT_OUT = Path("runs/spatial_tower/spatial_tower_wildfake/mbe_ablation.json")


def _clip_tfm(*, img_size: int) -> T.Compose:
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
    ])


class ConditionPairDataset(Dataset):
    """Official transform, then CLIP tensor and MBE→CLIP tensor."""

    def __init__(
        self,
        root: str | Path,
        *,
        img_size: int,
        transform_name: str | None,
        value: float | None,
    ) -> None:
        self.base = AIGCDataset(root, transform=None)
        self.clip_tfm = _clip_tfm(img_size=img_size)
        self.transform_name = transform_name
        self.value = value

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image, label = self.base[idx]
        if self.transform_name is not None:
            image = apply_transform(
                image, self.transform_name, value=self.value, seed=42 + idx,
            )
        plain = self.clip_tfm(image)
        boosted = self.clip_tfm(enhance(image))
        return plain, boosted, label


def _load_model(*, ckpt: Path, device: torch.device) -> torch.nn.Module:
    model = CLIPTower(
        unfreeze_blocks=2, proj_dim=512, dropout=0.1, load_weights=False,
    )
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def _run_pair(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_bf16: bool,
    desc: str,
    show_progress: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base_logits, mbe_logits, labels = [], [], []
    for plain, boosted, y in tqdm(loader, desc=desc, disable=not show_progress):
        plain = plain.to(device, non_blocking=True)
        boosted = boosted.to(device, non_blocking=True)
        with autocast_ctx(use_bf16):
            b = model(plain)
            m = model(boosted)
        base_logits.append(b.float().cpu())
        mbe_logits.append(m.float().cpu())
        labels.append(y.float().cpu())
    return (
        torch.cat(base_logits),
        torch.cat(mbe_logits),
        torch.cat(labels),
    )


def _auc(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(roc_auc_score(labels, logits))


def _official(aucs: dict[str, float]) -> tuple[float, float, float]:
    clean = aucs["clean"]
    robust = float(np.mean([v for k, v in aucs.items() if k != "clean"]))
    return clean, robust, 0.5 * clean + 0.5 * robust


def main() -> None:
    p = argparse.ArgumentParser(description="Spatial MBE ablation on 15 conditions.")
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    p.add_argument("--data", default=str(DEFAULT_VAL))
    p.add_argument("--output", default=str(DEFAULT_OUT))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--img-size", type=int, default=224)
    args = p.parse_args()

    dist_info = init_distributed()
    device = dist_info.device
    ckpt = Path(args.ckpt)
    val_root = Path(args.data)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    if dist_info.is_main:
        print(
            f"[mbe ablation] ckpt={ckpt}  val={val_root}  "
            f"device={device}  world={dist_info.world_size}  bf16={use_bf16}"
        )

    model = _load_model(ckpt=ckpt, device=device)
    base_aucs: dict[str, float] = {}
    mbe_aucs: dict[str, float] = {}
    n_val = 0

    for name, transform_name, value in EVAL_CONDITIONS:
        ds = ConditionPairDataset(
            val_root, img_size=args.img_size,
            transform_name=transform_name, value=value,
        )
        sampler = eval_sampler(len(ds), dist_info)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=False,
        )
        if dist_info.is_main:
            print(f"  {name}  n={len(ds)}")
        base_l, mbe_l, labels = _run_pair(
            model, loader, device,
            use_bf16=use_bf16, desc=name, show_progress=dist_info.is_main,
        )
        base_l = gather_cat(base_l, dist_info).cpu()
        mbe_l = gather_cat(mbe_l, dist_info).cpu()
        labels = gather_cat(labels, dist_info).cpu()
        if dist_info.is_main:
            y = labels.numpy()
            b_auc = _auc(base_l.numpy(), y)
            m_auc = _auc(mbe_l.numpy(), y)
            base_aucs[name] = b_auc
            mbe_aucs[name] = m_auc
            n_val = int(y.shape[0])
            print(
                f"    baseline={b_auc:.4f}  mbe={m_auc:.4f}  "
                f"Δ={m_auc - b_auc:+.4f}"
            )
        barrier(dist_info)

    if dist_info.is_main:
        b_clean, b_robust, b_off = _official(base_aucs)
        m_clean, m_robust, m_off = _official(mbe_aucs)
        conditions = {
            name: {
                "baseline": base_aucs[name],
                "mbe": mbe_aucs[name],
                "delta": mbe_aucs[name] - base_aucs[name],
            }
            for name, _, _ in EVAL_CONDITIONS
        }
        summary = {
            "n_val": n_val,
            "val_root": str(val_root),
            "ckpt": str(ckpt),
            "pipeline": "degrade → MBE → OpenCLIP resize/crop/normalize → Spatial",
            "conditions": conditions,
            "auc_clean_baseline": b_clean,
            "auc_clean_mbe": m_clean,
            "auc_robust_baseline": b_robust,
            "auc_robust_mbe": m_robust,
            "official_baseline": b_off,
            "official_mbe": m_off,
            "official_delta": m_off - b_off,
        }
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n")
        print()
        print(f"{'condition':<16} {'baseline':>9} {'mbe':>9} {'Δ':>8}")
        for name, _, _ in EVAL_CONDITIONS:
            row = conditions[name]
            print(
                f"{name:<16} {row['baseline']:9.4f} {row['mbe']:9.4f} "
                f"{row['delta']:+8.4f}"
            )
        print(
            f"{'official':<16} {b_off:9.4f} {m_off:9.4f} {m_off - b_off:+8.4f}"
        )
        print(f"wrote {out}")

    barrier(dist_info)
    cleanup(dist_info)


if __name__ == "__main__":
    main()
