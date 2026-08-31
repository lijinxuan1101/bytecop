"""Per-condition Spatial / RGPA / Fusion AUC on the fusion val slice.

Extracts frozen-tower logits once per official transform, then scores the
saved GatedFusion without training.

Usage
-----
    source ~/techjam/venv/bin/activate
    CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --standalone --nproc_per_node=4 \\
        experiments/fusion/eval_conditions.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from data.transforms import apply_transform  # noqa: E402
from evaluate import EVAL_CONDITIONS  # noqa: E402
from experiments.common.amp import bf16_enabled  # noqa: E402
from experiments.common.distributed import (  # noqa: E402
    barrier,
    cleanup,
    init_distributed,
)
from experiments.fusion.train import (  # noqa: E402
    DualViewDataset,
    _auc,
    _eval_gate,
    _extract_logits,
    _gather_payload,
    _load_cache,
    _load_cfg,
    _load_towers,
    _make_val_loader,
    _save_cache,
    _standardize,
    _CLIP_MEAN,
    _CLIP_STD,
)
from models.gated_fusion import GatedFusion  # noqa: E402

# User-facing rows: JPEG kept per quality; other families averaged.
ROW_SPEC: list[tuple[str, tuple[str, ...]]] = [
    ("Clean", ("clean",)),
    ("JPEG 90", ("jpeg_q90",)),
    ("JPEG 70", ("jpeg_q70",)),
    ("JPEG 50", ("jpeg_q50",)),
    ("JPEG 30", ("jpeg_q30",)),
    ("Blur", ("blur_s0.5", "blur_s1.0", "blur_s2.0")),
    ("Resize", ("resize_0.5", "resize_0.25")),
    ("Noise", ("noise_s0.02", "noise_s0.05", "noise_s0.10")),
    ("Color", ("color_jitter",)),
    ("Crop", ("center_crop_80",)),
]


class ConditionDualView(DualViewDataset):
    """DualViewDataset with one fixed official transform (or clean)."""

    def __init__(
        self,
        root: str | Path,
        *,
        img_size: int,
        transform_name: str | None,
        value: float | None,
    ) -> None:
        super().__init__(root, img_size=img_size, augment=False, clean_prob=1.0)
        self.transform_name = transform_name
        self.value = value

    def __getitem__(self, idx: int):
        image, label = self.base[idx]
        is_clean = self.transform_name is None
        if self.transform_name is not None:
            image = apply_transform(
                image, self.transform_name, value=self.value, seed=42 + idx,
            )
        return self.clip_tfm(image), self.rgpa_tfm(image), label, int(is_clean)


def _score_pack(
    pack: dict[str, torch.Tensor],
    gate: GatedFusion,
    ckpt: dict,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    labels = pack["label"]
    spatial = _auc(pack["clip"].numpy(), labels.numpy())
    forensic = _auc(pack["rgpa"].numpy(), labels.numpy())
    clip_z = _standardize(pack["clip"], float(ckpt["clip_mean"]), float(ckpt["clip_std"]))
    rgpa_z = _standardize(pack["rgpa"], float(ckpt["rgpa_mean"]), float(ckpt["rgpa_std"]))
    fused_auc, _, weights = _eval_gate(gate, clip_z, rgpa_z, labels, device, batch_size)
    w = weights.mean(axis=0)
    return {
        "spatial": spatial,
        "rgpa": forensic,
        "fusion": fused_auc,
        "delta": fused_auc - spatial,
        "gate_w_spatial": float(w[0]),
        "gate_w_forensic": float(w[1]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Per-condition tower vs fusion AUC.")
    p.add_argument("--config", default="experiments/fusion/configs/fusion_wildfake.yaml")
    p.add_argument("--ckpt", default="runs/fusion/fusion_wildfake/fusion.pt")
    p.add_argument("--val", default="data/datasets/WildFake_fusion_50k/val")
    p.add_argument("--output", default="runs/fusion/fusion_wildfake/val_conditions.json")
    args = p.parse_args()

    cfg = _load_cfg(args.config)
    dist_info = init_distributed()
    device = dist_info.device
    ckpt_path = Path(args.ckpt)
    val_root = Path(args.val)
    cache_dir = ckpt_path.parent / "logits_cache" / "conditions"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    img_size = int(cfg["img_size"])
    extract_bs = int(cfg["extract_batch_size"])
    workers = int(cfg["workers"])
    use_bf16 = bf16_enabled(cfg, device)

    missing = []
    for name, _, _ in EVAL_CONDITIONS:
        path = cache_dir / f"{name}.pt"
        n_ok = False
        if path.is_file():
            n_ok = int(torch.load(path, map_location="cpu", weights_only=True)["clip"].shape[0]) > 0
        if not n_ok:
            missing.append(name)
    clip = rgpa = None
    if missing:
        if dist_info.is_main:
            print(f"[cond eval] extracting {len(missing)} conditions on {val_root}")
        clip, rgpa = _load_towers(cfg, dist_info)
    elif dist_info.is_main:
        print(f"[cond eval] reusing cached logits under {cache_dir}")

    packs: dict[str, dict[str, torch.Tensor]] = {}
    for name, transform_name, value in EVAL_CONDITIONS:
        path = cache_dir / f"{name}.pt"
        ds = ConditionDualView(
            val_root, img_size=img_size,
            transform_name=transform_name, value=value,
        )
        if path.is_file() and int(torch.load(path, map_location="cpu", weights_only=True)["clip"].shape[0]) == len(ds):
            if dist_info.is_main:
                print(f"  reuse {name}  n={len(ds)}")
            if dist_info.is_main:
                packs[name] = _load_cache(path)
            barrier(dist_info)
            continue
        if dist_info.is_main:
            print(f"  extract {name}  n={len(ds)}")
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
            packs[name] = payload
            print(f"    {name}: {payload['clip'].shape[0]}")
        barrier(dist_info)

    if clip is not None:
        del clip, rgpa
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if dist_info.is_main:
        gate = GatedFusion(hidden_dim=int(ckpt.get("hidden_dim") or 8)).to(device)
        gate.load_state_dict(ckpt["gate"])
        gate.eval()
        gate_bs = int(cfg.get("gate_batch_size") or 4096)
        raw = {
            name: _score_pack(packs[name], gate, ckpt, device, gate_bs)
            for name, _, _ in EVAL_CONDITIONS
        }

        def _mean_field(keys: tuple[str, ...], field: str) -> float:
            return float(np.mean([raw[k][field] for k in keys]))

        rows = []
        for label, keys in ROW_SPEC:
            spatial = _mean_field(keys, "spatial")
            rgpa_auc = _mean_field(keys, "rgpa")
            fusion = _mean_field(keys, "fusion")
            rows.append({
                "condition": label,
                "keys": list(keys),
                "spatial": spatial,
                "rgpa": rgpa_auc,
                "fusion": fusion,
                "fusion_minus_spatial": fusion - spatial,
            })
        degraded = [r for r in rows if r["condition"] != "Clean"]
        summary = {
            "n_val": int(packs["clean"]["clip"].shape[0]),
            "val_root": str(val_root),
            "ckpt": str(ckpt_path),
            "rows": rows,
            "raw": raw,
            "official_spatial": 0.5 * rows[0]["spatial"] + 0.5 * float(np.mean([r["spatial"] for r in degraded])),
            "official_rgpa": 0.5 * rows[0]["rgpa"] + 0.5 * float(np.mean([r["rgpa"] for r in degraded])),
            "official_fusion": 0.5 * rows[0]["fusion"] + 0.5 * float(np.mean([r["fusion"] for r in degraded])),
        }
        # Official robust is mean over 14 raw conditions, not the 9 grouped rows.
        robust_keys = [n for n, _, _ in EVAL_CONDITIONS if n != "clean"]
        summary["official_spatial"] = 0.5 * raw["clean"]["spatial"] + 0.5 * float(
            np.mean([raw[k]["spatial"] for k in robust_keys])
        )
        summary["official_rgpa"] = 0.5 * raw["clean"]["rgpa"] + 0.5 * float(
            np.mean([raw[k]["rgpa"] for k in robust_keys])
        )
        summary["official_fusion"] = 0.5 * raw["clean"]["fusion"] + 0.5 * float(
            np.mean([raw[k]["fusion"] for k in robust_keys])
        )
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n")
        print()
        print(f"{'condition':<12} {'spatial':>8} {'rgpa':>8} {'fusion':>8} {'Δ':>8}")
        for row in rows:
            print(
                f"{row['condition']:<12} {row['spatial']:8.4f} {row['rgpa']:8.4f} "
                f"{row['fusion']:8.4f} {row['fusion_minus_spatial']:+8.4f}"
            )
        print(
            f"{'official':<12} {summary['official_spatial']:8.4f} "
            f"{summary['official_rgpa']:8.4f} {summary['official_fusion']:8.4f} "
            f"{summary['official_fusion'] - summary['official_spatial']:+8.4f}"
        )
        print(f"wrote {out}")

    barrier(dist_info)
    cleanup(dist_info)


if __name__ == "__main__":
    main()
