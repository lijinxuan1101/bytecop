"""Evaluate a saved GatedFusion checkpoint on a WildFake val slice.

Does **not** train. Reuses cached val logits when they match the subset size;
otherwise extracts clean + robust logits once.

Usage
-----
    source ~/techjam/venv/bin/activate
    python experiments/fusion/eval.py \
        --ckpt runs/fusion/fusion_wildfake/fusion.pt \
        --val  data/datasets/WildFake_fusion_50k/val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.fusion.train import (  # noqa: E402
    DualViewDataset,
    _CACHE_NAMES,
    _eval_gate,
    _extract_logits,
    _load_cache,
    _load_towers,
    _make_val_loader,
    _save_cache,
    _standardize,
)
from experiments.common.amp import bf16_enabled  # noqa: E402
from experiments.common.distributed import DistInfo, init_distributed, cleanup  # noqa: E402
from models.gated_fusion import GatedFusion  # noqa: E402


def _eval_split(
    gate: GatedFusion,
    pack: dict[str, torch.Tensor],
    *,
    clip_mean: float,
    clip_std: float,
    rgpa_mean: float,
    rgpa_std: float,
    device: torch.device,
    batch_size: int,
) -> tuple[float, object]:
    clip_z = _standardize(pack["clip"], clip_mean, clip_std)
    rgpa_z = _standardize(pack["rgpa"], rgpa_mean, rgpa_std)
    auc, fused, weights = _eval_gate(
        gate, clip_z, rgpa_z, pack["label"], device, batch_size,
    )
    return auc, weights.mean(axis=0)


def _ensure_val_cache(
    cache_path: Path,
    ds: DualViewDataset,
    name: str,
    *,
    clip,
    rgpa,
    dist_info: DistInfo,
    device: torch.device,
    extract_bs: int,
    workers: int,
    use_bf16: bool,
) -> dict[str, torch.Tensor]:
    if cache_path.is_file():
        pack = _load_cache(cache_path)
        if pack["clip"].shape[0] == len(ds):
            if dist_info.is_main:
                print(f"  reuse {cache_path.name}  n={len(ds)}")
            return pack
        if dist_info.is_main:
            print(f"  stale {cache_path.name} n={pack['clip'].shape[0]} != {len(ds)}, redo")
            cache_path.unlink()
    if dist_info.is_main:
        print(f"  extract {name}  n={len(ds)}")
    if clip is None or rgpa is None:
        raise RuntimeError("val cache missing; pass --extract to run the frozen towers")
    payload = _extract_logits(
        clip, rgpa,
        _make_val_loader(ds, extract_bs, workers, dist_info),
        device,
        use_bf16=use_bf16, max_samples=None, desc=f"extract {name}",
        show_progress=dist_info.is_main,
    )
    if dist_info.is_main:
        _save_cache(cache_path, payload)
        print(f"    {name} logits: {payload['clip'].shape[0]}")
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description="Eval GatedFusion on a val slice. No training.")
    p.add_argument("--ckpt", default="runs/fusion/fusion_wildfake/fusion.pt")
    p.add_argument("--config", default="experiments/fusion/configs/fusion_wildfake.yaml")
    p.add_argument("--val", default="data/datasets/WildFake_fusion_50k/val")
    p.add_argument("--output", default=None)
    p.add_argument("--extract", action="store_true", help="Run frozen towers if val cache is missing.")
    p.add_argument("--batch-size", type=int, default=4096)
    args = p.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cache_dir = ckpt_path.parent / "logits_cache"
    val_root = Path(args.val)
    out_path = Path(args.output) if args.output else ckpt_path.parent / "val_official_slice.json"

    dist_info = DistInfo(
        enabled=False, rank=0, local_rank=0, world_size=1,
        device=torch.device("cpu"),
    )
    device = dist_info.device
    img_size = int(cfg.get("img_size") or 224)

    ds_clean = DualViewDataset(val_root, img_size=img_size, augment=False, clean_prob=1.0)
    ds_robust = DualViewDataset(val_root, img_size=img_size, augment=True, clean_prob=0.0)
    print(f"[fusion eval] ckpt={ckpt_path}  val={val_root}  n={len(ds_clean)}")

    clip = rgpa = None
    need_extract = False
    for name, ds in (("val_clean", ds_clean), ("val_robust", ds_robust)):
        path = cache_dir / _CACHE_NAMES[name]
        if not path.is_file():
            need_extract = True
            break
        n_cached = int(torch.load(path, map_location="cpu", weights_only=True)["clip"].shape[0])
        if n_cached != len(ds):
            need_extract = True
            break
    if need_extract:
        if not args.extract:
            raise SystemExit(
                "val logit cache missing or size mismatch. "
                "Re-run with --extract (uses frozen CLIP-H + RGPA, does not train the MLP)."
            )
        dist_info = init_distributed()
        device = dist_info.device
        clip, rgpa = _load_towers(cfg, dist_info)

    extract_bs = int(cfg.get("extract_batch_size") or 16)
    workers = int(cfg.get("workers") or 8)
    use_bf16 = bf16_enabled(cfg, device)
    clean_pack = _ensure_val_cache(
        cache_dir / _CACHE_NAMES["val_clean"], ds_clean, "val_clean",
        clip=clip, rgpa=rgpa, dist_info=dist_info, device=device,
        extract_bs=extract_bs, workers=workers, use_bf16=use_bf16,
    )
    robust_pack = _ensure_val_cache(
        cache_dir / _CACHE_NAMES["val_robust"], ds_robust, "val_robust",
        clip=clip, rgpa=rgpa, dist_info=dist_info, device=device,
        extract_bs=extract_bs, workers=workers, use_bf16=use_bf16,
    )
    if clip is not None:
        cleanup(dist_info)

    gate = GatedFusion(hidden_dim=int(ckpt.get("hidden_dim") or 8))
    gate.load_state_dict(ckpt["gate"])
    gate.to(device)
    gate.eval()

    auc_c, w_c = _eval_split(
        gate, clean_pack,
        clip_mean=float(ckpt["clip_mean"]), clip_std=float(ckpt["clip_std"]),
        rgpa_mean=float(ckpt["rgpa_mean"]), rgpa_std=float(ckpt["rgpa_std"]),
        device=device, batch_size=args.batch_size,
    )
    auc_r, w_r = _eval_split(
        gate, robust_pack,
        clip_mean=float(ckpt["clip_mean"]), clip_std=float(ckpt["clip_std"]),
        rgpa_mean=float(ckpt["rgpa_mean"]), rgpa_std=float(ckpt["rgpa_std"]),
        device=device, batch_size=args.batch_size,
    )
    official = 0.5 * auc_c + 0.5 * auc_r
    w_mean = 0.5 * (w_c + w_r)
    record = {
        "n_val": int(clean_pack["clip"].shape[0]),
        "val_root": str(val_root),
        "ckpt": str(ckpt_path),
        "val_auc_clean": auc_c,
        "val_auc_robust": auc_r,
        "official": official,
        "gate_w_spatial": float(w_mean[0]),
        "gate_w_forensic": float(w_mean[1]),
        "trained": False,
    }
    if dist_info.is_main:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2) + "\n")
        print(
            f"  clean={auc_c:.4f}  robust={auc_r:.4f}  official={official:.4f}  "
            f"w=[{w_mean[0]:.2f} clip, {w_mean[1]:.2f} rgpa]"
        )
        print(f"  wrote {out_path}  (fusion.pt not modified)")


if __name__ == "__main__":
    main()
