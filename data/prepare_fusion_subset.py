"""Sample a type-balanced fusion subset from WildFake.

Train: 50/50 real/fake from WildFake **train**, fakes round-robin over Architecture.
Val: the same recipe, but from WildFake **val** (a small slice of the official 66k).

Output is manifest-only (absolute paths). ``AIGCDataset`` can load it directly.

Usage
-----
    python data/prepare_fusion_subset.py
    python data/prepare_fusion_subset.py --n 50000 --n-val 5000
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from data.dataset import AIGCDataset  # noqa: E402
from data.type_balanced_sampler import build_type_groups  # noqa: E402


def sample_balanced(
    reals: list[int],
    fake_by_arch: dict[str, list[int]],
    n: int,
    *,
    seed: int,
) -> list[int]:
    rng = random.Random(seed)
    n_real = n // 2
    n_fake = n - n_real
    arches = sorted(fake_by_arch)
    pools = {arch: idxs[:] for arch, idxs in fake_by_arch.items()}
    for arch in arches:
        rng.shuffle(pools[arch])
    cursors = {arch: 0 for arch in arches}
    fakes: list[int] = []
    while len(fakes) < n_fake:
        progressed = False
        for arch in arches:
            i = cursors[arch]
            if i < len(pools[arch]) and len(fakes) < n_fake:
                fakes.append(pools[arch][i])
                cursors[arch] = i + 1
                progressed = True
        if not progressed:
            break
    if len(fakes) < n_fake:
        raise RuntimeError(f"only {len(fakes)} unique fakes, need {n_fake}")
    reals_shuffled = reals[:]
    rng.shuffle(reals_shuffled)
    if len(reals_shuffled) < n_real:
        raise RuntimeError(f"only {len(reals_shuffled)} reals, need {n_real}")
    return reals_shuffled[:n_real] + fakes


def _rows_from_indices(dataset: AIGCDataset, indices: list[int]) -> list[dict]:
    rows = []
    for idx in indices:
        path, label = dataset.samples[idx]
        arch = dataset.architectures[idx] if idx < len(dataset.architectures) else ""
        rows.append({
            "path": str(Path(path).resolve()),
            "label": int(label),
            "architecture": arch,
        })
    return rows


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "label", "architecture"])
        writer.writeheader()
        writer.writerows(rows)


def _load_manifest(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _counts(rows: list[dict]) -> tuple[int, int]:
    real = sum(1 for r in rows if int(r["label"]) == 0)
    return real, len(rows) - real


def prepare(
    src: Path,
    dest: Path,
    *,
    n: int = 50_000,
    n_val: int = 5_000,
    seed: int = 42,
    keep_train: bool = True,
) -> dict:
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    train_manifest = dest / "train" / "manifest.csv"
    val_manifest = dest / "val" / "manifest.csv"

    if keep_train and train_manifest.is_file():
        train_rows = _load_manifest(train_manifest)
    else:
        train_ds = AIGCDataset(src / "train", transform=None)
        reals, fakes = build_type_groups(train_ds)
        train_rows = _rows_from_indices(
            train_ds, sample_balanced(reals, fakes, n, seed=seed),
        )
        _write_manifest(train_manifest, train_rows)

    val_ds = AIGCDataset(src / "val", transform=None)
    val_reals, val_fakes = build_type_groups(val_ds)
    val_rows = _rows_from_indices(
        val_ds, sample_balanced(val_reals, val_fakes, n_val, seed=seed + 2),
    )
    _write_manifest(val_manifest, val_rows)

    train_real, train_fake = _counts(train_rows)
    val_real, val_fake = _counts(val_rows)
    summary = {
        "source": str(src.resolve()),
        "n_requested": n,
        "n_val_requested": n_val,
        "train": len(train_rows),
        "val": len(val_rows),
        "seed": seed,
        "train_source": "wildfake_images_train",
        "val_source": "wildfake_images_val",
        "train_real": train_real,
        "train_fake": train_fake,
        "val_real": val_real,
        "val_fake": val_fake,
    }
    (dest / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build a type-balanced fusion subset (train from train, val from val).",
    )
    p.add_argument("--src", default="data/datasets/WildFake_images")
    p.add_argument("--dest", default="data/datasets/WildFake_fusion_50k")
    p.add_argument("--n", type=int, default=50_000)
    p.add_argument("--n-val", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--resample-train",
        action="store_true",
        help="Redraw train even if dest/train/manifest.csv already exists.",
    )
    args = p.parse_args()
    summary = prepare(
        Path(args.src), Path(args.dest),
        n=args.n, n_val=args.n_val, seed=args.seed,
        keep_train=not args.resample_train,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
