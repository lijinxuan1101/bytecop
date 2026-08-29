"""Download CIFAKE and sample it into an image directory tree.

CIFAKE (source: dragonintelligence/CIFAKE-image-dataset on HuggingFace) contains
60,000 real images (from CIFAR-10) and 60,000 AI-generated images. This script
downloads the two parquet shards, samples ``--n-total`` images balanced between
the two classes, and writes them out as JPEG files split into four folders.

Output layout
-------------
<dest>/
    train/{real,fake}/
    val/{real,fake}/
    test/{real,fake}/
    calibration/{real,fake}/

Defaults (--n-total=10000, --val=1000, --test=1000, --cal=1000):
    train:        7000 (3500 real + 3500 fake)
    val:          1000 ( 500 real +  500 fake)
    test:         1000 ( 500 real +  500 fake)
    calibration:  1000 ( 500 real +  500 fake)

Class labels
------------
    0 = real  (label field 'REAL' or int 0)
    1 = fake  (label field 'FAKE' or int 1)

CIFAKE native resolution is 32x32. Images are stored as-is; upsampling to a
model-friendly size (e.g. 224 for CLIP) happens in the training pipeline.

Usage
-----
    python data/prepare_cifake.py --dest data/datasets/CIFAKE_images
    python data/prepare_cifake.py --dest data/datasets/CIFAKE_images --seed 42
    python data/prepare_cifake.py --dest data/datasets/CIFAKE_images --n-total 4000
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import hf_hub_download  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402


REPO_ID = "dragonintelligence/CIFAKE-image-dataset"
PARQUETS = [
    "data/train-00000-of-00001.parquet",
    "data/test-00000-of-00001.parquet",
]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _to_int_label(raw) -> int | None:
    """Return 0 for real, 1 for fake, or None if unknown."""
    if isinstance(raw, int):
        return raw if raw in (0, 1) else None
    s = str(raw).strip().upper()
    if s in ("0", "REAL"):
        return 0
    if s in ("1", "FAKE"):
        return 1
    return None


def _extract_image(raw) -> Image.Image | None:
    """Decode a parquet image cell into a PIL Image."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        raw = raw.get("bytes") or raw.get("data")
    if isinstance(raw, (bytes, bytearray)):
        try:
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            return None
    if hasattr(raw, "convert"):
        return raw.convert("RGB")
    return None


def _download_parquets(dest_dir: Path) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name in PARQUETS:
        print(f"Downloading {name} ...")
        p = hf_hub_download(
            repo_id=REPO_ID,
            filename=name,
            repo_type="dataset",
            local_dir=str(dest_dir),
        )
        paths.append(Path(p))
    return paths


def _collect_rows(parquets: list[Path]) -> tuple[list[bytes], list[bytes]]:
    """Return (real_rows, fake_rows) as lists of raw image bytes."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pyarrow not installed. Run: pip install pyarrow")

    real_rows: list[bytes] = []
    fake_rows: list[bytes] = []

    for shard in parquets:
        print(f"Reading {shard.name} ...")
        table = pq.read_table(shard)
        df = table.to_pydict()

        img_col = next((c for c in ("image", "img", "pixel_values") if c in df), None)
        lbl_col = next((c for c in ("label", "labels", "class") if c in df), None)
        if img_col is None or lbl_col is None:
            print(f"  WARNING: could not find image/label columns in {shard.name}")
            continue

        n = len(df[img_col])
        for i in tqdm(range(n), desc="  Rows", leave=False):
            lbl = _to_int_label(df[lbl_col][i])
            if lbl is None:
                continue
            raw = df[img_col][i]
            if isinstance(raw, dict):
                raw = raw.get("bytes") or raw.get("data")
            if not isinstance(raw, (bytes, bytearray)):
                continue
            (real_rows if lbl == 0 else fake_rows).append(bytes(raw))

        del df, table

    return real_rows, fake_rows


def _write_split(
    rows: list[bytes],
    label_name: str,
    split_name: str,
    dest_root: Path,
    start_idx: int,
) -> int:
    out_dir = dest_root / split_name / label_name
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for i, raw in enumerate(rows):
        img = _extract_image(raw)
        if img is None:
            continue
        fname = f"{start_idx + i:06d}.jpg"
        img.save(out_dir / fname, format="JPEG", quality=95)
        written += 1
    return written


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def prepare(args: argparse.Namespace) -> None:
    dest_root = Path(args.dest)
    dest_root.mkdir(parents=True, exist_ok=True)
    parquet_dir = dest_root / "_parquets"

    # 1. Download
    parquets = _download_parquets(parquet_dir)

    # 2. Load into memory (CIFAKE is small; ~120 MB uncompressed as 32x32 RGB)
    real_rows, fake_rows = _collect_rows(parquets)
    print(f"\nLoaded: real={len(real_rows):,}  fake={len(fake_rows):,}")

    # 3. Compute per-class split sizes
    train_n, val_n, test_n, cal_n = args.train, args.val, args.test, args.cal
    total = train_n + val_n + test_n + cal_n
    assert total == args.n_total, (
        f"train+val+test+cal ({total}) must equal --n-total ({args.n_total})"
    )
    if total % 2 != 0:
        sys.exit(f"Total split size {total} must be even (real/fake 1:1).")

    per_class = {
        "train":       train_n // 2,
        "val":         val_n // 2,
        "test":        test_n // 2,
        "calibration": cal_n // 2,
    }
    per_class_total = sum(per_class.values())
    if per_class_total > min(len(real_rows), len(fake_rows)):
        sys.exit(
            f"Need {per_class_total} images per class, but only have "
            f"real={len(real_rows)}, fake={len(fake_rows)}."
        )

    print("\nSplit sizes (per class):")
    for split, n in per_class.items():
        print(f"  {split:<12s} {n:>5d}")

    # 4. Deterministic sample + split
    rng = random.Random(args.seed)
    rng.shuffle(real_rows)
    rng.shuffle(fake_rows)

    counts: dict[str, dict[str, int]] = {}
    cursor = 0
    for split, n in per_class.items():
        real_slice = real_rows[cursor : cursor + n]
        fake_slice = fake_rows[cursor : cursor + n]
        cursor += n

        r_written = _write_split(real_slice, "real", split, dest_root, start_idx=0)
        f_written = _write_split(fake_slice, "fake", split, dest_root, start_idx=n)
        counts[split] = {"real": r_written, "fake": f_written}
        print(f"  wrote {split:<12s} real={r_written}  fake={f_written}")

    # 5. Manifest
    manifest = {
        "source": REPO_ID,
        "n_total": args.n_total,
        "seed": args.seed,
        "splits": counts,
    }
    manifest_path = dest_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")

    # 6. Delete parquet cache (unless --keep-parquet)
    if not args.keep_parquet:
        for p in parquets:
            p.unlink(missing_ok=True)
        # Best-effort remove the _parquets folder if empty
        try:
            for sub in sorted(parquet_dir.rglob("*"), reverse=True):
                if sub.is_dir():
                    sub.rmdir()
            parquet_dir.rmdir()
        except OSError:
            pass
        print("Deleted downloaded parquet shards.")

    print("\nDone.")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare a CIFAKE image-folder dataset.")
    p.add_argument("--dest", required=True, help="Output root directory.")
    p.add_argument("--n-total", type=int, default=10_000, dest="n_total",
                   help="Total images to sample (default 10000).")
    p.add_argument("--train", type=int, default=7000)
    p.add_argument("--val",   type=int, default=1000)
    p.add_argument("--test",  type=int, default=1000)
    p.add_argument("--cal",   type=int, default=1000)
    p.add_argument("--seed",  type=int, default=42)
    p.add_argument("--keep-parquet", action="store_true", dest="keep_parquet",
                   help="Keep the downloaded parquet cache instead of deleting.")
    return p.parse_args()


if __name__ == "__main__":
    prepare(_parse_args())
