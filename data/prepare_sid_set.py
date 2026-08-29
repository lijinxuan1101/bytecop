"""Convert SID_Set parquet shards to an image directory tree.

Reads parquet shards one at a time, extracts images to JPEG files, then
deletes each shard after extraction — so the peak extra disk usage is only
~one shard (~500 MB) rather than the full dataset size.

Output layout
-------------
<dest>/
    train/
        real/   ... Real images
        fake/   ... Full Synthetic images
    val/
        real/
        fake/
    calibration/
        real/
        fake/

Split strategy (by shard index, not by image, to prevent data leakage):
    - Last 10 shards  → calibration  (independent split, never used for training)
    - Prev 20 shards  → val
    - Remaining       → train

Usage
-----
    python data/prepare_sid_set.py \
        --src  data/datasets/SID_Set \
        --dest data/datasets/SID_Set_images \
        [--delete-parquet]   # delete each shard after extracting (saves disk space)
        [--limit 20]         # only process first N shards (for smoke testing)
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm


# ------------------------------------------------------------------
# Label mapping
# ------------------------------------------------------------------

# SID_Set integer label mapping (confirmed from HuggingFace metadata):
#   0 = Real, 1 = Full Synthetic (fake), 2 = Tampered (skip)
_SID_INT_LABEL: dict[int, str | None] = {
    0: "real",
    1: "fake",
    2: None,   # Tampered — skip
}


def _to_split_label(raw_label) -> str | None:
    """Map a SID_Set label (int or str) to 'real' or 'fake', or None to skip."""
    if isinstance(raw_label, int):
        return _SID_INT_LABEL.get(raw_label)
    low = str(raw_label).lower()
    if "real" in low:
        return "real"
    if "synthetic" in low or "fake" in low or "generated" in low:
        return "fake"
    return None


# ------------------------------------------------------------------
# Shard → split assignment
# ------------------------------------------------------------------

def _assign_splits(
    shards: list[Path],
    *,
    val_frac: float = 0.15,
    cal_frac: float = 0.10,
) -> dict[Path, str]:
    """Assign each shard to train / val / calibration by index.

    Uses fractions so the split works correctly even with very few shards.
    At least 1 shard is always assigned to each non-empty split.
    """
    shards = sorted(shards)
    n = len(shards)

    n_cal = max(1, round(n * cal_frac))
    n_val = max(1, round(n * val_frac))
    # Make sure we don't exceed total shard count
    if n_cal + n_val >= n:
        n_cal = max(1, n // 3)
        n_val = max(1, n // 3)
        if n_cal + n_val >= n:
            n_val = 0  # only train + calibration when n=1 or n=2

    assignment: dict[Path, str] = {}
    for i, shard in enumerate(shards):
        if i >= n - n_cal:
            assignment[shard] = "calibration"
        elif i >= n - n_cal - n_val:
            assignment[shard] = "val"
        else:
            assignment[shard] = "train"
    return assignment


# ------------------------------------------------------------------
# Image extraction
# ------------------------------------------------------------------

def _extract_image(raw) -> Image.Image | None:
    """Extract a PIL Image from a parquet cell (bytes, dict, or PIL Image)."""
    if raw is None:
        return None
    if isinstance(raw, bytes):
        try:
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            return None
    if isinstance(raw, dict):
        # HuggingFace image feature: {"bytes": b"...", "path": "..."}
        data = raw.get("bytes") or raw.get("data")
        if data:
            try:
                return Image.open(io.BytesIO(data)).convert("RGB")
            except Exception:
                return None
    if hasattr(raw, "convert"):
        return raw.convert("RGB")
    return None


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def prepare(args: argparse.Namespace) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pyarrow not installed. Run: pip install pyarrow")

    src = Path(args.src)
    dest = Path(args.dest)

    shards = sorted((src / "data").glob("*.parquet"))
    if not shards:
        sys.exit(f"No parquet files found under {src / 'data'}")

    if args.limit:
        shards = shards[: args.limit]

    print(f"Found {len(shards)} shards under {src / 'data'}")
    split_map = _assign_splits(shards, val_frac=args.val_frac, cal_frac=args.cal_frac)
    counts_by_split = {}
    for p, s in split_map.items():
        counts_by_split.setdefault(s, []).append(p.name)
    for s in ("train", "val", "calibration"):
        print(f"  {s}: {len(counts_by_split.get(s, []))} shards")

    split_counts: dict[str, dict[str, int]] = {
        s: {"real": 0, "fake": 0, "skipped": 0}
        for s in ("train", "val", "calibration")
    }

    for shard in shards:
        split = split_map[shard]
        print(f"\n[{split}] Processing {shard.name} …")

        table = pq.read_table(shard)
        df = table.to_pydict()

        # Detect column names
        img_col = next(
            (c for c in ("image", "img", "pixel_values") if c in df), None
        )
        lbl_col = next(
            (c for c in ("label", "category", "class", "type") if c in df), None
        )
        if img_col is None:
            print(f"  WARNING: no image column found in {shard.name}, skipping.")
            continue

        n_rows = len(df[img_col])
        for i in tqdm(range(n_rows), desc=f"  Rows", leave=False):
            raw_img = df[img_col][i]
            raw_lbl = df[lbl_col][i] if lbl_col else 1

            label = _to_split_label(raw_lbl)
            if label is None:
                split_counts[split]["skipped"] += 1
                continue

            img = _extract_image(raw_img)
            if img is None:
                split_counts[split]["skipped"] += 1
                continue

            out_dir = dest / split / label
            out_dir.mkdir(parents=True, exist_ok=True)

            shard_idx = shard.stem.split("-")[1]  # e.g. "00042"
            filename = f"{shard_idx}_{i:06d}.jpg"
            img.save(out_dir / filename, format="JPEG", quality=95)
            split_counts[split][label] += 1

        del df, table

        if args.delete_parquet:
            shard.unlink()
            print(f"  Deleted {shard.name}")

    # Summary
    print("\n" + "=" * 50)
    print("Done. Image counts:")
    total_images = 0
    for split, counts in split_counts.items():
        r, f, s = counts["real"], counts["fake"], counts["skipped"]
        total_images += r + f
        print(f"  {split:<14s} real={r:>6,}  fake={f:>6,}  skipped={s:>6,}")
    print(f"  Total images: {total_images:,}")

    # Write manifest
    manifest_path = dest / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as fp:
        json.dump({"splits": split_counts, "source_shards": len(shards)}, fp, indent=2)
    print(f"\nManifest saved to {manifest_path}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SID_Set parquet shards to image folders.")
    parser.add_argument("--src",  required=True, help="SID_Set download directory (contains data/*.parquet).")
    parser.add_argument("--dest", required=True, help="Output directory for image folders.")
    parser.add_argument(
        "--delete-parquet", action="store_true", dest="delete_parquet",
        help="Delete each parquet shard after extracting its images (saves disk space).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N shards (useful for smoke testing).",
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.15, dest="val_frac",
        help="Fraction of shards assigned to the validation split (default 0.15).",
    )
    parser.add_argument(
        "--cal-frac", type=float, default=0.10, dest="cal_frac",
        help="Fraction of shards assigned to the calibration split (default 0.10).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    prepare(_parse_args())
