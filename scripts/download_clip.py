"""Download CLIP ViT-H/14 (DFN-5B weights) from hf-mirror.com."""

import os
import sys
import urllib.request
from pathlib import Path

REPO_ID = "apple/DFN5B-CLIP-ViT-H-14"
FILENAME = "open_clip_pytorch_model.bin"
MIRROR = "https://hf-mirror.com"
OUT_DIR = Path(__file__).parent.parent / "weights" / "clip_h"
URL = f"{MIRROR}/{REPO_ID}/resolve/main/{FILENAME}"


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        done = int(pct / 2)
        bar = "█" * done + "░" * (50 - done)
        gb = downloaded / 1e9
        total_gb = total_size / 1e9
        print(f"\r  [{bar}] {pct:5.1f}%  {gb:.2f}/{total_gb:.2f} GB", end="", flush=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / FILENAME

    if out_path.exists():
        print(f"Already exists: {out_path}")
        sys.exit(0)

    tmp_path = out_path.with_suffix(".bin.tmp")
    print(f"Downloading {FILENAME} (~3.9 GB)")
    print(f"  URL : {URL}")
    print(f"  Dest: {out_path}")

    try:
        urllib.request.urlretrieve(URL, tmp_path, reporthook=_progress)
        print()
        tmp_path.rename(out_path)
        print(f"Saved to {out_path}")
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        print(f"\nFailed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
