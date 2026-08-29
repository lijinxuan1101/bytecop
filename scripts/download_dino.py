"""Download DINOv3 ViT-H+/16 weights from hf-mirror.com."""

import os
import sys
import urllib.request
from pathlib import Path

REPO_ID = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
MIRROR = "https://hf-mirror.com"
OUT_DIR = Path(__file__).parent.parent / "weights" / "dino_h"

FILES = [
    ("model.safetensors", "~3.2 GB"),
    ("config.json", ""),
    ("preprocessor_config.json", ""),
]


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        done = int(pct / 2)
        bar = "█" * done + "░" * (50 - done)
        gb = downloaded / 1e9
        total_gb = total_size / 1e9
        print(f"\r  [{bar}] {pct:5.1f}%  {gb:.2f}/{total_gb:.2f} GB", end="", flush=True)


def download_file(filename: str, size_hint: str) -> None:
    out_path = OUT_DIR / filename
    if out_path.exists():
        print(f"Already exists: {filename}")
        return

    url = f"{MIRROR}/{REPO_ID}/resolve/main/{filename}"
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    hint = f"  ({size_hint})" if size_hint else ""
    print(f"Downloading {filename}{hint}")
    print(f"  URL : {url}")

    try:
        urllib.request.urlretrieve(url, tmp_path, reporthook=_progress)
        print()
        tmp_path.rename(out_path)
        print(f"Saved to {out_path}")
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        print(f"\nFailed: {exc}")
        sys.exit(1)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, size_hint in FILES:
        download_file(filename, size_hint)
    print("\nAll done.")


if __name__ == "__main__":
    main()
