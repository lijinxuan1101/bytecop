"""Download DINOv3 ViT-H+/16 weights.

DINOv3 is a gated model on HuggingFace. Authenticate first via either:
    1. huggingface-cli login   (writes ~/.cache/huggingface/token)
    2. export HF_TOKEN=hf_xxxxxxxxxxxxxxxx

Uses hf-mirror.com by default (fast in CN). Set HF_ENDPOINT=https://huggingface.co
to switch to the official endpoint.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import get_token, hf_hub_download  # noqa: E402

REPO_ID = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
OUT_DIR = Path(__file__).parent.parent / "weights" / "dino_h"

FILES = [
    ("model.safetensors", "~3.2 GB"),
    ("config.json", ""),
    ("preprocessor_config.json", ""),
]


def main() -> None:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or get_token()
    )
    if not token:
        print("ERROR: no HF token found. DINOv3 is a gated model and requires auth.", file=sys.stderr)
        print("Either run `huggingface-cli login`, or set HF_TOKEN env var.", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Endpoint: {os.environ['HF_ENDPOINT']}")
    print(f"Dest    : {OUT_DIR}")
    print(f"Repo    : {REPO_ID}\n")

    for filename, size_hint in FILES:
        target = OUT_DIR / filename
        if target.exists() and target.stat().st_size > 0:
            print(f"Already exists: {filename}")
            continue

        hint = f" ({size_hint})" if size_hint else ""
        print(f"Downloading {filename}{hint} ...")
        try:
            cached_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                token=token,
                local_dir=str(OUT_DIR),
                force_download=False,
            )
            print(f"  Saved to {cached_path}\n")
        except Exception as exc:
            print(f"  Failed: {exc}", file=sys.stderr)
            sys.exit(1)

    print("All done.")


if __name__ == "__main__":
    main()
