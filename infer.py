"""Official directory inference — Spatial tower only.

    python infer.py --input /path/to/images --output predictions.json

Contest JSON is ``[{image_path, pred}, ...]``. ``pred`` is P(AI-generated).
Pass ``--full`` to also write ``logit`` / ``label`` for the visualization layer.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from serve.spatial_backend import DEFAULT_CKPT, SpatialDetector, official_records


def infer(args: argparse.Namespace) -> None:
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"{input_dir} is not a directory.")

    detector = SpatialDetector(ckpt=args.ckpt, temperature=args.temperature)
    print(json.dumps(detector.info(), indent=2))

    records = detector.score_dir(
        input_dir,
        batch_size=args.batch_size,
        workers=args.workers,
        show_progress=True,
    )
    payload = records if args.full else official_records(records)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved {len(payload)} predictions to {out_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a directory with the Spatial CLIP-H detector.",
    )
    parser.add_argument("--input", required=True, help="Directory of images to score.")
    parser.add_argument("--output", required=True, help="Output JSON file path.")
    parser.add_argument(
        "--ckpt",
        default=str(DEFAULT_CKPT),
        help="Spatial checkpoint (default: WildFake best.pt).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Logit temperature. WildFake run has no calibrator; leave at 1.0.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Write logit/label as well as pred (visualization JSON).",
    )
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    return parser.parse_args()


if __name__ == "__main__":
    infer(_parse_args())
