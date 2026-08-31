"""Build a SID-style train/val tree for WildFake (symlinks + manifests).

Reads official label CSVs, maps Image_path onto WildFake_extracted, then
splits **inside each Category**:
    n >= 50  → 98% train / 2% val (at least one val image)
    n <  50  → all train

Every Category that resolves at least one file goes into train (covers all
generator types). Official train/test and the four cross-* splits are ignored.
``wukong`` and unresolved paths are dropped.

Output
------
<dest>/
    train/{real,fake}/<Architecture>/<Category>/<file>   # symlink
    val/{real,fake}/...
    train/manifest.csv
    val/manifest.csv
    manifest.json

Usage
-----
    python data/prepare_wildfake.py
    python data/prepare_wildfake.py --dry-run
    python data/prepare_wildfake.py --no-links          # manifests only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_SKIP_ARCH = {"wukong"}


def _norm_rel(image_path: str) -> str:
    return image_path.replace("\\", "/").lstrip("./")


def candidate_rels(image_path: str) -> list[str]:
    """Likely relative paths under WildFake_extracted for one CSV Image_path."""
    p = _norm_rel(image_path)
    parts = [x for x in p.split("/") if x]
    out: list[str] = [p]
    if len(parts) < 2:
        return out
    family = parts[0]
    if family in {"GAN_based", "Other_based"}:
        out.append("/".join([family, *parts]))
    if family == "Real":
        src = parts[1]
        out.append("/".join([family, src, src, *parts[2:]]))
    if family == "Diffusion_based":
        arch = parts[1]
        out.append("/".join([family, arch, arch, *parts[2:]]))
        if arch == "SD" and len(parts) >= 3:
            weight = parts[2]
            out.append("/".join([family, "SD", weight, weight, *parts[3:]]))
    seen: set[str] = set()
    uniq: list[str] = []
    for item in out:
        if item not in seen:
            seen.add(item)
            uniq.append(item)
    return uniq


def walk_files(root: Path) -> list[str]:
    """Absolute paths of image files under ``root``."""
    found: list[str] = []
    stack = [str(root)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        if Path(entry.name).suffix.lower() in _IMG_EXT:
                            found.append(entry.path)
        except FileNotFoundError:
            continue
    return found


class PathIndex:
    """Indexes extracted files for CSV path resolution."""

    def __init__(self, extracted: Path) -> None:
        self.root = extracted.resolve()
        self.by_rel: dict[str, Path] = {}
        self.by_parent_name: dict[tuple[str, str], list[Path]] = defaultdict(list)
        self.by_name: dict[str, list[Path]] = defaultdict(list)

    def build(self) -> int:
        files = walk_files(self.root)
        for abs_s in tqdm(files, desc="index extracted", unit="img"):
            path = Path(abs_s)
            rel = path.relative_to(self.root).as_posix()
            self.by_rel[rel] = path
            name = path.name
            self.by_name[name].append(path)
            self.by_parent_name[(path.parent.name, name)].append(path)
        return len(files)

    def resolve(self, image_path: str, *, architecture: str, category: str) -> Path | None:
        name = Path(_norm_rel(image_path)).name
        for cand in candidate_rels(image_path):
            hit = self.by_rel.get(cand)
            if hit is not None:
                return hit
            parent = Path(cand).parent.name
            hits = self.by_parent_name.get((parent, name), [])
            if len(hits) == 1:
                return hits[0]
        parent_hits = self.by_parent_name.get((category, name), [])
        if len(parent_hits) == 1:
            return parent_hits[0]
        if len(parent_hits) > 1:
            narrowed = [p for p in parent_hits if architecture in p.as_posix()]
            if len(narrowed) == 1:
                return narrowed[0]
            if narrowed:
                return narrowed[0]
        names = self.by_name.get(name, [])
        if len(names) == 1:
            return names[0]
        if len(names) > 1:
            scored: list[Path] = []
            for path in names:
                posix = path.as_posix()
                if architecture and architecture not in posix:
                    continue
                if category and category not in posix:
                    continue
                scored.append(path)
            if len(scored) == 1:
                return scored[0]
            if len(scored) > 1:
                return scored[0]
            arch_only = [p for p in names if architecture in p.as_posix()]
            if len(arch_only) == 1:
                return arch_only[0]
        return None


def _safe_name(text: str) -> str:
    cleaned = text.replace("/", "_").replace("\\", "_").strip() or "unknown"
    return cleaned


def _unique_link(dest_file: Path) -> Path:
    if not dest_file.exists() and not dest_file.is_symlink():
        return dest_file
    stem, suffix = dest_file.stem, dest_file.suffix
    for i in range(1, 10000):
        cand = dest_file.with_name(f"{stem}__{i}{suffix}")
        if not cand.exists() and not cand.is_symlink():
            return cand
    raise RuntimeError(f"Could not uniquify {dest_file}")


def load_label_rows(csv_dir: Path, *, limit: int | None) -> list[dict]:
    rows: list[dict] = []
    files = sorted(csv_dir.glob("*.csv"))
    for path in files:
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            n = 0
            for row in reader:
                rows.append(row)
                n += 1
                if limit is not None and n >= limit:
                    break
    return rows


def split_group(paths: list[Path], *, seed: int, val_frac: float, min_n: int) -> tuple[list[Path], list[Path]]:
    ordered = sorted(paths, key=lambda p: p.as_posix())
    rng = random.Random(seed)
    rng.shuffle(ordered)
    if len(ordered) < min_n:
        return ordered, []
    n_val = max(1, int(round(len(ordered) * val_frac)))
    n_val = min(n_val, len(ordered) - 1)
    return ordered[n_val:], ordered[:n_val]


def write_manifest_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["path", "label", "architecture", "category"],
        )
        writer.writeheader()
        writer.writerows(rows)


def prepare(args: argparse.Namespace) -> int:
    extracted = Path(args.extracted).expanduser().resolve()
    csv_dir = Path(args.csv_dir).expanduser().resolve()
    dest = Path(args.dest).expanduser().resolve()
    if not extracted.is_dir():
        print(f"ERROR: extracted root not found: {extracted}", file=sys.stderr)
        return 1
    if not csv_dir.is_dir():
        print(f"ERROR: label CSV dir not found: {csv_dir}", file=sys.stderr)
        return 1

    print(f"extracted : {extracted}")
    print(f"csv       : {csv_dir}")
    print(f"dest      : {dest}")

    index = PathIndex(extracted)
    n_files = index.build()
    print(f"indexed   : {n_files:,} files")

    raw_rows = load_label_rows(csv_dir, limit=args.limit)
    print(f"csv rows  : {len(raw_rows):,}")

    grouped: dict[tuple[int, str, str], list[Path]] = defaultdict(list)
    skipped = {"wukong": 0, "unresolved": 0, "bad_label": 0}
    unresolved_arch: dict[str, int] = defaultdict(int)

    for row in tqdm(raw_rows, desc="resolve csv", unit="row"):
        try:
            is_fake = int(row["IsFake"])
        except (KeyError, ValueError, TypeError):
            skipped["bad_label"] += 1
            continue
        if is_fake not in (0, 1):
            skipped["bad_label"] += 1
            continue
        arch = str(row.get("Architecture") or "unknown")
        category = str(row.get("Category") or "unknown")
        image_path = str(row.get("Image_path") or "")
        if arch in _SKIP_ARCH or "wukong" in image_path.lower():
            skipped["wukong"] += 1
            continue
        resolved = index.resolve(image_path, architecture=arch, category=category)
        if resolved is None:
            skipped["unresolved"] += 1
            unresolved_arch[arch] += 1
            continue
        grouped[(is_fake, arch, category)].append(resolved)

    split_rows: dict[str, list[dict]] = {"train": [], "val": []}
    per_type: dict[str, dict] = {}
    for (is_fake, arch, category), paths in sorted(grouped.items()):
        # One file may appear in several CSV rows; keep unique paths.
        unique: list[Path] = []
        seen: set[str] = set()
        for path in paths:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            unique.append(path)
        train_paths, val_paths = split_group(
            unique, seed=args.seed, val_frac=args.val_frac, min_n=args.min_split_n,
        )
        label = 1 if is_fake else 0
        kind = "fake" if is_fake else "real"
        key = f"{kind}/{arch}/{category}"
        per_type[key] = {
            "label": label,
            "architecture": arch,
            "category": category,
            "n": len(unique),
            "train": len(train_paths),
            "val": len(val_paths),
        }
        for split_name, split_paths in (("train", train_paths), ("val", val_paths)):
            for src in split_paths:
                rel = (
                    Path(split_name) / kind / _safe_name(arch)
                    / _safe_name(category) / src.name
                )
                dest_file = dest / rel
                split_rows[split_name].append({
                    "path": str(src),
                    "label": str(label),
                    "architecture": arch,
                    "category": category,
                    "dest": str(dest_file),
                    "kind": kind,
                })

    summary = {
        "extracted": str(extracted),
        "csv_dir": str(csv_dir),
        "dest": str(dest),
        "seed": args.seed,
        "val_frac": args.val_frac,
        "min_split_n": args.min_split_n,
        "batch_sampler": "type_balanced",
        "indexed_files": n_files,
        "csv_rows": len(raw_rows),
        "skipped": skipped,
        "unresolved_by_architecture": dict(sorted(unresolved_arch.items())),
        "splits": {
            name: {
                "total": len(rows),
                "real": sum(1 for r in rows if r["label"] == "0"),
                "fake": sum(1 for r in rows if r["label"] == "1"),
            }
            for name, rows in split_rows.items()
        },
        "types": per_type,
    }

    print("skipped   :", skipped)
    print("train     :", summary["splits"]["train"])
    print("val       :", summary["splits"]["val"])
    if unresolved_arch:
        print("unresolved by architecture (top 15):")
        for arch, n in sorted(unresolved_arch.items(), key=lambda x: -x[1])[:15]:
            print(f"  {arch:20s} {n:,}")

    if args.dry_run:
        print("dry-run: not writing dest")
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    for split_name, rows in split_rows.items():
        manifest_rows = [
            {
                "path": r["path"],
                "label": r["label"],
                "architecture": r["architecture"],
                "category": r["category"],
            }
            for r in rows
        ]
        write_manifest_csv(dest / split_name / "manifest.csv", manifest_rows)
        if args.no_links:
            continue
        for row in tqdm(rows, desc=f"symlink {split_name}", unit="img"):
            dest_file = Path(row["dest"])
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            dest_file = _unique_link(dest_file)
            src = Path(row["path"])
            if dest_file.is_symlink() or dest_file.exists():
                continue
            os.symlink(src, dest_file)

    with (dest / "manifest.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {dest / 'manifest.json'}")
    return 0


def _parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extracted",
        default=str(Path.home() / "techjam/raw/WildFake_extracted"),
        help="Unzipped WildFake image root",
    )
    parser.add_argument(
        "--csv-dir",
        default=str(repo / "data/WildFake/label_csv_files"),
        help="Official per-source label CSVs",
    )
    parser.add_argument(
        "--dest",
        default=str(repo / "data/datasets/WildFake_images"),
        help="SID-style output root (symlinks + manifests)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.02)
    parser.add_argument("--min-split-n", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None, help="Max rows per CSV (smoke)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-links", action="store_true", help="Write manifests only")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(prepare(_parse_args()))
