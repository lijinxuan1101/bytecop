"""Generic AIGC image dataset.

Supports directory mode and manifest mode.

Labels:
    0 = real
    1 = AI-generated
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset

from data.transforms import apply_train_policy


def _infer_architecture(path: Path, class_root: Path) -> str:
    """First directory under ``real/`` or ``fake/`` is the generator type."""
    try:
        rel = path.relative_to(class_root)
    except ValueError:
        return ""
    parts = rel.parts
    return parts[0] if len(parts) >= 2 else ""


_IMG_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


class AIGCDataset(Dataset):
    """Dataset for AI-generated image detection."""

    def __init__(
        self,
        root: str | Path,
        *,
        transform: Callable[[Image.Image], Image.Image] | None = None,
        label_map: dict[str, int] | None = None,
        extensions: set[str] = _IMG_EXTENSIONS,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.label_map = label_map or {
            "real": 0,
            "fake": 1,
        }
        self.extensions = extensions

        self.samples: list[tuple[Path, int]] = []
        self.architectures: list[str] = []

        nested_manifest = self.root / "manifest.csv"
        if self.root.is_file():
            self._load_manifest(self.root)
        elif nested_manifest.is_file():
            self._load_manifest(nested_manifest)
        else:
            self._load_directory(self.root)

        if not self.samples:
            raise FileNotFoundError(
                f"No images found under {self.root!r}. "
                "Expected sub-folders matching label_map keys, "
                "or a manifest file."
            )

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load_directory(self, root: Path) -> None:
        for folder_name, label in self.label_map.items():
            folder = root / folder_name

            if not folder.is_dir():
                continue

            for path in sorted(folder.rglob("*")):
                if path.is_file() and path.suffix.lower() in self.extensions:
                    self.samples.append((path, label))
                    self.architectures.append(_infer_architecture(path, folder))

    def _load_manifest(self, manifest: Path) -> None:
        delimiter = (
            "\t"
            if manifest.suffix.lower() == ".tsv"
            else ","
        )

        with manifest.open(newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)

            for row in reader:
                path = Path(row["path"])
                label = int(row["label"])
                arch = (
                    row.get("architecture")
                    or row.get("Architecture")
                    or ""
                )

                if path.suffix.lower() in self.extensions:
                    self.samples.append((path, label))
                    self.architectures.append(str(arch))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        """Load one image.

        If an image is corrupted or unreadable, skip it immediately and
        try the next sample. No full-dataset pre-scan is performed.
        """

        total = len(self.samples)

        for _ in range(total):
            path, label = self.samples[idx]

            try:
                with Image.open(path) as img:
                    image = img.convert("RGB")

                if self.transform is not None:
                    image = self.transform(image)

                return image, label

            except Exception as exc:
                print(
                    f"[dataset] Skipping unreadable image: "
                    f"{path} ({type(exc).__name__}: {exc})",
                    flush=True,
                )

                idx = (idx + 1) % total

        raise RuntimeError(
            f"No readable images found under {self.root}"
        )

    def class_counts(self) -> dict[int, int]:
        """Return a mapping from label to sample count."""

        counts: dict[int, int] = {}

        for _, label in self.samples:
            counts[label] = counts.get(label, 0) + 1

        return counts


class FlaggedAugmentDataset(Dataset):
    """``AIGCDataset`` plus the official train policy, with a clean/robust flag.

    Each item is ``(tensor, label, is_clean)``. ``is_clean=1`` means no official
    transform was applied. ``clean_prob=0`` forces one random official transform
    (used for a cheap robust val pass).
    """

    def __init__(
        self,
        root: str | Path,
        *,
        tensor_transform: Callable[[Image.Image], object],
        augment: bool,
        clean_prob: float,
    ) -> None:
        self.base = AIGCDataset(root, transform=None)
        self.tensor_transform = tensor_transform
        self.augment = augment
        self.clean_prob = clean_prob

    def __len__(self) -> int:
        return len(self.base)

    def class_counts(self) -> dict[int, int]:
        return self.base.class_counts()

    def __getitem__(self, idx: int) -> tuple[object, int, int]:
        image, label = self.base[idx]
        is_clean = True
        if self.augment:
            image, is_clean = apply_train_policy(image, clean_prob=self.clean_prob)
        return self.tensor_transform(image), label, int(is_clean)