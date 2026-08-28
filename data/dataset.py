"""Generic AIGC image dataset.

Supports two loading modes:

1. **Directory mode** – a root folder with sub-folders ``real/`` and ``fake/``
   (or any pair of label names configured via ``label_map``).
2. **Manifest mode** – a TSV/CSV file with columns ``path`` and ``label``
   (0 = real, 1 = fake).

Labels are always integers: 0 = real, 1 = AI-generated.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset

_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class AIGCDataset(Dataset):
    """Dataset for AI-generated image detection.

    Args:
        root: Path to the dataset root directory (for directory mode) or a
            manifest file (for manifest mode).
        transform: Optional callable applied to each ``PIL.Image`` before
            returning.  Receives and returns a ``PIL.Image``; conversion to
            tensor is the caller's responsibility.
        label_map: Mapping from sub-folder name to integer label used in
            directory mode.  Defaults to ``{"real": 0, "fake": 1}``.
        extensions: Set of lower-case file extensions to accept.
    """

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
        self.label_map = label_map or {"real": 0, "fake": 1}
        self.extensions = extensions

        self.samples: list[tuple[Path, int]] = []
        if self.root.is_file():
            self._load_manifest(self.root)
        else:
            self._load_directory(self.root)

        if not self.samples:
            raise FileNotFoundError(
                f"No images found under {self.root!r}. "
                "Expected sub-folders matching label_map keys, or a manifest file."
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
                if path.suffix.lower() in self.extensions:
                    self.samples.append((path, label))

    def _load_manifest(self, manifest: Path) -> None:
        delimiter = "\t" if manifest.suffix.lower() == ".tsv" else ","
        with manifest.open(newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                path = Path(row["path"])
                label = int(row["label"])
                if path.suffix.lower() in self.extensions:
                    self.samples.append((path, label))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        path, label = self.samples[idx]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def class_counts(self) -> dict[int, int]:
        """Return a dict mapping label → number of samples."""
        counts: dict[int, int] = {}
        for _, label in self.samples:
            counts[label] = counts.get(label, 0) + 1
        return counts
