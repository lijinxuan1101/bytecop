"""Ratio-controlled real/fake batch sampler with per-arch fake round-robin.

One epoch consumes every fake once (no replacement). Reals cycle. Each
batch holds ``round(batch_size * real_frac)`` reals and the rest fakes;
the fake part walks Architecture types in order, one image each, then
repeats. Used when ``manifest.json`` at the
dataset root sets ``batch_sampler: type_balanced``.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from torch.utils.data import DataLoader, Dataset, Sampler


def uses_type_balanced(data_root: str | Path) -> bool:
    manifest = Path(data_root) / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        payload = json.loads(manifest.read_text())
    except json.JSONDecodeError:
        return False
    return payload.get("batch_sampler") == "type_balanced"


def _architecture_of(path: Path, split_root: Path, recorded: str) -> str:
    if recorded:
        return recorded
    try:
        rel = Path(path).resolve().relative_to(Path(split_root).resolve())
    except ValueError:
        rel = Path(path)
    parts = rel.parts
    # train/fake/<Architecture>/...  or  fake/<Architecture>/...
    if len(parts) >= 3 and parts[0] in {"real", "fake"}:
        return parts[1]
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def build_type_groups(dataset: Dataset) -> tuple[list[int], dict[str, list[int]]]:
    """Split dataset indices into reals and fake-by-architecture.

    Works with ``AIGCDataset`` or ``FlaggedAugmentDataset``.
    """
    base = getattr(dataset, "base", dataset)
    samples = base.samples
    arches = getattr(base, "architectures", None)
    if arches is None or len(arches) != len(samples):
        arches = [""] * len(samples)
    split_root = base.root
    reals: list[int] = []
    fakes: dict[str, list[int]] = defaultdict(list)
    for idx, ((path, label), arch) in enumerate(zip(samples, arches)):
        if int(label) == 0:
            reals.append(idx)
        else:
            fakes[_architecture_of(path, split_root, arch)].append(idx)
    return reals, dict(fakes)


class TypeBalancedBatchSampler(Sampler[list[int]]):
    """Yield ``batch_size`` indices: reals first (cycled), then fakes."""

    def __init__(
        self,
        real_indices: list[int],
        fake_by_arch: dict[str, list[int]],
        batch_size: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        drop_last: bool = True,
        real_frac: float = 0.5,
    ) -> None:
        if batch_size < 2:
            raise ValueError(f"batch_size must be >= 2, got {batch_size}")
        if not 0.0 < real_frac < 1.0:
            raise ValueError(f"real_frac must be in (0, 1), got {real_frac}")
        if not real_indices:
            raise ValueError("TypeBalancedBatchSampler needs at least one real image")
        if not fake_by_arch or not any(fake_by_arch.values()):
            raise ValueError("TypeBalancedBatchSampler needs at least one fake image")
        # Round to whole images; keep at least one of each so no batch is single-class.
        self.n_real = min(batch_size - 1, max(1, round(batch_size * real_frac)))
        self.n_fake = batch_size - self.n_real
        self.real_frac = real_frac
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.rank = rank
        self.world_size = world_size
        self.real_indices = list(real_indices)
        self.fake_by_arch = {
            arch: idxs[rank::world_size]
            for arch, idxs in sorted(fake_by_arch.items())
            if idxs[rank::world_size]
        }
        if not self.fake_by_arch:
            raise ValueError(f"rank {rank} received no fake indices")
        # Reals are reused; keep a local shard, fall back to the full list.
        real_shard = self.real_indices[rank::world_size]
        self.real_local = real_shard if real_shard else list(self.real_indices)
        self._len = sum(len(v) for v in self.fake_by_arch.values()) // self.n_fake

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._len

    def _fake_stream(self, rng: random.Random) -> Iterator[int]:
        order = list(self.fake_by_arch)
        pools = {arch: idxs[:] for arch, idxs in self.fake_by_arch.items()}
        for arch in order:
            rng.shuffle(pools[arch])
        cursors = {arch: 0 for arch in order}
        while True:
            progressed = False
            for arch in order:
                i = cursors[arch]
                if i < len(pools[arch]):
                    yield pools[arch][i]
                    cursors[arch] = i + 1
                    progressed = True
            if not progressed:
                return

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch * 1009 + self.rank)
        reals = self.real_local[:]
        rng.shuffle(reals)
        real_i = 0

        def next_real() -> int:
            nonlocal real_i
            if real_i >= len(reals):
                rng.shuffle(reals)
                real_i = 0
            value = reals[real_i]
            real_i += 1
            return value

        batch_fakes: list[int] = []
        for fake_idx in self._fake_stream(rng):
            batch_fakes.append(fake_idx)
            if len(batch_fakes) == self.n_fake:
                yield [next_real() for _ in range(self.n_real)] + batch_fakes
                batch_fakes = []
        if batch_fakes and not self.drop_last:
            n = max(1, round(len(batch_fakes) * self.real_frac / (1.0 - self.real_frac)))
            extra_reals = [next_real() for _ in range(n)]
            yield extra_reals + batch_fakes


def make_train_loader(
    dataset: Dataset,
    data_root: str | Path,
    *,
    batch_size: int,
    workers: int,
    rank: int,
    world_size: int,
    shuffle_fallback,
    pin_memory: bool = True,
    real_frac: float = 0.5,
) -> tuple[DataLoader, object | None]:
    """Build the train loader; use type-balanced batches when the manifest says so.

    Returns ``(loader, sampler_or_batch_sampler)``. Call ``set_epoch`` on the
    second value when it is not ``None``.
    """
    if uses_type_balanced(data_root):
        reals, fakes = build_type_groups(dataset)
        batch_sampler = TypeBalancedBatchSampler(
            reals, fakes, batch_size,
            rank=rank, world_size=world_size, real_frac=real_frac,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=workers,
            pin_memory=pin_memory,
        )
        return loader, batch_sampler

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_fallback is None,
        sampler=shuffle_fallback,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    return loader, shuffle_fallback
