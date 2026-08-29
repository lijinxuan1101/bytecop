"""Warmup + cosine learning-rate schedule (per optimizer step)."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


def build_warmup_cosine(
    optimizer: Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float,
) -> tuple[LRScheduler, int]:
    """Linear warmup for ``warmup_ratio`` of steps, then cosine to 0."""
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError(f"warmup_ratio must be in [0, 1), got {warmup_ratio}")

    warmup_steps = max(1, int(total_steps * warmup_ratio)) if warmup_ratio > 0 else 0

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, lr_lambda), warmup_steps
