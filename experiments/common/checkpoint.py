"""Best-checkpoint monitor with optional early stopping."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn


class MetricMonitor:
    """Persist ``best.pt`` / ``best_metrics.json`` when the monitor metric improves."""

    def __init__(
        self,
        output_dir: Path,
        *,
        metric: str = "val_auc",
        mode: str = "max",
        patience: int | None = None,
    ) -> None:
        if mode not in {"max", "min"}:
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.output_dir = Path(output_dir)
        self.metric = metric
        self.mode = mode
        self.patience = patience if patience and patience > 0 else None
        self.best = float("-inf") if mode == "max" else float("inf")
        self.best_epoch = 0
        self.stale = 0
        self.best_record: dict = {}

    def _improved(self, value: float) -> bool:
        return value > self.best if self.mode == "max" else value < self.best

    def update(self, epoch: int, metrics: dict, model: nn.Module) -> bool:
        if self.metric not in metrics:
            raise KeyError(f"monitor metric {self.metric!r} missing from {sorted(metrics)}")
        value = float(metrics[self.metric])
        if not self._improved(value):
            self.stale += 1
            return False

        self.best = value
        self.best_epoch = epoch
        self.stale = 0
        self.best_record = {"epoch": epoch, "monitor": self.metric, **metrics}
        torch.save(model.state_dict(), self.output_dir / "best.pt")
        with open(self.output_dir / "best_metrics.json", "w") as f:
            json.dump(self.best_record, f, indent=2)
        return True

    def should_stop(self) -> bool:
        return self.patience is not None and self.stale >= self.patience
