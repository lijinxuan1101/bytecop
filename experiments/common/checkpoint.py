"""Best-checkpoint monitor, last.pt snapshot, and resume helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


LAST_NAME = "last.pt"
BEST_NAME = "best.pt"


def is_foreign_resume(resume: str | Path, output_dir: Path) -> bool:
    """True when the resume source is not this run directory (start a new log)."""
    src = Path(resume).resolve()
    out = Path(output_dir).resolve()
    if src.is_dir():
        return src != out
    return src.parent != out


def resolve_resume_path(resume: str | Path, *, prefer_best: bool = False) -> Path:
    """Accept a file or a run directory (prefer ``last.pt``, then ``best.pt``)."""
    path = Path(resume)
    if path.is_dir():
        names = (BEST_NAME, LAST_NAME) if prefer_best else (LAST_NAME, BEST_NAME)
        for name in names:
            candidate = path / name
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"No {LAST_NAME} or {BEST_NAME} under {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    return path


def epoch_snapshot_steps(n_steps: int, frac: float | None) -> dict[int, int]:
    """1-based step → percent (e.g. 5, 10, … for frac=0.05), excluding 100%."""
    if not frac or frac <= 0 or n_steps < 2:
        return {}
    n_marks = max(1, int(round(1.0 / frac)) - 1)
    out: dict[int, int] = {}
    for k in range(1, n_marks + 1):
        step = max(1, int(round(n_steps * k * frac)))
        if step >= n_steps:
            continue
        out[step] = int(round(100 * k * frac))
    return out


def load_checkpoint(path: Path, *, map_location) -> dict[str, Any]:
    """Load ``last.pt`` (full) or ``best.pt`` (weights-only)."""
    obj = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(obj, dict) and "model" in obj:
        return obj
    return {"model": obj}


def move_optimizer_state(optimizer: torch.optim.Optimizer, device) -> None:
    """Remap Adam moments onto ``device``.

    ``last.pt`` is loaded on rank 0 and then broadcast. Unpickled CUDA tensors
    stay on that rank's ``cuda:0``, so other ranks would step with params on
    ``cuda:N`` and moments on ``cuda:0``.
    """
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value) and key != "step":
                state[key] = value.to(device, non_blocking=True)


def save_last(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    history: list[dict],
    monitor: MetricMonitor | None,
    step_in_epoch: int | None = None,
    epoch_pct: int | None = None,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "history": history,
        "monitor": None if monitor is None else monitor.state_dict(),
        "step_in_epoch": step_in_epoch,
        "epoch_pct": epoch_pct,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def last_positive_lr(history: list[dict]) -> float | None:
    """Most recent non-zero LR recorded in ``history.json``."""
    for record in reversed(history):
        for key in ("lr", "lr_head"):
            value = record.get(key)
            if value is not None and float(value) > 0:
                return float(value)
    return None


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

    def update(
        self, epoch: int, metrics: dict, model: nn.Module, *, count_stale: bool = True,
    ) -> bool:
        if self.metric not in metrics:
            raise KeyError(f"monitor metric {self.metric!r} missing from {sorted(metrics)}")
        value = float(metrics[self.metric])
        if not self._improved(value):
            if count_stale:
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

    def state_dict(self) -> dict:
        return {
            "best": self.best,
            "best_epoch": self.best_epoch,
            "stale": self.stale,
            "best_record": self.best_record,
        }

    def load_state_dict(self, state: dict) -> None:
        self.best = float(state["best"])
        self.best_epoch = int(state["best_epoch"])
        self.stale = int(state.get("stale", 0))
        self.best_record = dict(state.get("best_record") or {})

    def restore_from_run(self, output_dir: Path, history: list[dict]) -> None:
        """Rebuild monitor from ``best_metrics.json`` + history after a weights-only resume."""
        metrics_path = output_dir / "best_metrics.json"
        if metrics_path.is_file():
            with open(metrics_path) as f:
                record = json.load(f)
            self.best = float(record[self.metric])
            self.best_epoch = int(record.get("epoch", 0))
            self.best_record = record
        if history:
            last_improved = 0
            for record in history:
                if record.get("epoch") == self.best_epoch:
                    last_improved = 0
                else:
                    last_improved += 1
            self.stale = last_improved
