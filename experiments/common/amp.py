"""BF16 autocast helpers. No GradScaler — BF16 does not need loss scaling."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch


def bf16_enabled(cfg: dict, device: torch.device) -> bool:
    if not cfg.get("bf16", False):
        return False
    if device.type != "cuda":
        return False
    return bool(torch.cuda.is_bf16_supported())


def autocast_ctx(enabled: bool) -> Any:
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
