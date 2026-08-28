"""Dual-tower logit fusion model for AIGC detection.

Fusion strategy (priority order from the technical spec):
    ① Logit average  – default; re-calibrated after fusion
    ② Feature concat – only if logit average still has headroom and budget allows

This module only implements ① (logit average).  Feature concat is a separate
experiment that can be added later once ablation results justify the complexity.

The two towers are expected to have been trained and saved independently.
DualTower loads their checkpoints and freezes both backbones; only the
temperature parameter (for post-fusion calibration) is learnable by default.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from models.clip_tower import CLIPTower
from models.dino_tower import DINOTower


class DualTower(nn.Module):
    """Fuse CLIP-H and DINOv3-H+ logits via simple averaging.

    The fused logit is: ``logit = (clip_logit + dino_logit) / 2``

    Temperature scaling is applied to the fused logit at inference time after
    fitting on an independent calibration split.

    Args:
        clip_ckpt: Path to the saved CLIPTower checkpoint (``state_dict``).
        dino_ckpt: Path to the saved DINOTower checkpoint (``state_dict``).
        clip_kwargs: Additional kwargs forwarded to ``CLIPTower.__init__``.
        dino_kwargs: Additional kwargs forwarded to ``DINOTower.__init__``.
    """

    def __init__(
        self,
        *,
        clip_ckpt: str | Path | None = None,
        dino_ckpt: str | Path | None = None,
        clip_kwargs: dict | None = None,
        dino_kwargs: dict | None = None,
    ) -> None:
        super().__init__()

        self.clip = CLIPTower(**(clip_kwargs or {}))
        self.dino = DINOTower(**(dino_kwargs or {}))

        if clip_ckpt is not None:
            state = torch.load(clip_ckpt, map_location="cpu", weights_only=True)
            self.clip.load_state_dict(state, strict=False)

        if dino_ckpt is not None:
            state = torch.load(dino_ckpt, map_location="cpu", weights_only=True)
            self.dino.load_state_dict(state, strict=False)

        # Freeze both backbones after loading; only the temperature is updated
        # during calibration.
        for param in self.clip.parameters():
            param.requires_grad = False
        for param in self.dino.parameters():
            param.requires_grad = False

        # Temperature parameter (scalar), initialized to 1.0 (= no scaling)
        self.temperature = nn.Parameter(torch.ones(1))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        clip_x: torch.Tensor,
        dino_x: torch.Tensor,
    ) -> torch.Tensor:
        """Return a calibrated logit per image.

        Args:
            clip_x: CLIP-preprocessed image tensor ``[B, 3, 224, 224]``.
            dino_x: DINO-preprocessed image tensor ``[B, 3, H, W]``.

        Returns:
            Fused and temperature-scaled logit of shape ``[B]``.
        """
        clip_logit = self.clip(clip_x)
        dino_logit = self.dino(dino_x)
        fused = (clip_logit + dino_logit) / 2.0
        return fused / self.temperature.clamp(min=1e-4)

    def forward_separate(
        self,
        clip_x: torch.Tensor,
        dino_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return individual logits alongside the fused logit (for ablation).

        Returns:
            (clip_logit, dino_logit, fused_logit) — all of shape ``[B]``.
        """
        clip_logit = self.clip(clip_x)
        dino_logit = self.dino(dino_x)
        fused = (clip_logit + dino_logit) / 2.0 / self.temperature.clamp(min=1e-4)
        return clip_logit, dino_logit, fused

    def param_count(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "clip": sum(p.numel() for p in self.clip.parameters()),
            "dino": sum(p.numel() for p in self.dino.parameters()),
        }
