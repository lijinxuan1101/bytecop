"""CLIP-H + RGPA standardized weighted logit fusion (Stage 3A).

The two branches are trained independently. This module loads their
checkpoints, freezes both, standardizes each logit with validation
statistics, then fuses with a fixed weight:

    fused = w * clip_z + (1 - w) * rgpa_z

Temperature scaling is applied to the fused logit after fitting on an
independent calibration split. Feature concat is a separate Stage 3B
experiment and is not implemented here.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from models.clip_tower import CLIPTower
from models.rgpa import RGPA


class DualTower(nn.Module):
    """Fuse OpenCLIP-H spatial logits with RGPA forensic logits.

    Args:
        clip_ckpt: Path to the saved CLIPTower checkpoint (``state_dict``).
        rgpa_ckpt: Path to the saved RGPA checkpoint (``state_dict``).
        clip_kwargs: Extra kwargs forwarded to ``CLIPTower.__init__``.
        rgpa_kwargs: Extra kwargs forwarded to ``RGPA.__init__``.
        fusion_weight: Fixed weight on the standardized CLIP logit.
        clip_logit_mean / clip_logit_std: Validation stats for CLIP logits.
        rgpa_logit_mean / rgpa_logit_std: Validation stats for RGPA logits.
    """

    def __init__(
        self,
        *,
        clip_ckpt: str | Path | None = None,
        rgpa_ckpt: str | Path | None = None,
        clip_kwargs: dict | None = None,
        rgpa_kwargs: dict | None = None,
        fusion_weight: float = 0.5,
        clip_logit_mean: float = 0.0,
        clip_logit_std: float = 1.0,
        rgpa_logit_mean: float = 0.0,
        rgpa_logit_std: float = 1.0,
    ) -> None:
        super().__init__()

        self.clip = CLIPTower(**(clip_kwargs or {}))
        self.rgpa = RGPA(**(rgpa_kwargs or {}))

        if clip_ckpt is not None:
            state = torch.load(clip_ckpt, map_location="cpu", weights_only=True)
            self.clip.load_state_dict(state, strict=False)

        if rgpa_ckpt is not None:
            state = torch.load(rgpa_ckpt, map_location="cpu", weights_only=True)
            self.rgpa.load_state_dict(state, strict=False)

        for param in self.clip.parameters():
            param.requires_grad = False
        for param in self.rgpa.parameters():
            param.requires_grad = False

        self.register_buffer("fusion_weight", torch.tensor(float(fusion_weight)))
        self.register_buffer("clip_mean", torch.tensor(float(clip_logit_mean)))
        self.register_buffer("clip_std", torch.tensor(float(clip_logit_std)))
        self.register_buffer("rgpa_mean", torch.tensor(float(rgpa_logit_mean)))
        self.register_buffer("rgpa_std", torch.tensor(float(rgpa_logit_std)))
        self.temperature = nn.Parameter(torch.ones(1))

    def set_logit_stats(
        self,
        *,
        clip_mean: float,
        clip_std: float,
        rgpa_mean: float,
        rgpa_std: float,
        fusion_weight: float | None = None,
    ) -> None:
        """Store validation logit statistics used for Stage 3A fusion."""
        self.clip_mean.fill_(clip_mean)
        self.clip_std.fill_(max(clip_std, 1e-4))
        self.rgpa_mean.fill_(rgpa_mean)
        self.rgpa_std.fill_(max(rgpa_std, 1e-4))
        if fusion_weight is not None:
            self.fusion_weight.fill_(fusion_weight)

    def _fuse(
        self,
        clip_logit: torch.Tensor,
        rgpa_logit: torch.Tensor,
    ) -> torch.Tensor:
        clip_z = (clip_logit - self.clip_mean) / self.clip_std.clamp(min=1e-4)
        rgpa_z = (rgpa_logit - self.rgpa_mean) / self.rgpa_std.clamp(min=1e-4)
        fused = self.fusion_weight * clip_z + (1.0 - self.fusion_weight) * rgpa_z
        return fused / self.temperature.clamp(min=1e-4)

    def forward(
        self,
        clip_x: torch.Tensor,
        rgpa_x: torch.Tensor,
    ) -> torch.Tensor:
        """Return a calibrated fused logit per image.

        Args:
            clip_x: CLIP-normalized image tensor ``[B, 3, 224, 224]``.
            rgpa_x: Pixel-scale RGB tensor ``[B, 3, 224, 224]``.
        """
        return self._fuse(self.clip(clip_x), self.rgpa(rgpa_x))

    def forward_separate(
        self,
        clip_x: torch.Tensor,
        rgpa_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(clip_logit, rgpa_logit, fused_logit)`` for ablation."""
        clip_logit = self.clip(clip_x)
        rgpa_logit = self.rgpa(rgpa_x)
        return clip_logit, rgpa_logit, self._fuse(clip_logit, rgpa_logit)

    def param_count(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "clip": sum(p.numel() for p in self.clip.parameters()),
            "rgpa": sum(p.numel() for p in self.rgpa.parameters()),
        }
