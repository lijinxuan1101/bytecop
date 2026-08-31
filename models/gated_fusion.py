"""Per-sample gated fusion of two frozen-tower logits.

Towers stay frozen. This module only sees the two standardized logits:

    z = [z_spatial, z_forensic]     # [B, 2]
    w = softmax(MLP(z))             # [B, 2]
    fused = w_s * z_s + w_f * z_f   # [B]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    def __init__(self, hidden_dim: int = 8) -> None:
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        spatial_logit: torch.Tensor,
        forensic_logit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spatial = spatial_logit.reshape(-1, 1)
        forensic = forensic_logit.reshape(-1, 1)
        gate_scores = self.gate_net(torch.cat([spatial, forensic], dim=-1))
        gate_weights = torch.softmax(gate_scores, dim=-1)
        w_spatial = gate_weights[:, 0:1]
        w_forensic = gate_weights[:, 1:2]
        fused_logit = (w_spatial * spatial + w_forensic * forensic).squeeze(-1)
        return fused_logit, gate_weights
