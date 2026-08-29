"""Residual-Guided Patch Aggregation (RGPA) forensic branch.

Stage 2 of the technical spec:

    RGB (pixel-scale, no CLIP/ImageNet normalize)
        → frozen whole-image SRM-inspired high-pass residual
        → 32×32 patches, shared lightweight CNN
        → high/low residual-energy aggregation
        → forensic logit

High-pass filtering is applied to the whole image before unfolding so that
per-patch convolution padding cannot invent artificial patch borders.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


IMG_SIZE = 224
PATCH_SIZE = 32
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 7 × 7 = 49
EMBED_DIM = 128

# Classic 5×5 SRM high-pass kernels (AIDE / Fridrich rich models subset).
_SRM_KERNELS = torch.tensor(
    [
        [
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ],
        [
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ],
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
    ],
    dtype=torch.float32,
)
_SRM_SCALES = torch.tensor([4.0, 12.0, 2.0])


def _srm_weight(in_ch: int = 3) -> torch.Tensor:
    """Build a frozen conv weight: 3 kernels × ``in_ch`` input channels."""
    n_k, k, _ = _SRM_KERNELS.shape
    weight = torch.zeros(n_k * in_ch, in_ch, k, k)
    kernels = _SRM_KERNELS / _SRM_SCALES[:, None, None]
    for ki in range(n_k):
        for c in range(in_ch):
            weight[ki * in_ch + c, c] = kernels[ki]
    return weight


class SRMResidual(nn.Module):
    """Fixed whole-image SRM high-pass residual. Output has 9 channels."""

    def __init__(self, *, in_ch: int = 3, clip: float = 3.0) -> None:
        super().__init__()
        weight = _srm_weight(in_ch)
        self.register_buffer("weight", weight)
        self.clip = clip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.conv2d(x, self.weight, padding=2)
        return residual.clamp(-self.clip, self.clip)


class ResidualEncoder(nn.Module):
    """Lightweight CNN for 32×32 residual patches."""

    def __init__(self, *, in_ch: int = 9, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _head(in_dim: int, *, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_dim, 1),
    )


class RGPA(nn.Module):
    """Residual-Guided Patch Aggregation forensic branch.

    After the whole-image SRM residual, the map is split into 49 patches of
    32×32. Each patch is encoded by the shared CNN. Intra-image z-scored
    residual energy drives a bidirectional softmax (high / low residual).
    The two aggregated vectors are concatenated and classified.
    """

    def __init__(
        self,
        *,
        embed_dim: int = EMBED_DIM,
        dropout: float = 0.1,
        tau: float = 1.0,
        energy_eps: float = 1e-6,
        img_size: int = IMG_SIZE,
        patch_size: int = PATCH_SIZE,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size {img_size} must be divisible by patch_size {patch_size}")
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.residual = SRMResidual()
        self.encoder = ResidualEncoder(embed_dim=embed_dim)
        self.head = _head(embed_dim * 2, dropout=dropout)
        self.residual.eval()
        self.register_buffer("tau", torch.tensor(float(tau)))
        self.energy_eps = energy_eps

    def train(self, mode: bool = True):
        super().train(mode)
        self.residual.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a scalar logit per image. ``x`` is pixel-scale RGB in ``[0, 1]``."""
        z_forensic, _, _ = self.forward_features(x)
        return self.head(z_forensic).squeeze(1)

    def forward_features(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return forensic feature and high/low aggregation weights.

        Returns:
            ``(z_forensic, w_high, w_low)`` with shapes ``[B, 2D]``, ``[B, N]``,
            ``[B, N]`` where ``N = (img_size / patch_size) ** 2``.
        """
        if x.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(
                f"RGPA expects {self.img_size}×{self.img_size} input, "
                f"got {tuple(x.shape[-2:])}"
            )

        residual = self.residual(x)
        patches = _unfold_patches(residual, self.patch_size)
        batch, n_patch, channels, _, _ = patches.shape

        # a_i = mean squared residual over (C, H, W); z-score inside the image.
        energy = patches.pow(2).mean(dim=(2, 3, 4))  # [B, 49]
        centered = energy - energy.mean(dim=1, keepdim=True)
        scale = energy.std(dim=1, keepdim=True, unbiased=False).clamp(min=self.energy_eps)
        energy_hat = centered / scale

        tau = self.tau.clamp(min=1e-4)
        w_high = torch.softmax(energy_hat / tau, dim=1)
        w_low = torch.softmax(-energy_hat / tau, dim=1)

        encoded = self.encoder(
            patches.reshape(batch * n_patch, channels, self.patch_size, self.patch_size)
        )
        encoded = encoded.reshape(batch, n_patch, -1)
        z_high = (w_high.unsqueeze(-1) * encoded).sum(dim=1)
        z_low = (w_low.unsqueeze(-1) * encoded).sum(dim=1)
        z_forensic = torch.cat([z_high, z_low], dim=1)
        return z_forensic, w_high, w_low

    @staticmethod
    def aggregation_stats(
        w_high: torch.Tensor,
        w_low: torch.Tensor,
    ) -> dict[str, float]:
        """Summarize high/low weight divergence for result interpretation.

        When residual energy is nearly uniform, high and low weights collapse
        toward each other; ``mean_l1`` then approaches 0.
        """
        l1 = (w_high - w_low).abs().sum(dim=1)
        entropy_high = -(w_high * w_high.clamp(min=1e-8).log()).sum(dim=1)
        entropy_low = -(w_low * w_low.clamp(min=1e-8).log()).sum(dim=1)
        return {
            "mean_l1_high_low": float(l1.mean()),
            "mean_entropy_high": float(entropy_high.mean()),
            "mean_entropy_low": float(entropy_low.mean()),
        }

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def param_count(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


def _unfold_patches(residual: torch.Tensor, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """Split ``[B, C, H, W]`` into ``[B, N, C, P, P]`` non-overlapping patches."""
    patches = residual.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # [B, C, n, n, P, P] → [B, n*n, C, P, P]
    return patches.permute(0, 2, 3, 1, 4, 5).contiguous().flatten(1, 2)
