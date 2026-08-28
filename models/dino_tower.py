"""DINOv3 ViT-H+ tower for AIGC detection.

Architecture:
    DINOv3 ViT-H+ backbone (via timm or torch.hub)
        → freeze all layers except the last ``unfreeze_blocks`` transformer blocks
        → global CLS token embedding  [B, 1280]
        → projection layer  [B, proj_dim]
        → binary classification head  [B, 1]  (logit, not probability)

DINOv3 (Meta AI, 2025) uses a ViT-H+ with ~840 M vision parameters and an
embedding dimension of 1280 (same as DINOv2-giant: 1536, but ViT-H uses 1280).

Model resolution: 224 × 224 (518 × 518 also available for higher accuracy at
the cost of compute).  We default to 224 for speed during training.

Loading priority:
    1. timm registry:   ``vit_huge_patch14_dinov3`` (when available)
    2. torch.hub:       ``facebookresearch/dinov2``, ``dinov2_vitg14``  (fallback)
    3. HuggingFace:     ``facebook/dinov2-giant``  (final fallback)
"""

from __future__ import annotations

import torch
import torch.nn as nn


_EMBED_DIM = 1280  # ViT-H hidden dim


def _load_dinov3() -> tuple[nn.Module, int]:
    """Try to load DINOv3 ViT-H+; fall back to DINOv2 ViT-G (similar scale).

    Returns:
        (backbone, embed_dim) where backbone outputs CLS features.
    """
    # --- Attempt 1: timm DINOv3 ViT-H ---
    try:
        import timm
        model = timm.create_model(
            "vit_huge_patch14_reg4_dinov3",
            pretrained=True,
            num_classes=0,
        )
        embed_dim = model.num_features
        return model, embed_dim
    except Exception:
        pass

    try:
        import timm
        model = timm.create_model(
            "vit_huge_patch14_dinov3",
            pretrained=True,
            num_classes=0,
        )
        embed_dim = model.num_features
        return model, embed_dim
    except Exception:
        pass

    # --- Attempt 2: torch.hub DINOv2 ViT-G (1536-d, ~1.1B, similar scale) ---
    try:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14", verbose=False)
        embed_dim = model.embed_dim  # 1536
        return model, embed_dim
    except Exception:
        pass

    # --- Attempt 3: HuggingFace DINOv2-Giant ---
    from transformers import AutoModel
    model = AutoModel.from_pretrained("facebook/dinov2-giant")
    embed_dim = model.config.hidden_size  # 1536
    return _HFDINOWrapper(model), embed_dim


class _HFDINOWrapper(nn.Module):
    """Thin wrapper around a HuggingFace DINOv2 model to expose CLS features."""

    def __init__(self, hf_model: nn.Module) -> None:
        super().__init__()
        self.model = hf_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=x)
        return outputs.last_hidden_state[:, 0]  # CLS token


class DINOTower(nn.Module):
    """DINOv3 ViT-H+ fine-tuned for binary AIGC detection.

    Args:
        unfreeze_blocks: Number of trailing transformer blocks to unfreeze
            (default 4).
        proj_dim: Intermediate projection dimension (0 = skip projection).
        dropout: Dropout probability before classification head.
    """

    def __init__(
        self,
        *,
        unfreeze_blocks: int = 4,
        proj_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.backbone, embed_dim = _load_dinov3()
        self._freeze_backbone(unfreeze_blocks)

        in_dim = embed_dim
        if proj_dim > 0:
            self.proj = nn.Sequential(
                nn.Linear(in_dim, proj_dim),
                nn.GELU(),
            )
            in_dim = proj_dim
        else:
            self.proj = nn.Identity()

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, 1),
        )

    # ------------------------------------------------------------------
    # Freezing helpers
    # ------------------------------------------------------------------

    def _freeze_backbone(self, unfreeze_blocks: int) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

        if unfreeze_blocks <= 0:
            return

        blocks = self._get_blocks()
        for block in blocks[-unfreeze_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

        # Unfreeze final norm layers
        for attr in ("norm", "ln_f", "layernorm"):
            module = getattr(self.backbone, attr, None)
            if module is not None and isinstance(module, nn.Module):
                for param in module.parameters():
                    param.requires_grad = True

    def _get_blocks(self) -> list[nn.Module]:
        """Return the list of transformer blocks regardless of backbone API."""
        for attr in ("blocks", "layers", "encoder.layer"):
            parts = attr.split(".")
            module = self.backbone
            try:
                for part in parts:
                    module = getattr(module, part)
                return list(module)
            except AttributeError:
                continue
        return []

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a scalar logit per image.

        Args:
            x: Pre-processed image tensor ``[B, 3, H, W]``.

        Returns:
            Logit tensor of shape ``[B]``.
        """
        # timm models return CLS directly when num_classes=0; hub models expose
        # forward_features; HF wrapper also returns CLS.
        if hasattr(self.backbone, "forward_features"):
            features = self.backbone.forward_features(x)
            # timm may return a dict or tensor
            if isinstance(features, dict):
                features = features["x_norm_clstoken"]
            elif features.ndim == 3:
                features = features[:, 0]
        else:
            features = self.backbone(x)

        features = self.proj(features)
        logit = self.head(features).squeeze(1)
        return logit

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def param_count(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
