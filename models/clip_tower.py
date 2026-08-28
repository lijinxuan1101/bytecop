"""CLIP ViT-H/14 tower for AIGC detection.

Architecture:
    CLIP ViT-H/14 backbone (open_clip)
        → freeze all layers except the last ``unfreeze_blocks`` transformer blocks
        → global CLS token embedding  [B, 1280]
        → projection layer  [B, proj_dim]
        → binary classification head  [B, 1]  (logit, not probability)

The model outputs a raw logit.  Apply ``torch.sigmoid`` at inference time to
get a probability, or use ``BCEWithLogitsLoss`` during training.

Model name in open_clip: ``ViT-H-14``, pretrained weights: ``laion2b_s32b_b79k``
    - Vision encoder parameters: ~986 M
    - Input resolution: 224 × 224
    - CLS embedding dim: 1280
"""

from __future__ import annotations

import torch
import torch.nn as nn
import open_clip


_CLIP_MODEL_NAME = "ViT-H-14"
_CLIP_PRETRAINED = "laion2b_s32b_b79k"
_EMBED_DIM = 1280


class CLIPTower(nn.Module):
    """CLIP ViT-H/14 fine-tuned for binary AIGC detection.

    Args:
        unfreeze_blocks: Number of trailing transformer blocks to unfreeze for
            fine-tuning (default 4).  Set to 0 to freeze the entire backbone
            (linear probe only).
        proj_dim: Intermediate projection dimension between backbone and head.
            Set to 0 to skip the projection layer.
        dropout: Dropout probability applied before the classification head.
        pretrained: open_clip pretrained weight tag.
    """

    def __init__(
        self,
        *,
        unfreeze_blocks: int = 4,
        proj_dim: int = 512,
        dropout: float = 0.1,
        pretrained: str = _CLIP_PRETRAINED,
    ) -> None:
        super().__init__()

        clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            _CLIP_MODEL_NAME, pretrained=pretrained
        )
        self.backbone: nn.Module = clip_model.visual

        self._freeze_backbone(unfreeze_blocks)

        # Classification head
        in_dim = _EMBED_DIM
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
        """Freeze the backbone, then selectively unfreeze the last N blocks."""
        for param in self.backbone.parameters():
            param.requires_grad = False

        if unfreeze_blocks <= 0:
            return

        # open_clip ViT stores transformer blocks in backbone.transformer.resblocks
        try:
            resblocks = list(self.backbone.transformer.resblocks)
        except AttributeError:
            # Fallback: unfreeze the whole backbone
            for param in self.backbone.parameters():
                param.requires_grad = True
            return

        for block in resblocks[-unfreeze_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

        # Always unfreeze layer-norm and projection at the top of the vision encoder
        for attr in ("ln_post", "proj"):
            module = getattr(self.backbone, attr, None)
            if module is not None:
                for param in (
                    module.parameters() if isinstance(module, nn.Module) else [module]
                ):
                    param.requires_grad = True

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a scalar logit per image.

        Args:
            x: Image tensor of shape ``[B, 3, 224, 224]`` pre-processed with
               ``self.preprocess``.

        Returns:
            Logit tensor of shape ``[B]``.
        """
        features = self.backbone(x)  # [B, 1280]
        features = self.proj(features)
        logit = self.head(features).squeeze(1)  # [B]
        return logit

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return only the parameters that require gradients."""
        return [p for p in self.parameters() if p.requires_grad]

    def param_count(self) -> dict[str, int]:
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
