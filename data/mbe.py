"""Multi-Scale Bounded Enhancement (MBE).

Fixed mathematical preprocess in front of the Spatial CLIP tower. No trainable
parameters, and no attempt to detect whether the input is degraded.

    RGB → luma → two-scale Gaussian residuals → MAD normalize → bounded boost
        → restore RGB → OpenCLIP resize/crop/normalize → Spatial tower

    from data.mbe import enhance, MultiScaleBoundedEnhancement

    image = enhance(pil_image)                  # PIL in, PIL out
    tensor_ready = MultiScaleBoundedEnhancement()(pil_image)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# Rec.601 luma. Keep chroma by scaling RGB with Y'/Y.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float64)

SIGMA_FINE = 0.8
SIGMA_MID = 1.6
ALPHA_FINE = 2.0 / 255.0
ALPHA_MID = 4.0 / 255.0
TANH_SCALE = 3.0
_EPS = 1e-6
_MAD_SCALE = 1.4826


def _as_float_rgb(image: Image.Image | np.ndarray | str | Path) -> np.ndarray:
    """Return float64 RGB in [0, 1], shape ``[H, W, 3]``."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            array = np.asarray(opened.convert("RGB"), dtype=np.float64)
    elif isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"), dtype=np.float64)
    else:
        array = np.asarray(image, dtype=np.float64)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"expected HxWx3 RGB, got shape {array.shape}")
    if array.max() > 1.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ _LUMA


def _mad_normalize(residual: np.ndarray, *, eps: float) -> np.ndarray:
    """Center by the median, scale by MAD. Robust to edges, grain, JPEG blocking."""
    center = np.median(residual)
    spread = _MAD_SCALE * np.median(np.abs(residual - center)) + eps
    return (residual - center) / spread


def enhance(
    image: Image.Image | np.ndarray | str | Path,
    *,
    sigma_fine: float = SIGMA_FINE,
    sigma_mid: float = SIGMA_MID,
    alpha_fine: float = ALPHA_FINE,
    alpha_mid: float = ALPHA_MID,
    tanh_scale: float = TANH_SCALE,
    eps: float = _EPS,
) -> Image.Image:
    """Apply multi-scale bounded enhancement and return an RGB PIL image.

    Does not try to guess the degradation. The boost is bounded by ``tanh``, so
    grain, ringing and JPEG blocks are not blown up.

    Args:
        image: RGB PIL image, HxWx3 array, or a path.
        sigma_fine: Gaussian σ for the fine residual D1 (default 0.8).
        sigma_mid: Gaussian σ for the mid residual D2 (default 1.6). Must be
            larger than ``sigma_fine``.
        alpha_fine: Weight on the fine residual, in [0, 1] luma units.
        alpha_mid: Weight on the mid residual.
        tanh_scale: ``c`` in ``tanh(D̂ / c)``. Larger = softer clipping.
        eps: Floor on the MAD scale.

    Returns:
        RGB ``PIL.Image`` the same size as the input.
    """
    if sigma_mid <= sigma_fine:
        raise ValueError(
            f"sigma_mid must be > sigma_fine, got {sigma_mid} and {sigma_fine}"
        )
    rgb = _as_float_rgb(image)
    y = _luma(rgb)

    blur_fine = gaussian_filter(y, sigma=sigma_fine, mode="reflect")
    blur_mid = gaussian_filter(y, sigma=sigma_mid, mode="reflect")
    fine = _mad_normalize(y - blur_fine, eps=eps)
    mid = _mad_normalize(blur_fine - blur_mid, eps=eps)

    boost = alpha_fine * np.tanh(fine / tanh_scale) + alpha_mid * np.tanh(mid / tanh_scale)
    y_hat = np.clip(y + boost, 0.0, 1.0)

    scale = y_hat / np.maximum(y, eps)
    enhanced = np.clip(rgb * scale[..., None], 0.0, 1.0)
    out = np.rint(enhanced * 255.0).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


class MultiScaleBoundedEnhancement:
    """Callable for ``torchvision``-style pipelines. No parameters are learned."""

    def __init__(
        self,
        *,
        sigma_fine: float = SIGMA_FINE,
        sigma_mid: float = SIGMA_MID,
        alpha_fine: float = ALPHA_FINE,
        alpha_mid: float = ALPHA_MID,
        tanh_scale: float = TANH_SCALE,
        eps: float = _EPS,
    ) -> None:
        self.sigma_fine = sigma_fine
        self.sigma_mid = sigma_mid
        self.alpha_fine = alpha_fine
        self.alpha_mid = alpha_mid
        self.tanh_scale = tanh_scale
        self.eps = eps

    def __call__(self, image: Image.Image | np.ndarray | str | Path) -> Image.Image:
        return enhance(
            image,
            sigma_fine=self.sigma_fine,
            sigma_mid=self.sigma_mid,
            alpha_fine=self.alpha_fine,
            alpha_mid=self.alpha_mid,
            tanh_scale=self.tanh_scale,
            eps=self.eps,
        )
