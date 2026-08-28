"""Real-world image transformations used for robustness evaluation.

The public entry point is :func:`apply_real_world_transform`.  Every
transformation returns an RGB image with the same size as the input so that
the result can be passed directly to a classifier.
"""

from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Literal

from PIL import Image, ImageEnhance, ImageFilter


TransformName = Literal[
    "jpeg",
    "gaussian_blur",
    "resize",
    "gaussian_noise",
    "color_jitter",
    "center_crop",
]


def apply_real_world_transform(
    image: Image.Image | str | Path,
    transform: TransformName,
    *,
    value: float | int | None = None,
    seed: int | None = None,
) -> Image.Image:
    """Apply one realistic post-processing operation to an image.

    Args:
        image: A Pillow image or a path to an image file.
        transform: One of ``jpeg``, ``gaussian_blur``, ``resize``,
            ``gaussian_noise``, ``color_jitter``, or ``center_crop``.
        value: Transformation strength. Defaults and expected values are:

            - ``jpeg``: JPEG quality, default 70 (suggested: 90/70/50/30).
            - ``gaussian_blur``: Gaussian radius/sigma, default 1.0.
            - ``resize``: downscale factor before upscaling, default 0.5.
            - ``gaussian_noise``: standard deviation on [0, 1], default 0.05.
            - ``color_jitter``: maximum proportional jitter, default 0.2.
            - ``center_crop``: retained width/height fraction, default 0.8.

        seed: Seed for reproducible noise and color jitter.

    Returns:
        A new RGB Pillow image with the same dimensions as the input.

    Raises:
        ValueError: If the transform name or strength is invalid.
    """
    source = _load_rgb(image)
    width, height = source.size
    rng = random.Random(seed)

    if transform == "jpeg":
        quality = int(70 if value is None else value)
        if not 1 <= quality <= 100:
            raise ValueError("JPEG quality must be between 1 and 100")
        buffer = io.BytesIO()
        source.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()

    if transform == "gaussian_blur":
        sigma = float(1.0 if value is None else value)
        if sigma < 0:
            raise ValueError("Gaussian blur sigma must be non-negative")
        return source.filter(ImageFilter.GaussianBlur(radius=sigma))

    if transform == "resize":
        scale = float(0.5 if value is None else value)
        if not 0 < scale <= 1:
            raise ValueError("Resize scale must be in the interval (0, 1]")
        reduced_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        reduced = source.resize(reduced_size, Image.Resampling.BICUBIC)
        return reduced.resize((width, height), Image.Resampling.BICUBIC)

    if transform == "gaussian_noise":
        sigma = float(0.05 if value is None else value)
        if sigma < 0:
            raise ValueError("Gaussian noise sigma must be non-negative")
        pixel_sigma = sigma * 255.0
        noisy_bytes = bytes(
            _clip(channel + rng.gauss(0.0, pixel_sigma))
            for channel in source.tobytes()
        )
        return Image.frombytes("RGB", source.size, noisy_bytes)

    if transform == "color_jitter":
        strength = float(0.2 if value is None else value)
        if not 0 <= strength <= 1:
            raise ValueError("Color jitter strength must be in the interval [0, 1]")
        result = source
        # Randomize both the factors and their order, as common training
        # augmentations do. A seed makes evaluation runs reproducible.
        operations = [
            (ImageEnhance.Brightness, rng.uniform(1 - strength, 1 + strength)),
            (ImageEnhance.Contrast, rng.uniform(1 - strength, 1 + strength)),
            (ImageEnhance.Color, rng.uniform(1 - strength, 1 + strength)),
        ]
        rng.shuffle(operations)
        for enhancer, factor in operations:
            result = enhancer(result).enhance(factor)
        return result

    if transform == "center_crop":
        fraction = float(0.8 if value is None else value)
        if not 0 < fraction <= 1:
            raise ValueError("Center-crop fraction must be in the interval (0, 1]")
        crop_width = max(1, round(width * fraction))
        crop_height = max(1, round(height * fraction))
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        cropped = source.crop((left, top, left + crop_width, top + crop_height))
        return cropped.resize((width, height), Image.Resampling.BICUBIC)

    raise ValueError(f"Unsupported transform: {transform!r}")


def _load_rgb(image: Image.Image | str | Path) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB").copy()
    with Image.open(image) as opened:
        return opened.convert("RGB").copy()


def _clip(value: float) -> int:
    return max(0, min(255, round(value)))
