"""Real-world image transformations used for robustness evaluation.

The public entry point is :func:`apply_real_world_transform`.  Every
transformation returns an RGB image with the same size as the input so that
the result can be passed directly to a classifier.

Allowed parameter values per transform (strictly following the official table):
    jpeg_compression : quality in {90, 70, 50, 30}
    gaussian_blur    : sigma  in {0.5, 1.0, 2.0}
    resize           : scale  in {0.5, 0.25}
    gaussian_noise   : sigma  in {0.02, 0.05, 0.10}
    color_jitter     : strength = 0.2  (±20%)
    center_crop      : fraction = 0.8  (80%)
"""

from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Literal

from PIL import Image, ImageEnhance, ImageFilter


TransformName = Literal[
    "jpeg_compression",
    "gaussian_blur",
    "resize",
    "gaussian_noise",
    "color_jitter",
    "center_crop",
]

_ALLOWED: dict[str, tuple[float, ...]] = {
    "jpeg_compression": (90, 70, 50, 30),
    "gaussian_blur":    (0.5, 1.0, 2.0),
    "resize":           (0.5, 0.25),
    "gaussian_noise":   (0.02, 0.05, 0.10),
    "color_jitter":     (0.2,),
    "center_crop":      (0.8,),
}


def apply_real_world_transform(
    image: Image.Image | str | Path,
    transform: TransformName,
    *,
    value: float | int,
    seed: int | None = None,
) -> Image.Image:
    """Apply one transform with a fixed, explicitly specified parameter value.

    Args:
        image: A Pillow image or a path to an image file.
        transform: One of ``jpeg_compression``, ``gaussian_blur``, ``resize``,
            ``gaussian_noise``, ``color_jitter``, or ``center_crop``.
        value: Must be one of the allowed values from the official table:

            - ``jpeg_compression``: quality in {90, 70, 50, 30}.
            - ``gaussian_blur``:    sigma  in {0.5, 1.0, 2.0}.
            - ``resize``:           scale  in {0.5, 0.25}.
            - ``gaussian_noise``:   sigma  in {0.02, 0.05, 0.10}.
            - ``color_jitter``:     strength = 0.2 (±20%).
            - ``center_crop``:      fraction = 0.8 (80%).

        seed: Seed for reproducible noise and color jitter.

    Returns:
        A new RGB Pillow image with the same dimensions as the input.

    Raises:
        ValueError: If the transform name or value is not in the allowed set.
    """
    allowed = _ALLOWED.get(transform)
    if allowed is None:
        raise ValueError(f"Unsupported transform: {transform!r}")
    if float(value) not in allowed:
        raise ValueError(
            f"{transform!r} value must be one of {allowed}, got {value!r}"
        )
    return _apply(image, transform, float(value), seed)


def apply_random_real_world_transform(
    image: Image.Image | str | Path,
    transform: TransformName,
    *,
    seed: int | None = None,
) -> tuple[Image.Image, float]:
    """Apply one transform with a parameter value sampled uniformly from the allowed set.

    Args:
        image: A Pillow image or a path to an image file.
        transform: One of ``jpeg_compression``, ``gaussian_blur``, ``resize``,
            ``gaussian_noise``, ``color_jitter``, or ``center_crop``.
        seed: Seed for reproducible value sampling and transform execution.

    Returns:
        A tuple of (transformed image, sampled value) so the caller knows
        which parameter was applied.

    Raises:
        ValueError: If the transform name is not supported.
    """
    allowed = _ALLOWED.get(transform)
    if allowed is None:
        raise ValueError(f"Unsupported transform: {transform!r}")
    rng = random.Random(seed)
    value = rng.choice(allowed)
    child_seed = rng.randint(0, 2**32 - 1) if seed is not None else None
    return _apply(image, transform, value, child_seed), value


def _apply(
    image: Image.Image | str | Path,
    transform: TransformName,
    value: float,
    seed: int | None,
) -> Image.Image:
    source = _load_rgb(image)
    width, height = source.size
    rng = random.Random(seed)

    if transform == "jpeg_compression":
        buffer = io.BytesIO()
        source.save(buffer, format="JPEG", quality=int(value))
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()

    if transform == "gaussian_blur":
        return source.filter(ImageFilter.GaussianBlur(radius=value))

    if transform == "resize":
        reduced_size = (max(1, round(width * value)), max(1, round(height * value)))
        reduced = source.resize(reduced_size, Image.Resampling.BICUBIC)
        return reduced.resize((width, height), Image.Resampling.BICUBIC)

    if transform == "gaussian_noise":
        pixel_sigma = value * 255.0
        noisy_bytes = bytes(
            _clip(channel + rng.gauss(0.0, pixel_sigma))
            for channel in source.tobytes()
        )
        return Image.frombytes("RGB", source.size, noisy_bytes)

    if transform == "color_jitter":
        # Randomize both the factors and their order, as common training
        # augmentations do. A seed makes evaluation runs reproducible.
        operations = [
            (ImageEnhance.Brightness, rng.uniform(1 - value, 1 + value)),
            (ImageEnhance.Contrast,   rng.uniform(1 - value, 1 + value)),
            (ImageEnhance.Color,      rng.uniform(1 - value, 1 + value)),
        ]
        rng.shuffle(operations)
        result = source
        for enhancer, factor in operations:
            result = enhancer(result).enhance(factor)
        return result

    if transform == "center_crop":
        crop_width = max(1, round(width * value))
        crop_height = max(1, round(height * value))
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        cropped = source.crop((left, top, left + crop_width, top + crop_height))
        return cropped.resize((width, height), Image.Resampling.BICUBIC)


def _load_rgb(image: Image.Image | str | Path) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB").copy()
    with Image.open(image) as opened:
        return opened.convert("RGB").copy()


def _clip(value: float) -> int:
    return max(0, min(255, round(value)))
