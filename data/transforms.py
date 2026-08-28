"""Image transforms and training augmentation policy.

Low-level transforms
--------------------
:func:`apply_transform`        — apply one transform with an explicit parameter value
:func:`apply_random_transform` — apply one transform with a randomly sampled value

All transforms return an RGB ``PIL.Image`` with the same size as the input,
so results can be passed directly to any torchvision pre-processing pipeline.

Allowed parameter values (strictly following the official table):
    jpeg_compression : quality in {90, 70, 50, 30}
    gaussian_blur    : sigma  in {0.5, 1.0, 2.0}
    resize           : scale  in {0.5, 0.25}   (downscale → upscale back)
    gaussian_noise   : sigma  in {0.02, 0.05, 0.10}
    color_jitter     : strength = 0.2  (±20 %)
    center_crop      : fraction = 0.8  (80 %)

Training augmentation policy
-----------------------------
:func:`build_train_augment`  — 30 % clean, 70 % single random transform
:func:`build_eval_augment`   — fixed transform for evaluation

Both real and AI-generated images receive the **same** augmentation pipeline to
prevent the model from learning spurious "has/hasn't been processed" shortcuts
(cf. DDA, NeurIPS 2025).
"""

from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Callable, Literal

from PIL import Image, ImageEnhance, ImageFilter


# ---------------------------------------------------------------------------
# Types and constants
# ---------------------------------------------------------------------------

TransformName = Literal[
    "jpeg_compression",
    "gaussian_blur",
    "resize",
    "gaussian_noise",
    "color_jitter",
    "center_crop",
]

TRANSFORM_POOL: tuple[TransformName, ...] = (
    "jpeg_compression",
    "gaussian_blur",
    "resize",
    "gaussian_noise",
    "color_jitter",
    "center_crop",
)

_ALLOWED: dict[str, tuple[float, ...]] = {
    "jpeg_compression": (90, 70, 50, 30),
    "gaussian_blur":    (0.5, 1.0, 2.0),
    "resize":           (0.5, 0.25),
    "gaussian_noise":   (0.02, 0.05, 0.10),
    "color_jitter":     (0.2,),
    "center_crop":      (0.8,),
}


# ---------------------------------------------------------------------------
# Low-level transforms
# ---------------------------------------------------------------------------

def apply_transform(
    image: Image.Image | str | Path,
    transform: TransformName,
    *,
    value: float | int,
    seed: int | None = None,
) -> Image.Image:
    """Apply one transform with a fixed, explicitly specified parameter value.

    Args:
        image: A Pillow image or a path to an image file.
        transform: One of the six official transform names.
        value: Must be one of the allowed values from the official table.
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


def apply_random_transform(
    image: Image.Image | str | Path,
    transform: TransformName,
    *,
    seed: int | None = None,
) -> tuple[Image.Image, float]:
    """Apply one transform with a parameter value sampled uniformly from the allowed set.

    Args:
        image: A Pillow image or a path to an image file.
        transform: One of the six official transform names.
        seed: Seed for reproducible value sampling and transform execution.

    Returns:
        A tuple of ``(transformed_image, sampled_value)``.

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


# ---------------------------------------------------------------------------
# Training / evaluation augment factories
# ---------------------------------------------------------------------------

def build_train_augment(*, clean_prob: float = 0.3) -> Callable[[Image.Image], Image.Image]:
    """Return a callable implementing the official single-transform training policy.

    With probability ``clean_prob`` the image is returned untouched; otherwise
    a single transform is drawn uniformly from :data:`TRANSFORM_POOL` and
    applied with a randomly sampled allowed parameter value.

    Args:
        clean_prob: Probability of no augmentation (default 0.3).

    Returns:
        A function ``augment(image: PIL.Image) -> PIL.Image``.
    """
    if not 0.0 <= clean_prob <= 1.0:
        raise ValueError(f"clean_prob must be in [0, 1], got {clean_prob}")

    def augment(image: Image.Image) -> Image.Image:
        if random.random() < clean_prob:
            return image
        transform = random.choice(TRANSFORM_POOL)
        augmented, _ = apply_random_transform(image, transform)
        return augmented

    return augment


def build_eval_augment(
    *,
    transform: TransformName,
    value: float,
) -> Callable[[Image.Image], Image.Image]:
    """Return a callable that applies a fixed transform for evaluation.

    Args:
        transform: One of the official transform names.
        value: The exact parameter value (must be in the allowed set).

    Returns:
        A function ``augment(image: PIL.Image) -> PIL.Image``.
    """
    def augment(image: Image.Image) -> Image.Image:
        return apply_transform(image, transform, value=value)

    return augment


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
