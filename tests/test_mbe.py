"""Smoke checks for Multi-Scale Bounded Enhancement. No GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.mbe import (
    ALPHA_FINE,
    ALPHA_MID,
    MultiScaleBoundedEnhancement,
    enhance,
)


def _gray(*, value: int = 128, size: int = 64) -> Image.Image:
    return Image.new("RGB", (size, size), (value, value, value))


def test_same_size_and_mode() -> None:
    src = Image.new("RGB", (97, 41), (40, 80, 160))
    out = enhance(src)
    assert out.size == src.size
    assert out.mode == "RGB"


def test_gray_stays_gray() -> None:
    out = np.asarray(enhance(_gray()), dtype=np.int16)
    chroma = np.max(out, axis=2) - np.min(out, axis=2)
    assert int(chroma.max()) <= 1


def test_boost_is_bounded() -> None:
    """Per-pixel luma shift cannot exceed α1 + α2 (tanh ≤ 1)."""
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    src = Image.fromarray(noise, mode="RGB")
    before = np.asarray(src, dtype=np.float64) / 255.0
    after = np.asarray(enhance(src), dtype=np.float64) / 255.0
    luma = np.array([0.299, 0.587, 0.114])
    delta = np.abs((after - before) @ luma)
    assert float(delta.max()) <= ALPHA_FINE + ALPHA_MID + 1.5 / 255.0


def test_callable_matches_enhance() -> None:
    src = Image.new("RGB", (32, 32), (200, 30, 90))
    a = np.asarray(enhance(src))
    b = np.asarray(MultiScaleBoundedEnhancement()(src))
    assert np.array_equal(a, b)


def test_rejects_bad_sigmas() -> None:
    try:
        enhance(_gray(), sigma_fine=1.6, sigma_mid=0.8)
    except ValueError:
        return
    raise AssertionError("expected ValueError when sigma_mid <= sigma_fine")


if __name__ == "__main__":
    test_same_size_and_mode()
    test_gray_stays_gray()
    test_boost_is_bounded()
    test_callable_matches_enhance()
    test_rejects_bad_sigmas()
    print("mbe ok")
