"""Temperature scaling calibration for binary classifiers.

After training, a classifier's raw logits may be over- or under-confident.
Temperature scaling fits a single scalar T on an independent calibration split:

    calibrated_prob = sigmoid(logit / T)

T is optimised to minimise NLL on the calibration set.

This module also computes:
    - ECE  (Expected Calibration Error, M=15 bins)
    - Brier Score

Usage
-----
    from calibration.temperature_scaling import TemperatureScaler

    scaler = TemperatureScaler()
    scaler.fit(logits, labels)          # logits: np.ndarray [N], labels: [N] in {0,1}
    cal_probs = scaler.predict_proba(logits)

    metrics = scaler.calibration_metrics(logits, labels)
    print(metrics)  # {"ece": ..., "brier": ..., "temperature": ...}
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit  # sigmoid


def _nll(temperature: float, logits: np.ndarray, labels: np.ndarray) -> float:
    """Binary NLL loss for a given temperature."""
    probs = expit(logits / max(temperature, 1e-8))
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    return -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))


def _ece(probs: np.ndarray, labels: np.ndarray, *, n_bins: int = 15) -> float:
    """Expected Calibration Error (equal-width bins on [0, 1])."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean()
        ece += mask.sum() / n * abs(bin_conf - bin_acc)
    return float(ece)


class TemperatureScaler:
    """Fit and apply temperature scaling on raw binary logits.

    Attributes:
        temperature_: Fitted temperature (1.0 before fitting).
    """

    def __init__(self) -> None:
        self.temperature_: float = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        """Fit temperature T to minimise NLL on the calibration split.

        Args:
            logits: Raw logits, shape ``[N]``.
            labels: Binary labels (0 or 1), shape ``[N]``.

        Returns:
            self (for chaining).
        """
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)

        result = minimize_scalar(
            _nll,
            bounds=(0.01, 10.0),
            method="bounded",
            args=(logits, labels),
        )
        self.temperature_ = float(result.x)
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities for positive (AI-generated) class.

        Args:
            logits: Raw logits, shape ``[N]``.

        Returns:
            Calibrated probabilities in ``[0, 1]``, shape ``[N]``.
        """
        return expit(np.asarray(logits, dtype=np.float64) / self.temperature_)

    def calibration_metrics(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        *,
        n_bins: int = 15,
    ) -> dict[str, float]:
        """Compute calibration quality metrics on the given split.

        Args:
            logits: Raw logits, shape ``[N]``.
            labels: Binary labels, shape ``[N]``.
            n_bins: Number of bins for ECE computation.

        Returns:
            Dict with keys ``temperature``, ``ece``, ``brier``.
        """
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)

        probs = self.predict_proba(logits)
        ece = _ece(probs, labels, n_bins=n_bins)
        brier = float(np.mean((probs - labels) ** 2))

        return {
            "temperature": self.temperature_,
            "ece": ece,
            "brier": brier,
        }
