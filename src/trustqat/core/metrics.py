"""Reference (numpy) implementations of predictive and calibration metrics.

These are the dataset-level metrics used for validation and test reporting. The
framework adapters (``trustqat.keras``, ``trustqat.torch``) are tested against them.

Binning convention (top-label ECE, Guo et al. 2017): the confidence range [0, 1] is split
into ``n_bins`` equal-width bins and a sample with confidence ``c`` goes to bin
``min(floor(c * n_bins), n_bins - 1)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def softmax(logits: ArrayLike) -> NDArray[np.float64]:
    """Numerically stable softmax over the last axis, computed in float64."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def _probs_and_labels(outputs: ArrayLike, labels: ArrayLike, from_logits: bool):
    probs = softmax(outputs) if from_logits else np.asarray(outputs, dtype=np.float64)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    if probs.ndim != 2:
        raise ValueError(f'expected outputs of shape (N, K), got {probs.shape}')
    if probs.shape[0] != labels.shape[0]:
        raise ValueError(f'outputs and labels disagree on N: {probs.shape[0]} vs {labels.shape[0]}')
    return probs, labels


def _bin_ids(confidence: NDArray, n_bins: int) -> NDArray[np.int64]:
    if n_bins <= 0:
        raise ValueError('n_bins must be positive')
    ids = np.floor(confidence * n_bins).astype(np.int64)
    return np.clip(ids, 0, n_bins - 1)


def calibration_bins(
    outputs: ArrayLike, labels: ArrayLike, n_bins: int = 15, from_logits: bool = True
) -> list[dict[str, Any]]:
    """Per-bin count, accuracy, mean confidence and gap (empty bins have ``None`` stats)."""
    probs, labels = _probs_and_labels(outputs, labels, from_logits)
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    ids = _bin_ids(conf, n_bins)
    bins = []
    for b in range(n_bins):
        mask = ids == b
        count = int(mask.sum())
        if count == 0:
            bins.append({'bin': b, 'count': 0, 'accuracy': None, 'confidence': None, 'gap': None})
            continue
        acc = float(correct[mask].mean())
        cbar = float(conf[mask].mean())
        bins.append({'bin': b, 'count': count, 'accuracy': acc, 'confidence': cbar, 'gap': abs(acc - cbar)})
    return bins


def expected_calibration_error(
    outputs: ArrayLike, labels: ArrayLike, n_bins: int = 15, from_logits: bool = True
) -> float:
    """Top-label expected calibration error, ``sum_m |B_m|/N * |acc(B_m) - conf(B_m)|``."""
    bins = calibration_bins(outputs, labels, n_bins, from_logits)
    n = sum(b['count'] for b in bins)
    return float(sum(b['count'] / n * b['gap'] for b in bins if b['count']))


def maximum_calibration_error(
    outputs: ArrayLike, labels: ArrayLike, n_bins: int = 15, from_logits: bool = True
) -> float:
    """Largest per-bin gap over non-empty bins."""
    bins = calibration_bins(outputs, labels, n_bins, from_logits)
    return float(max((b['gap'] for b in bins if b['count']), default=0.0))


def negative_log_likelihood(outputs: ArrayLike, labels: ArrayLike, from_logits: bool = True) -> float:
    probs, labels = _probs_and_labels(outputs, labels, from_logits)
    p = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def brier_score(outputs: ArrayLike, labels: ArrayLike, from_logits: bool = True) -> float:
    """Multi-class Brier score, ``mean_i sum_k (p_ik - y_ik)^2``."""
    probs, labels = _probs_and_labels(outputs, labels, from_logits)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def calibration_report(
    outputs: ArrayLike, labels: ArrayLike, n_bins: int = 15, from_logits: bool = True, with_bins: bool = True
) -> dict[str, Any]:
    """Accuracy and calibration metrics in one dictionary (the format written by the benchmarks)."""
    probs, labels_ = _probs_and_labels(outputs, labels, from_logits)
    report: dict[str, Any] = {
        'acc': float(np.mean(probs.argmax(axis=1) == labels_)),
        'ece': expected_calibration_error(probs, labels_, n_bins, from_logits=False),
        'mce': maximum_calibration_error(probs, labels_, n_bins, from_logits=False),
        'nll': negative_log_likelihood(probs, labels_, from_logits=False),
        'brier': brier_score(probs, labels_, from_logits=False),
        'n_bins': int(n_bins),
    }
    if with_bins:
        report['bins'] = calibration_bins(probs, labels_, n_bins, from_logits=False)
    return report
