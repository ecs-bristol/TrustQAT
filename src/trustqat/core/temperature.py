"""Post-hoc temperature scaling (Guo et al. 2017), as a baseline that composes with TrustQAT.

TrustQAT changes training and candidate selection; temperature scaling rescales the
logits of a fixed model. ``fit_temperature`` finds the scalar ``T > 0`` minimising the
validation NLL of ``softmax(logits / T)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .metrics import negative_log_likelihood


def apply_temperature(logits: ArrayLike, temperature: float) -> NDArray[np.float64]:
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    return np.asarray(logits, dtype=np.float64) / temperature


def fit_temperature(
    logits: ArrayLike,
    labels: ArrayLike,
    log_t_range: tuple[float, float] = (-3.0, 3.0),
    tol: float = 1e-5,
    max_iter: int = 200,
) -> float:
    """Golden-section search over ``log T`` (NLL is unimodal in ``log T`` in practice)."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels).reshape(-1)

    def nll(log_t: float) -> float:
        return negative_log_likelihood(z / np.exp(log_t), y)

    a, b = log_t_range
    g = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll(d)
    return float(np.exp((a + b) / 2.0))
