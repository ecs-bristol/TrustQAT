"""Paired comparisons between a baseline and a TrustQAT design (e.g. HGQ vs TrustQAT)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def relative_reduction(baseline: float, new: float) -> float:
    """``(baseline - new) / baseline``; positive means ``new`` is lower (better for ECE)."""
    if baseline == 0:
        raise ZeroDivisionError('baseline is zero')
    return (baseline - new) / baseline


def paired_summary(
    pairs: Sequence[tuple[dict[str, float], dict[str, float]]],
    metric: str = 'ece',
    accuracy: str = 'acc',
) -> dict[str, Any]:
    """Summarise matched ``(baseline, candidate)`` result pairs.

    Reports the median relative reduction of ``metric`` (cases where it increased are
    included, as in the paper), how many pairs improved, and the accuracy changes in
    percentage points.
    """
    if not pairs:
        raise ValueError('no pairs given')
    rel = np.array([relative_reduction(b[metric], c[metric]) for b, c in pairs])
    dacc = np.array([(c[accuracy] - b[accuracy]) * 100.0 for b, c in pairs])
    return {
        'n_pairs': len(pairs),
        'n_improved': int(np.sum(rel > 0)),
        f'median_relative_{metric}_reduction': float(np.median(rel)),
        'median_abs_accuracy_change_pp': float(np.median(np.abs(dacc))),
        'accuracy_change_pp_range': (float(dacc.min()), float(dacc.max())),
    }
