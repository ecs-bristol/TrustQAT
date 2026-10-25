"""HLS4ML LHC jet dataset (150 constituents) for the JEDI-linear benchmark.

Adapted from JEDI-linear (Apache-2.0, Copyright 2025 Chang Sun); see NOTICE. The
normalisation, shuffling and feature selection are kept identical so that runs match
the TrustQAT-AE artifact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: feature indices of (pT, eta, phi) used by the 3-feature models
PT_ETA_PHI = [5, 8, 11]
LABELS = 'gqWZt'


def get_data(data_path: str | Path, n_constituents: int, pt_eta_phi: bool):
    """Return ``x_train_val, x_test, y_train_val, y_test``.

    Features are standardised with the train/val mean and std (per feature). The
    train/val set is shuffled with the *global* numpy RNG, so seed ``np.random`` first
    to reproduce a split; the first ``val_split`` fraction is used for validation.
    """
    import h5py

    data_path = Path(data_path)
    with h5py.File(data_path / '150c-train.h5') as f:
        x_train_val = np.array(f['feature'][:, :n_constituents])
        y_train_val = np.array(f['label'])
    with h5py.File(data_path / '150c-test.h5') as f:
        x_test = np.array(f['feature'][:, :n_constituents])
        y_test = np.array(f['label'])

    x_train_val = x_train_val.astype(np.float32)
    y_train_val = y_train_val.astype(np.int32)
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.int32)

    scale = np.std(x_train_val, axis=(0, 1), keepdims=True)
    shift = np.mean(x_train_val, axis=(0, 1), keepdims=True)
    x_train_val = (x_train_val - shift) / scale
    x_test = (x_test - shift) / scale

    order = np.arange(len(x_train_val))
    np.random.shuffle(order)
    x_train_val, y_train_val = x_train_val[order], y_train_val[order]
    if pt_eta_phi:
        x_train_val = x_train_val[..., PT_ETA_PHI]
        x_test = x_test[..., PT_ETA_PHI]
    return x_train_val, x_test, y_train_val, y_test


def split_train_val(x, y, val_split: float = 0.1, input_dtype: str | None = 'float16'):
    """First ``val_split`` of the (already shuffled) data is validation, as in the artifact."""
    if input_dtype:
        x = x.astype(input_dtype)
    y = y.astype(np.int32)
    n_val = int(len(x) * val_split)
    return x[n_val:], y[n_val:], x[:n_val], y[:n_val]
