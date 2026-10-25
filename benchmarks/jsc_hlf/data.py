"""HLF JSC data: OpenML ``hls4ml_lhc_jets_hlf`` (16 features, 5 classes, 830k jets).

Split procedure (as in the HGQ artifact and TrustQAT-AE): 80/20 train+val/test split with
``train_test_split(random_state=42)``, ``StandardScaler`` fitted on train+val, then the
first 90% of train+val for training and the last 10% for validation.

    python -m benchmarks.jsc_hlf.data --cache openml_cache.h5 --output dataset/jsc_hlf_openml_splits.npz

``--cache`` is an HDF5 file with datasets ``X`` and ``y``; it is downloaded from OpenML
(and written) when it does not exist.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

CLASSES = {'g': 0, 'q': 1, 'w': 2, 'z': 3, 't': 4}


def load_openml(cache: str | Path) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    cache = Path(cache)
    if not cache.exists():
        from sklearn.datasets import fetch_openml

        print('Downloading hls4ml_lhc_jets_hlf from OpenML...', flush=True)
        data = fetch_openml('hls4ml_lhc_jets_hlf')
        x = np.array(data['data'])
        y = np.array([CLASSES[c] for c in data['target']])
        cache.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(cache, 'w') as f:
            f.create_dataset('X', data=x, compression='gzip')
            f.create_dataset('y', data=y, compression='gzip')
        return x, y
    with h5py.File(cache, 'r') as f:
        return np.array(f['X']), np.array(f['y'])


def make_splits(x: np.ndarray, y: np.ndarray, seed: int = 42) -> dict[str, np.ndarray]:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    x_tv, x_test, y_tv, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    x_tv, x_test = x_tv.astype(np.float32), x_test.astype(np.float32)
    scaler = StandardScaler()
    x_tv = scaler.fit_transform(x_tv).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)
    y_tv, y_test = y_tv.astype(np.int32), y_test.astype(np.int32)
    n_train = int(0.9 * len(x_tv))
    return {
        'x_train': x_tv[:n_train],
        'y_train': y_tv[:n_train],
        'x_val': x_tv[n_train:],
        'y_val': y_tv[n_train:],
        'x_test': x_test,
        'y_test': y_test,
    }


def load_splits(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'{path} not found; create it with: python -m benchmarks.jsc_hlf.data --output {path}')
    with np.load(path) as arrays:
        return {k: arrays[k] for k in arrays.files}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cache', default='dataset/openml_cache.h5')
    p.add_argument('--output', default='dataset/jsc_hlf_openml_splits.npz')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args(argv)
    splits = make_splits(*load_openml(args.cache), seed=args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **splits)
    print(' '.join(f'{k}={v.shape}' for k, v in splits.items()), f'-> {out}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
