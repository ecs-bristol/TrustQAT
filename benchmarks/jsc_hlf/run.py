"""HLF JSC benchmark driver: train, test (accuracy + calibration), export (da4ml RTL), select.

Usage (from the repository root, ``KERAS_BACKEND=jax``)::

    python -m benchmarks.jsc_hlf.data --cache /path/openml_cache.h5 --output dataset/jsc_hlf_openml_splits.npz
    python -m benchmarks.jsc_hlf -c benchmarks/jsc_hlf/configs/trustqat/ebops530.yaml -r train test select
    python -m benchmarks.jsc_hlf -c CONFIG -r export --rtl-samples 10000

See :mod:`benchmarks.common.driver` for what each phase writes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmarks.common.driver import Data, plain
from benchmarks.common.driver import main as driver_main


def load_data(cfg) -> Data:
    from .data import load_splits

    d = load_splits(cfg.data.path)
    x_calib = np.concatenate([d['x_train'], d['x_val']])
    return Data(
        x_test=d['x_test'],
        y_test=d['y_test'],
        x_calib=x_calib,
        splits=lambda: (d['x_train'], d['y_train'], d['x_val'], d['y_val']),
    )


def build_model(cfg, backend):
    from trustqat.models import get_model_factory

    from . import models  # noqa: F401  (registers the factory)

    print(f'Loading HGQ checkpoint {cfg.model.checkpoint}', flush=True)
    model = get_model_factory(cfg.model.name, backend.name)(plain(cfg))
    model.summary()
    return model


def load_checkpoint(cfg, backend, path: Path):
    return backend.load(path)


def main(argv: list[str] | None = None) -> int:
    return driver_main(
        argv, load_data=load_data, build_model=build_model, load_checkpoint=load_checkpoint, description=__doc__
    )


if __name__ == '__main__':
    raise SystemExit(main())
