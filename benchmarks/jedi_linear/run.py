"""JEDI-linear benchmark driver: train, test (accuracy + calibration), export (da4ml RTL), select.

Usage (from the repository root, ``KERAS_BACKEND=jax``)::

    python -m benchmarks.jedi_linear -c benchmarks/jedi_linear/configs/trustqat/perminv-n16-f3.yaml -r train test select
    python -m benchmarks.jedi_linear -c CONFIG -r export --rtl-samples 10000
    python -m benchmarks.jedi_linear -c CONFIG -r train --set train.epochs=5 save_path=runs/smoke

See :mod:`benchmarks.common.driver` for what each phase writes.
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.common.driver import Data, plain, seed_all
from benchmarks.common.driver import main as driver_main


def load_data(cfg) -> Data:
    from .data import get_data, split_train_val

    x_train_val, x_test, y_train_val, y_test = get_data(
        cfg.data.path, int(cfg.data.n_constituents), bool(cfg.data.pt_eta_phi)
    )

    def splits():
        return split_train_val(
            x_train_val, y_train_val, float(cfg.data.get('val_split', 0.1)), cfg.data.get('input_dtype', 'float16')
        )

    return Data(x_test=x_test, y_test=y_test, x_calib=x_train_val, splits=splits)


def _factory(cfg, backend):
    from trustqat.models import get_model_factory

    from . import models  # noqa: F401  (registers the JEDI-linear factory)

    return get_model_factory(cfg.model.name, backend.name)


def build_model(cfg, backend):
    model = _factory(cfg, backend)(plain(cfg))
    model.summary()
    seed_all(int(cfg.seed))
    init = cfg.train.get('init_weights')
    if init:
        print(f'Loading initial weights from {init}', flush=True)
        model = backend.load_weights(model, Path(init))
    return model


def load_checkpoint(cfg, backend, path: Path):
    return backend.load_weights(_factory(cfg, backend)(plain(cfg)), path)


def main(argv: list[str] | None = None) -> int:
    return driver_main(
        argv, load_data=load_data, build_model=build_model, load_checkpoint=load_checkpoint, description=__doc__
    )


if __name__ == '__main__':
    raise SystemExit(main())
