"""Default Keras training loop for TrustQAT.

The trainer is backend-agnostic: everything QAT-specific comes from the backend object
(cost surrogate, extra callbacks such as HGQ's beta schedule, dataset placement, how to
save checkpoints, whether precision can be frozen).

Config sections used (``dict`` or OmegaConf):

``train``
    ``batch_size``, ``epochs``, ``optimizer`` (``{name, **kwargs}``, default Adam),
    ``lr_schedule`` (``{name: cosine_restarts | constant, **kwargs}``), ``early_stopping``
    (Keras ``EarlyStopping`` kwargs plus ``enabled``), ``freeze_precision`` (bool),
    ``shuffle`` (default False), ``drop_last`` (default True), ``log_every``.
``trust``
    ``objectives``: list of ``{name, weight, **params}``; empty or missing = task loss only.
``selection``
    ``objectives`` (``{key: max|min}``) and ``constraints`` (``{key: {min|max: value}}``).
    Defaults to validation accuracy (max), each trust metric, and each backend cost (min).
``backend``
    passed unchanged to ``backend.training_hooks``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import keras

from ..backends.base import FREEZE_PRECISION, require_capability
from ..objectives import objectives_from_config
from .callbacks import EpochLogger, ParetoCheckpoint
from .objectives import TrustAwareLoss
from .schedules import get_schedule


def to_plain(cfg: Any) -> Any:
    """OmegaConf -> plain containers (no-op for dicts/lists/scalars)."""
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(cfg, (DictConfig, ListConfig)):
            return OmegaConf.to_container(cfg, resolve=True)
    except ImportError:  # pragma: no cover
        pass
    return cfg


def default_selection(backend, objectives) -> dict[str, Any]:
    sel: dict[str, str] = {'val_accuracy': 'max'}
    for obj, _ in objectives:
        sel[f'val_{obj.metric_name}'] = obj.direction
    for name in backend.cost_names:
        sel[f'cost.{name}'] = 'min'
    return {'objectives': sel, 'constraints': {}}


class KerasTrainer:
    def __init__(self, backend, cfg: Any):
        if backend.framework != 'keras':
            raise ValueError(f'KerasTrainer needs a keras backend, got {backend.name!r} ({backend.framework})')
        cfg = to_plain(cfg) or {}
        self.backend = backend
        self.train_cfg: dict[str, Any] = dict(cfg.get('train') or {})
        self.trust_cfg: dict[str, Any] = dict(cfg.get('trust') or {})
        self.backend_cfg: dict[str, Any] = dict(cfg.get('backend') or {})
        self.objectives = objectives_from_config(self.trust_cfg.get('objectives'))
        self.selection = cfg.get('selection') or default_selection(backend, self.objectives)

    # ------------------------------------------------------------------ pieces
    def loss(self):
        if not self.objectives:
            return keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return TrustAwareLoss(objectives=self.trust_cfg['objectives'], from_logits=True)

    def metrics(self) -> list:
        return ['accuracy'] + [obj.keras_metric() for obj, _ in self.objectives]

    def optimizer(self):
        spec = dict(self.train_cfg.get('optimizer') or {'name': 'adam'})
        name = spec.pop('name', 'adam')
        return keras.optimizers.get({'class_name': name, 'config': spec})

    def callbacks(self, save_dir: Path) -> list:
        cbs: list = []
        sched = self.train_cfg.get('lr_schedule')
        if sched:
            sched = dict(sched)
            cbs.append(keras.callbacks.LearningRateScheduler(get_schedule(sched.pop('name'), **sched)))
        cbs.extend(self.backend.training_hooks(self.backend_cfg))
        cbs.append(
            ParetoCheckpoint(
                save_dir,
                self.backend,
                objectives=self.selection['objectives'],
                constraints=self.selection.get('constraints'),
            )
        )
        keys = ['loss', 'val_loss', 'accuracy', 'val_accuracy']
        keys += [f'{p}{obj.metric_name}' for obj, _ in self.objectives for p in ('', 'val_')]
        keys += list(self.backend.cost_names) + ['beta']
        cbs.append(EpochLogger(keys, every=int(self.train_cfg.get('log_every', 1))))
        cbs.append(keras.callbacks.TerminateOnNaN())
        es = dict(self.train_cfg.get('early_stopping') or {})
        if es.pop('enabled', False):
            cbs.append(keras.callbacks.EarlyStopping(verbose=1, **es))
        return cbs

    def dataset(self, x, y, shuffle: bool):
        bs = int(self.train_cfg['batch_size'])
        drop_last = bool(self.train_cfg.get('drop_last', True))
        ds = self.backend.make_dataset(x, y, batch_size=bs, shuffle=shuffle, drop_last=drop_last, cfg=self.backend_cfg)
        if ds is None:
            from .data import ArrayDataset

            ds = ArrayDataset(x, y, batch_size=bs, shuffle=shuffle, drop_last=drop_last)
        return ds

    # ------------------------------------------------------------------ main
    def fit(self, model, x_train, y_train, x_val, y_val, save_dir: str | Path):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if self.train_cfg.get('freeze_precision', False):
            require_capability(self.backend, FREEZE_PRECISION)
            self.backend.freeze_precision(model)

        model.compile(optimizer=self.optimizer(), loss=self.loss(), metrics=self.metrics())
        shuffle = bool(self.train_cfg.get('shuffle', False))
        train_ds = self.dataset(x_train, y_train, shuffle=shuffle)
        val_ds = self.dataset(x_val, y_val, shuffle=False)

        with (save_dir / 'selection.json').open('w') as f:
            json.dump(self.selection, f, indent=2)

        model.fit(
            train_ds,
            epochs=int(self.train_cfg['epochs']),
            validation_data=val_ds,
            callbacks=self.callbacks(save_dir),
            verbose=0,
        )
        history = {k: [float(v) for v in vs] for k, vs in model.history.history.items()}
        with (save_dir / 'history.json').open('w') as f:
            json.dump(history, f)
        self.backend.save(model, save_dir / f'last{self.backend.checkpoint_suffix}')
        return model, history
