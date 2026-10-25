"""Keras callbacks: per-epoch records with online Pareto checkpointing, and a compact logger."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import keras

from ..core.pareto import ParetoArchive
from ..core.records import EpochRecord, write_record


def _to_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class ParetoCheckpoint(keras.callbacks.Callback):
    """Record every epoch and keep checkpoints only for the current Pareto front.

    At the end of each epoch this callback

    1. asks the backend for the hardware-cost surrogate(s) (``backend.cost(model)``) and
       adds them to the Keras logs, so later callbacks (e.g. early stopping) can see them;
    2. appends an :class:`~trustqat.core.records.EpochRecord` to ``records.jsonl``;
    3. if the epoch is admissible (constraints) and non-dominated, saves a checkpoint
       through ``backend.save`` and deletes checkpoints it dominates;
    4. rewrites ``pareto.json`` with the current front.
    """

    def __init__(
        self,
        directory: str | Path,
        backend,
        objectives: Mapping[str, str],
        constraints: Mapping[str, Mapping[str, float]] | None = None,
        records_name: str = 'records.jsonl',
    ):
        super().__init__()
        self.directory = Path(directory)
        self.backend = backend
        self.archive = ParetoArchive(objectives, constraints)
        self.records_path = self.directory / records_name
        self.ckpt_dir = self.directory / 'ckpts'

    def on_train_begin(self, logs=None):
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        cost = {k: float(v) for k, v in self.backend.cost(self.model).items()}
        for k, v in cost.items():
            logs.setdefault(k, v)
        metrics = {k: f for k, v in logs.items() if k not in cost and (f := _to_float(v)) is not None}
        record = EpochRecord(epoch=int(epoch), metrics=metrics, cost=cost)

        if all(math.isfinite(record.get(k)) for k in self.archive.objectives):
            accepted, evicted = self.archive.add(record)
        else:
            accepted, evicted = False, []
        if accepted:
            path = self.ckpt_dir / f'epoch={epoch:04d}{self.backend.checkpoint_suffix}'
            self.backend.save(self.model, path)
            record.checkpoint = str(path.relative_to(self.directory))
            for old in evicted:
                if old.checkpoint:
                    (self.directory / old.checkpoint).unlink(missing_ok=True)
        write_record(self.records_path, record)
        with (self.directory / 'pareto.json').open('w') as f:
            json.dump(self.archive.to_dict(), f, indent=2)


class EpochLogger(keras.callbacks.Callback):
    """One line per epoch with the chosen log keys."""

    def __init__(self, keys: Sequence[str] = ('loss', 'val_loss', 'accuracy', 'val_accuracy'), every: int = 1):
        super().__init__()
        self.keys = list(keys)
        self.every = max(1, int(every))

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or epoch % self.every:
            return
        parts = [f'epoch {epoch:4d}']
        for k in self.keys:
            if k in logs:
                v = float(logs[k])
                parts.append(f'{k}={v:.4g}' if abs(v) < 1e4 else f'{k}={v:.0f}')
        lr = getattr(getattr(self.model, 'optimizer', None), 'learning_rate', None)
        if lr is not None:
            parts.append(f'lr={float(keras.ops.convert_to_numpy(lr)):.2e}')
        print(' '.join(parts), flush=True)
