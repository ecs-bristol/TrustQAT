"""Per-epoch records: the contract between any training loop and TrustQAT selection.

One JSON object per line (JSONL). The canonical form is::

    {"epoch": 37,
     "metrics": {"val_accuracy": 0.7349, "val_ece": 0.0041, ...},
     "cost": {"ebops": 171253},
     "checkpoint": "ckpts/epoch=0037.keras"}

``checkpoint`` is where that epoch's checkpoint was saved, if it was. The file can be
removed later when a newer epoch dominates it; ``pareto.json`` next to the records lists
the current front. Records are append-only.

A flat form is also accepted when reading, which makes it easy to integrate an existing
training script: every numeric top-level key other than ``epoch`` goes to ``metrics`` and
keys prefixed ``cost.`` go to ``cost``::

    {"epoch": 37, "val_accuracy": 0.7349, "val_ece": 0.0041, "cost.ebops": 171253}
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass
class EpochRecord:
    epoch: int
    metrics: dict[str, float] = field(default_factory=dict)
    cost: dict[str, float] = field(default_factory=dict)
    checkpoint: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str) -> float:
        """Resolve ``"epoch"``, ``"cost.<name>"`` or a metric name."""
        if key == 'epoch':
            return float(self.epoch)
        if key.startswith('cost.'):
            name = key[5:]
            if name not in self.cost:
                raise KeyError(f'cost {name!r} not in record (available: {sorted(self.cost)})')
            return float(self.cost[name])
        if key not in self.metrics:
            raise KeyError(f'metric {key!r} not in record (available: {sorted(self.metrics)})')
        return float(self.metrics[key])

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {'epoch': int(self.epoch), 'metrics': self.metrics, 'cost': self.cost}
        if self.checkpoint is not None:
            d['checkpoint'] = self.checkpoint
        if self.extra:
            d['extra'] = self.extra
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EpochRecord:
        validate_record(d)
        if 'metrics' in d:
            return cls(
                epoch=int(d['epoch']),
                metrics={k: float(v) for k, v in d['metrics'].items()},
                cost={k: float(v) for k, v in d.get('cost', {}).items()},
                checkpoint=d.get('checkpoint'),
                extra=dict(d.get('extra', {})),
            )
        metrics, cost = {}, {}
        for k, v in d.items():
            if k in ('epoch', 'checkpoint', 'extra'):
                continue
            if k.startswith('cost.'):
                cost[k[5:]] = float(v)
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                metrics[k] = float(v)
        return cls(int(d['epoch']), metrics, cost, d.get('checkpoint'), dict(d.get('extra', {})))


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_record(d: dict[str, Any]) -> None:
    """Raise ``ValueError`` if ``d`` is not a valid record (canonical or flat form)."""
    if not isinstance(d, dict):
        raise ValueError(f'record must be a dict, got {type(d).__name__}')
    if 'epoch' not in d or not isinstance(d['epoch'], int) or isinstance(d['epoch'], bool) or d['epoch'] < 0:
        raise ValueError('record needs a non-negative integer "epoch"')
    if 'checkpoint' in d and d['checkpoint'] is not None and not isinstance(d['checkpoint'], str):
        raise ValueError('"checkpoint" must be a string path or null')
    if 'metrics' in d:
        for section in ('metrics', 'cost'):
            values = d.get(section, {})
            if not isinstance(values, dict):
                raise ValueError(f'"{section}" must be a mapping')
            for k, v in values.items():
                if not _is_number(v):
                    raise ValueError(f'{section}.{k} must be a number, got {v!r}')
        for k, v in d.get('cost', {}).items():
            if not math.isfinite(v) or v < 0:
                raise ValueError(f'cost.{k} must be finite and non-negative, got {v!r}')
        if not d['metrics']:
            raise ValueError('"metrics" must not be empty')
    else:
        if not any(_is_number(v) for k, v in d.items() if k != 'epoch'):
            raise ValueError('flat record has no numeric metrics')


def write_record(path: str | Path, record: EpochRecord) -> None:
    """Append one record to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(record.to_dict(), allow_nan=True) + '\n')


def read_records(path: str | Path) -> list[EpochRecord]:
    records = []
    with Path(path).open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(EpochRecord.from_dict(json.loads(line)))
            except ValueError as e:
                raise ValueError(f'{path}:{line_no}: {e}') from None
    return records
