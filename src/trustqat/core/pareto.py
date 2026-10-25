"""Multi-objective (Pareto) candidate selection.

A candidate is retained when no other candidate is at least as good in every objective
and strictly better in at least one (standard Pareto dominance). Constraints such as a
minimum validation accuracy remove candidates before dominance is checked.

Objectives are given as ``{key: "max" | "min"}``; keys are resolved against
:class:`~trustqat.core.records.EpochRecord` (``"val_accuracy"``, ``"cost.ebops"``, ...).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal

import numpy as np

from .records import EpochRecord

Direction = Literal['max', 'min']

_SIDES = {'max': 1.0, 'min': -1.0}


def _sides(directions: Sequence[str]) -> np.ndarray:
    try:
        return np.array([_SIDES[d] for d in directions])
    except KeyError as e:
        raise ValueError(f"objective direction must be 'max' or 'min', got {e.args[0]!r}") from None


def dominates(a: Sequence[float], b: Sequence[float], directions: Sequence[str]) -> bool:
    """True if ``a`` Pareto-dominates ``b``."""
    s = _sides(directions)
    a_, b_ = s * np.asarray(a, float), s * np.asarray(b, float)
    return bool(np.all(a_ >= b_) and np.any(a_ > b_))


def pareto_front(points: Sequence[Sequence[float]], directions: Sequence[str]) -> list[int]:
    """Indices of the non-dominated rows of ``points`` (shape ``(n, k)``), in input order."""
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return []
    if pts.ndim != 2 or pts.shape[1] != len(directions):
        raise ValueError(f'points must have shape (n, {len(directions)}), got {pts.shape}')
    p = pts * _sides(directions)
    keep = []
    for i in range(len(p)):
        ge = np.all(p >= p[i], axis=1)
        gt = np.any(p > p[i], axis=1)
        if not np.any(ge & gt):
            keep.append(i)
    return keep


def _check_constraint(value: float, rule: Mapping[str, float]) -> bool:
    for op, bound in rule.items():
        if op == 'min' and not value >= bound:
            return False
        if op == 'max' and not value <= bound:
            return False
        if op == 'gt' and not value > bound:
            return False
        if op == 'lt' and not value < bound:
            return False
        if op not in ('min', 'max', 'gt', 'lt'):
            raise ValueError(f"unknown constraint operator {op!r} (use 'min', 'max', 'gt', 'lt')")
    return True


def make_filter(constraints: Mapping[str, Mapping[str, float]] | None) -> Callable[[EpochRecord], bool]:
    constraints = dict(constraints or {})

    def ok(record: EpochRecord) -> bool:
        return all(_check_constraint(record.get(k), rule) for k, rule in constraints.items())

    return ok


def select_records(
    records: Iterable[EpochRecord],
    objectives: Mapping[str, Direction],
    constraints: Mapping[str, Mapping[str, float]] | None = None,
    require_checkpoint: bool = False,
) -> list[EpochRecord]:
    """Pareto-optimal records among those satisfying ``constraints``."""
    ok = make_filter(constraints)
    cand = [r for r in records if ok(r) and (r.checkpoint or not require_checkpoint)]
    keys, dirs = list(objectives), list(objectives.values())
    idx = pareto_front([[r.get(k) for k in keys] for r in cand], dirs)
    return [cand[i] for i in idx]


class ParetoArchive:
    """Online Pareto archive, used to decide which checkpoints to keep during training.

    ``add`` returns ``(accepted, evicted)``: whether the new record joins the front and
    which previously archived records it dominates.
    """

    def __init__(
        self,
        objectives: Mapping[str, Direction],
        constraints: Mapping[str, Mapping[str, float]] | None = None,
    ):
        self.objectives = dict(objectives)
        self.constraints = dict(constraints or {})
        self._filter = make_filter(self.constraints)
        self._dirs = list(self.objectives.values())
        self.members: list[EpochRecord] = []

    def _vec(self, r: EpochRecord) -> list[float]:
        return [r.get(k) for k in self.objectives]

    def admissible(self, record: EpochRecord) -> bool:
        return self._filter(record)

    def add(self, record: EpochRecord) -> tuple[bool, list[EpochRecord]]:
        if not self._filter(record):
            return False, []
        v = self._vec(record)
        s = _sides(self._dirs)
        for m in self.members:
            mv = s * np.asarray(self._vec(m), float)
            # dominated by, or identical to, an archived point: reject
            if np.all(mv >= s * np.asarray(v, float)):
                return False, []
        evicted = [m for m in self.members if dominates(v, self._vec(m), self._dirs)]
        self.members = [m for m in self.members if m not in evicted] + [record]
        return True, evicted

    def to_dict(self) -> dict[str, Any]:
        return {
            'objectives': self.objectives,
            'constraints': self.constraints,
            'members': [m.to_dict() for m in self.members],
        }
