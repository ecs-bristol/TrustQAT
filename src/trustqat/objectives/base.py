"""Trustworthiness objectives: one spec, several framework implementations.

A :class:`TrustObjective` bundles

- a training-time loss term (per framework: :meth:`keras_loss`, :meth:`torch_loss`),
- a validation/test metric with a reference numpy implementation (:meth:`numpy_metric`),
- the direction in which the metric is better (``"min"`` or ``"max"``).

The total training objective is ``L_task + sum_k weight_k * L_k`` plus whatever
cost regulariser the QAT backend adds (e.g. HGQ's ``beta * EBOPs``).
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from numpy.typing import ArrayLike

_REGISTRY: dict[str, type[TrustObjective]] = {}


class TrustObjective:
    #: registry name, e.g. ``"soft_ece"``
    name: ClassVar[str]
    #: name of the logged metric (Keras logs it as ``metric_name`` and ``val_<metric_name>``)
    metric_name: ClassVar[str]
    direction: ClassVar[Literal['min', 'max']]

    def __init__(self, **params: Any):
        self.params = params

    def numpy_metric(self, logits: ArrayLike, labels: ArrayLike) -> float:
        raise NotImplementedError

    def keras_loss(self):
        raise NotImplementedError(f'{self.name} has no Keras implementation')

    def keras_metric(self):
        raise NotImplementedError(f'{self.name} has no Keras metric')

    def torch_loss(self):
        raise NotImplementedError(f'{self.name} has no PyTorch implementation')

    def get_config(self) -> dict[str, Any]:
        return {'name': self.name, **self.params}

    def __repr__(self) -> str:
        args = ', '.join(f'{k}={v!r}' for k, v in self.params.items())
        return f'{type(self).__name__}({args})'


def register_objective(cls: type[TrustObjective]) -> type[TrustObjective]:
    if not getattr(cls, 'name', None):
        raise ValueError('objective class needs a non-empty `name`')
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        raise ValueError(f'objective {cls.name!r} already registered')
    _REGISTRY[cls.name] = cls
    return cls


def get_objective(name: str, **params: Any) -> TrustObjective:
    if name not in _REGISTRY:
        raise KeyError(f'unknown objective {name!r}; available: {sorted(_REGISTRY)}')
    return _REGISTRY[name](**params)


def list_objectives() -> list[str]:
    return sorted(_REGISTRY)


def objectives_from_config(entries: Any) -> list[tuple[TrustObjective, float]]:
    """Parse ``[{"name": "soft_ece", "weight": 2.0, "n_bins": 15}, ...]`` into (objective, weight)."""
    out = []
    for entry in entries or []:
        entry = dict(entry)
        name = entry.pop('name')
        weight = float(entry.pop('weight', 1.0))
        if weight < 0:
            raise ValueError(f'objective {name!r}: weight must be non-negative')
        out.append((get_objective(name, **entry), weight))
    return out
