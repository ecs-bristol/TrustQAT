"""The QAT backend interface.

A backend adapts one QAT library (HGQ2, QKerasV3, Brevitas, ...) to TrustQAT. It does
**not** define models; models are registered per (benchmark, backend) in
:mod:`trustqat.models`.

Required members
----------------
``name``, ``framework`` (``"keras"`` or ``"torch"``), ``api_version``, ``capabilities``,
``cost_names``, ``checkpoint_suffix`` and the methods ``cost``, ``save``, ``load``.

Optional members
----------------
Declared through ``capabilities`` and only required when declared:

=========================  =====================================================
capability                 method(s)
=========================  =====================================================
``trainable_precision``    ``precision_variables(model)``: all gradient-learned
                           bit-width parameters, including currently frozen ones
``freeze_precision``       ``freeze_precision(model)``: stop their updates (called
                           before compiling/creating the optimizer)
``precision_summary``      ``precision_summary(model) -> PrecisionSummary``
``cost_regularizer``       ``training_hooks(cfg)`` adds the cost term/schedule
``export:<target>``        ``export(model, target, out_dir, **kwargs)``
=========================  =====================================================

Plain hooks with defaults in :class:`BaseBackend`: ``training_hooks``, ``make_dataset``,
``load_weights``, ``prepare_inference``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

API_VERSION = '0.1'

TRAINABLE_PRECISION = 'trainable_precision'
FREEZE_PRECISION = 'freeze_precision'
PRECISION_SUMMARY = 'precision_summary'
COST_REGULARIZER = 'cost_regularizer'


def export_capability(target: str) -> str:
    return f'export:{target}'


class CapabilityError(RuntimeError):
    pass


def require_capability(backend: Any, capability: str) -> None:
    if capability not in backend.capabilities:
        raise CapabilityError(
            f'backend {backend.name!r} does not provide {capability!r} (capabilities: {sorted(backend.capabilities)})'
        )


def api_compatible(version: str, current: str = API_VERSION) -> bool:
    """Same major version; while at 0.x, the same minor version too."""
    v, c = (tuple(int(p) for p in s.split('.')[:2]) for s in (version, current))
    return v == c if c[0] == 0 else v[0] == c[0]


@dataclass
class QuantizerStats:
    name: str
    kind: Literal['weight', 'activation', 'other']
    shape: tuple[int, ...]
    n_elements: int
    bits_min: float
    bits_mean: float
    bits_max: float
    n_zero: int  # elements with zero bits (pruned)
    histogram: dict[int, int] = field(default_factory=dict)  # bit-width -> count


@dataclass
class PrecisionSummary:
    """Backend-neutral view of a model's precision allocation.

    One entry per quantizer; a per-tensor quantizer is simply ``shape == ()``.
    """

    quantizers: list[QuantizerStats]

    def totals(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for kind in ('weight', 'activation'):
            qs = [q for q in self.quantizers if q.kind == kind]
            n = sum(q.n_elements for q in qs)
            out[kind] = {
                'n_quantizers': len(qs),
                'n_elements': n,
                'mean_bits': (sum(q.bits_mean * q.n_elements for q in qs) / n) if n else 0.0,
                'n_zero': sum(q.n_zero for q in qs),
            }
        return out

    def to_dict(self) -> dict[str, Any]:
        return {'quantizers': [asdict(q) for q in self.quantizers], 'totals': self.totals()}


@runtime_checkable
class QATBackend(Protocol):
    name: ClassVar[str]
    framework: ClassVar[Literal['keras', 'torch']]
    api_version: ClassVar[str]
    capabilities: ClassVar[frozenset[str]]
    cost_names: ClassVar[tuple[str, ...]]
    checkpoint_suffix: ClassVar[str]

    def cost(self, model: Any) -> dict[str, float]: ...

    def save(self, model: Any, path: Path) -> None: ...

    def load(self, path: Path, cfg: Any = None) -> Any: ...


class BaseBackend:
    """Convenience base class with defaults for the optional hooks."""

    name: ClassVar[str] = ''
    framework: ClassVar[Literal['keras', 'torch']]
    api_version: ClassVar[str] = API_VERSION
    capabilities: ClassVar[frozenset[str]] = frozenset()
    cost_names: ClassVar[tuple[str, ...]] = ()
    checkpoint_suffix: ClassVar[str] = ''

    # ---------------------------------------------------------------- required
    def cost(self, model: Any) -> dict[str, float]:
        raise NotImplementedError

    def save(self, model: Any, path: Path) -> None:
        raise NotImplementedError

    def load(self, path: Path, cfg: Any = None) -> Any:
        raise NotImplementedError

    # ---------------------------------------------------------------- hooks
    def training_hooks(self, cfg: Any = None) -> list:
        """Extra framework callbacks for training (e.g. a cost-regulariser schedule)."""
        return []

    def make_dataset(self, x, y, batch_size: int, shuffle: bool, drop_last: bool, cfg: Any = None):
        """Return a framework dataset, or ``None`` to let the trainer use its default."""
        return None

    def load_weights(self, model: Any, path: Path) -> Any:
        """Load weights into an existing model and return the model to use."""
        model.load_weights(path)
        return model

    def prepare_inference(self, model: Any, calibration_data: Any = None, **kwargs) -> Any:
        """Make a trained model ready for evaluation/export (e.g. calibrate ranges)."""
        return model

    # ---------------------------------------------------------------- capabilities
    def precision_variables(self, model: Any) -> list:
        raise CapabilityError(f'{self.name}: {TRAINABLE_PRECISION} not supported')

    def freeze_precision(self, model: Any) -> None:
        raise CapabilityError(f'{self.name}: {FREEZE_PRECISION} not supported')

    def precision_summary(self, model: Any) -> PrecisionSummary:
        raise CapabilityError(f'{self.name}: {PRECISION_SUMMARY} not supported')

    def export(self, model: Any, target: str, out_dir: Path, **kwargs) -> Path:
        raise CapabilityError(f'{self.name}: export to {target!r} not supported')

    def __repr__(self) -> str:
        return f'<{type(self).__name__} {self.name!r} ({self.framework}) caps={sorted(self.capabilities)}>'
