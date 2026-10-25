"""Backend discovery.

Backends are found through the ``trustqat.backends`` entry-point group (the bundled HGQ2
backend is registered the same way, in ``pyproject.toml``), or registered at runtime with
:func:`register_backend` (handy for tests and notebooks). Loading is lazy: importing this
module never imports a QAT library.
"""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points
from typing import Any

from .base import (
    API_VERSION,
    COST_REGULARIZER,
    FREEZE_PRECISION,
    PRECISION_SUMMARY,
    TRAINABLE_PRECISION,
    BaseBackend,
    CapabilityError,
    PrecisionSummary,
    QATBackend,
    QuantizerStats,
    api_compatible,
    export_capability,
    require_capability,
)

ENTRY_POINT_GROUP = 'trustqat.backends'

_RUNTIME: dict[str, Any] = {}


def register_backend(name: str, target: Any) -> None:
    """Register a backend class (or ``"module:Class"`` string) under ``name`` for this process."""
    _RUNTIME[name] = target


def available_backends() -> dict[str, str]:
    """Name -> import target, from entry points and runtime registrations."""
    found = {ep.name: ep.value for ep in entry_points(group=ENTRY_POINT_GROUP)}
    for name, target in _RUNTIME.items():
        found[name] = target if isinstance(target, str) else f'{target.__module__}:{target.__qualname__}'
    return dict(sorted(found.items()))


def _load_target(target: Any):
    if not isinstance(target, str):
        return target
    module, _, attr = target.partition(':')
    obj = importlib.import_module(module)
    for part in attr.split('.') if attr else []:
        obj = getattr(obj, part)
    return obj


def get_backend_class(name: str):
    if name in _RUNTIME:
        return _load_target(_RUNTIME[name])
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name == name:
            return ep.load()
    raise KeyError(f'unknown backend {name!r}; available: {list(available_backends())}')


def get_backend(name: str, **kwargs: Any):
    """Instantiate a backend by name and check that it speaks a compatible API version."""
    cls = get_backend_class(name)
    backend = cls(**kwargs)
    version = getattr(backend, 'api_version', None)
    if version is None or not api_compatible(version):
        raise RuntimeError(f'backend {name!r} targets TrustQAT backend API {version}, this is {API_VERSION}')
    if not isinstance(backend, QATBackend):
        raise TypeError(f'backend {name!r} does not implement the QATBackend protocol')
    return backend


__all__ = [
    'API_VERSION',
    'COST_REGULARIZER',
    'ENTRY_POINT_GROUP',
    'FREEZE_PRECISION',
    'PRECISION_SUMMARY',
    'TRAINABLE_PRECISION',
    'BaseBackend',
    'CapabilityError',
    'PrecisionSummary',
    'QATBackend',
    'QuantizerStats',
    'api_compatible',
    'available_backends',
    'export_capability',
    'get_backend',
    'get_backend_class',
    'register_backend',
    'require_capability',
]
