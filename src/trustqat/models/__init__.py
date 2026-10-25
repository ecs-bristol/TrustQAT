"""Model factories, registered per (model name, backend).

The same benchmark model is written once per QAT backend (an HGQ2 model uses HGQ layers,
a Brevitas model is an ``nn.Module``), so factories live with the benchmark and are
registered here::

    @register_model('jedi_linear', backend='hgq2')
    def build(cfg): ...
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_FACTORIES: dict[tuple[str, str], Callable[[Any], Any]] = {}


def register_model(name: str, backend: str):
    def deco(fn: Callable[[Any], Any]):
        key = (name, backend)
        if key in _FACTORIES and _FACTORIES[key] is not fn:
            raise ValueError(f'model {name!r} already registered for backend {backend!r}')
        _FACTORIES[key] = fn
        return fn

    return deco


def get_model_factory(name: str, backend: str) -> Callable[[Any], Any]:
    try:
        return _FACTORIES[(name, backend)]
    except KeyError:
        have = sorted(_FACTORIES)
        raise KeyError(f'no model {name!r} for backend {backend!r}; registered: {have}') from None


def list_models() -> list[tuple[str, str]]:
    return sorted(_FACTORIES)


__all__ = ['get_model_factory', 'list_models', 'register_model']
