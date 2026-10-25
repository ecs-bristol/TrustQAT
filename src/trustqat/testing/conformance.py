"""Backend conformance checks.

Third-party backends run :func:`check_backend` in their own CI::

    from trustqat.testing import check_backend
    report = check_backend(MyBackend(), make_tiny_model, x, y, tmp_path)

``model_factory`` should build a *small* classifier (logits output) for ``x``/``y``.
Avoid bit-width regularisers in it, so that the precision-gradient check isolates the
trust loss. Checks that need a capability are skipped when it is not declared.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ..backends.base import (
    FREEZE_PRECISION,
    PRECISION_SUMMARY,
    TRAINABLE_PRECISION,
    PrecisionSummary,
    QATBackend,
    api_compatible,
)
from ..core.records import EpochRecord, read_records, validate_record


class ConformanceError(AssertionError):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ConformanceError(msg)


# ---------------------------------------------------------------- framework shims
class _KerasOps:
    def __init__(self, backend):
        import keras

        self.keras = keras
        self.backend = backend

    def snapshot(self, variables) -> list[np.ndarray]:
        return [np.array(self.keras.ops.convert_to_numpy(v)) for v in variables]

    def model_vars(self, model):
        prec = {id(v) for v in self._precision(model)}
        return [v for v in model.trainable_variables if id(v) not in prec]

    def _precision(self, model):
        if TRAINABLE_PRECISION in self.backend.capabilities:
            return self.backend.precision_variables(model)
        return []

    def precision_vars(self, model):
        return self._precision(model)

    def train_steps(self, model, x, y, steps: int = 3, lr: float = 1e-2):
        from ..keras.objectives import SoftECE

        model.compile(optimizer=self.keras.optimizers.Adam(learning_rate=lr), loss=SoftECE(n_bins=15))
        for _ in range(steps):
            model.train_on_batch(x, y)

    def predict(self, model, x) -> np.ndarray:
        return np.asarray(model.predict(x, verbose=0, batch_size=len(x)))


class _TorchOps:
    def __init__(self, backend):
        import torch

        self.torch = torch
        self.backend = backend

    def snapshot(self, variables) -> list[np.ndarray]:
        return [v.detach().cpu().numpy().copy() for v in variables]

    def _precision(self, model):
        if TRAINABLE_PRECISION in self.backend.capabilities:
            return list(self.backend.precision_variables(model))
        return []

    def precision_vars(self, model):
        return self._precision(model)

    def model_vars(self, model):
        prec = {id(v) for v in self._precision(model)}
        return [p for p in model.parameters() if p.requires_grad and id(p) not in prec]

    def train_steps(self, model, x, y, steps: int = 3, lr: float = 1e-2):
        from ..torch.objectives import SoftECELoss

        torch = self.torch
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=lr)
        loss_fn = SoftECELoss(n_bins=15)
        xt, yt = torch.as_tensor(x), torch.as_tensor(y)
        model.train()
        for _ in range(steps):
            opt.zero_grad()
            loss = loss_fn(model(xt), yt)
            loss.backward()
            opt.step()

    def predict(self, model, x) -> np.ndarray:
        model.eval()
        with self.torch.no_grad():
            return model(self.torch.as_tensor(x)).cpu().numpy()


def _changed(before, after, atol=0.0) -> bool:
    return any(not np.allclose(b, a, atol=atol, rtol=0) for b, a in zip(before, after, strict=True))


# ---------------------------------------------------------------- checks
def check_backend(
    backend: Any,
    model_factory: Callable[[], Any],
    x: np.ndarray,
    y: np.ndarray,
    tmp_dir: str | Path,
    check_trainer: bool = True,
) -> dict[str, str]:
    """Run the conformance checks; returns ``{check: "ok" | "skipped"}`` or raises ``ConformanceError``."""
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, str] = {}

    # 1. static interface
    _check(isinstance(backend, QATBackend), 'backend does not implement the QATBackend protocol')
    _check(isinstance(backend.name, str) and backend.name, 'backend.name must be a non-empty string')
    _check(backend.framework in ('keras', 'torch'), f'unsupported framework {backend.framework!r}')
    _check(api_compatible(backend.api_version), f'incompatible api_version {backend.api_version!r}')
    _check(isinstance(backend.capabilities, frozenset), 'capabilities must be a frozenset')
    _check(all(isinstance(c, str) for c in backend.capabilities), 'capabilities must be strings')
    _check(len(backend.cost_names) > 0, 'cost_names must name at least one cost')
    report['interface'] = 'ok'

    fw = _KerasOps(backend) if backend.framework == 'keras' else _TorchOps(backend)

    # 2. the trust loss trains the model (and the precision, when trainable)
    model = model_factory()
    fw.predict(model, x[:2])  # build
    mv, pv = fw.model_vars(model), fw.precision_vars(model)
    _check(len(mv) > 0, 'model has no trainable parameters')
    m0, p0 = fw.snapshot(mv), fw.snapshot(pv)
    fw.train_steps(model, x, y)
    _check(_changed(m0, fw.snapshot(mv)), 'soft-ECE gradients did not update any model parameter')
    report['trust_gradient_to_params'] = 'ok'
    if TRAINABLE_PRECISION in backend.capabilities:
        _check(len(pv) > 0, 'trainable_precision declared but precision_variables() is empty')
        _check(_changed(p0, fw.snapshot(pv)), 'soft-ECE gradients did not update any precision parameter')
        report['trust_gradient_to_precision'] = 'ok'
    else:
        report['trust_gradient_to_precision'] = 'skipped'

    # 3. cost surrogate
    cost = backend.cost(model)
    _check(isinstance(cost, dict) and set(cost) == set(backend.cost_names), f'cost() keys {list(cost)} != cost_names')
    _check(all(math.isfinite(float(v)) and float(v) >= 0 for v in cost.values()), f'cost values must be >= 0: {cost}')
    report['cost'] = 'ok'

    # 4. save / load round trip
    path = tmp_dir / f'roundtrip{backend.checkpoint_suffix}'
    out0 = fw.predict(model, x)
    backend.save(model, path)
    loaded = backend.load(path)
    _check(np.allclose(out0, fw.predict(loaded, x), atol=1e-5), 'outputs differ after save/load')
    report['save_load'] = 'ok'

    # 5. freezing precision
    if FREEZE_PRECISION in backend.capabilities:
        model = model_factory()
        fw.predict(model, x[:2])
        backend.freeze_precision(model)
        pv_all = fw.precision_vars(model)
        _check(len(pv_all) > 0, 'precision_variables() must also list frozen bit-width parameters')
        mv = fw.model_vars(model)
        m0, p0 = fw.snapshot(mv), fw.snapshot(pv_all)
        fw.train_steps(model, x, y)
        _check(_changed(m0, fw.snapshot(mv)), 'after freeze_precision, model parameters stopped training')
        _check(not _changed(p0, fw.snapshot(pv_all)), 'freeze_precision did not stop precision updates')
        report['freeze_precision'] = 'ok'
    else:
        report['freeze_precision'] = 'skipped'

    # 6. precision summary
    if PRECISION_SUMMARY in backend.capabilities:
        summary = backend.precision_summary(model)
        _check(isinstance(summary, PrecisionSummary), 'precision_summary() must return a PrecisionSummary')
        _check(len(summary.quantizers) > 0, 'precision_summary() found no quantizers')
        _check(all(math.isfinite(q.bits_mean) for q in summary.quantizers), 'non-finite bit-widths in summary')
        report['precision_summary'] = 'ok'
    else:
        report['precision_summary'] = 'skipped'

    # 7. records / trainer integration
    if backend.framework == 'keras' and check_trainer:
        from ..keras.trainer import KerasTrainer

        run_dir = tmp_dir / 'trainer_run'
        cfg = {
            'train': {'batch_size': max(2, len(x) // 2), 'epochs': 2, 'drop_last': False},
            'trust': {'objectives': [{'name': 'soft_ece', 'weight': 1.0, 'n_bins': 15}]},
        }
        n_val = max(2, len(x) // 4)
        KerasTrainer(backend, cfg).fit(model_factory(), x[n_val:], y[n_val:], x[:n_val], y[:n_val], run_dir)
        records = read_records(run_dir / 'records.jsonl')
        _check(len(records) == 2, f'expected 2 records, got {len(records)}')
        for r in records:
            _check(set(backend.cost_names) <= set(r.cost), f'record misses costs: {r.cost}')
            _check('val_ece' in r.metrics and 'val_accuracy' in r.metrics, f'record misses metrics: {r.metrics}')
        import json

        front = json.loads((run_dir / 'pareto.json').read_text())['members']
        _check(len(front) >= 1, 'no checkpoint kept on the Pareto front')
        _check(all((run_dir / m['checkpoint']).exists() for m in front), 'Pareto-front checkpoint file missing')
        saved = {r.checkpoint for r in records if r.checkpoint}
        on_disk = {str(p.relative_to(run_dir)) for p in (run_dir / 'ckpts').iterdir()}
        _check(on_disk <= saved, f'unrecorded checkpoint files: {sorted(on_disk - saved)}')
        _check(on_disk == {m['checkpoint'] for m in front}, 'dominated checkpoints were not removed')
        report['trainer_records'] = 'ok'
    else:
        rec = EpochRecord(0, {'val_accuracy': 0.5, 'val_ece': 0.1}, {k: float(v) for k, v in cost.items()})
        validate_record(rec.to_dict())
        report['trainer_records'] = 'skipped' if backend.framework == 'keras' else 'ok (schema only)'
    return report
