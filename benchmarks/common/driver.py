"""Benchmark-independent phases.

A benchmark supplies three callables and gets the command line for free::

    main(argv, load_data=..., build_model=..., load_checkpoint=...)

``load_data(cfg) -> Data``
    the data, see :class:`Data`.
``build_model(cfg, backend) -> model``
    the model to train, including its initial weights.
``load_checkpoint(cfg, backend, path) -> model``
    a model restored from a training checkpoint.

Phases write into ``cfg.save_path``:

``train``   ``records.jsonl`` (every epoch), ``ckpts/`` (current Pareto front), ``pareto.json``,
            ``selection.json``, ``history.json``, ``last.<suffix>``, ``config.yaml``, ``init_metrics.json``
``test``    ``models/`` (inference-ready copies), ``test_results.json`` / ``.csv``
``export``  ``<target>/<checkpoint>/`` hardware projects, ``export_results.json``
``select``  ``selected.json``: validation Pareto front joined with test results, plus the
            single ``chosen`` model when ``selection.final`` is set
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault('KERAS_BACKEND', 'jax')

import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

PHASES = ('train', 'test', 'export', 'select')


@dataclass
class Data:
    """``x_calib`` calibrates inference ranges (train+val); ``splits()`` gives the training split."""

    x_test: np.ndarray
    y_test: np.ndarray
    x_calib: np.ndarray
    splits: Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]


def load_config(path: str, overrides: list[str] | None = None):
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def plain(cfg) -> Any:
    return OmegaConf.to_container(cfg, resolve=True)


def seed_all(seed: int, keras_seed: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if keras_seed:
        import keras

        keras.utils.set_random_seed(seed)


def accuracy(model, x, y, batch_size: int) -> float:
    pred = model.predict(x, batch_size=batch_size, verbose=0)
    return float(np.mean(np.argmax(pred, axis=1) == np.asarray(y).ravel()))


def resolve_selection(selection: dict[str, Any] | None, init_val_accuracy: float | None) -> dict[str, Any] | None:
    """Turn ``accuracy_tolerance`` into a ``val_accuracy >= init - tolerance`` constraint.

    The minimum validation accuracy is then derived from the starting (baseline) model,
    as in the paper. Returns a new dict; ``None`` stays ``None``.
    """
    if selection is None:
        return None
    sel = json.loads(json.dumps(selection))
    tol = sel.pop('accuracy_tolerance', None)
    if tol is not None:
        if init_val_accuracy is None:
            raise ValueError('accuracy_tolerance needs the initial validation accuracy')
        floor = float(init_val_accuracy) - float(tol)
        sel.setdefault('constraints', {}).setdefault('val_accuracy', {})['min'] = floor
        sel['accuracy_floor'] = {'init_val_accuracy': float(init_val_accuracy), 'tolerance': float(tol), 'min': floor}
    return sel


def choose_final(front: list[dict[str, Any]], final: dict[str, str] | None) -> dict[str, Any] | None:
    """Pick one record of the front, e.g. ``{"min": "val_ece"}`` (ties: lowest epoch)."""
    if not final or not front:
        return None
    ((op, key),) = final.items()
    if op not in ('min', 'max'):
        raise ValueError(f'selection.final must be {{min|max: key}}, got {final}')

    def value(r):
        if key.startswith('cost.'):
            return r['cost'][key[5:]]
        return r['metrics'][key]

    sign = 1.0 if op == 'min' else -1.0
    return min(front, key=lambda r: (sign * value(r), r['epoch']))


# ---------------------------------------------------------------------------- phases
def phase_train(cfg, backend, model, data: Data) -> None:
    from trustqat.keras import KerasTrainer

    save_path = Path(cfg.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, save_path / 'config.yaml', resolve=True)
    bs = int(cfg.get('eval', {}).get('batch_size', 16384))

    x_tr, y_tr, x_val, y_val = data.splits()
    init = {'test_accuracy': accuracy(model, data.x_test, data.y_test, bs)}
    full = plain(cfg)
    if (full.get('selection') or {}).get('accuracy_tolerance') is not None:
        init['val_accuracy'] = accuracy(model, x_val, y_val, bs)
    full['selection'] = resolve_selection(full.get('selection'), init.get('val_accuracy'))
    print(f'initial model: {init}', flush=True)
    (save_path / 'init_metrics.json').write_text(json.dumps(init, indent=2) + '\n')

    KerasTrainer(backend, full).fit(model, x_tr, y_tr, x_val, y_val, save_path)


def _checkpoints(save_path: Path, suffix: str) -> list[Path]:
    return sorted((save_path / 'ckpts').glob(f'*{suffix}'))


def evaluate(cfg, backend, model, data: Data) -> tuple[dict[str, Any], Any]:
    """Test-set report; with ``eval.calibrate`` (default) after ``backend.prepare_inference``."""
    from trustqat.core import calibration_report

    ev = cfg.get('eval', {})
    n_bins, bs = int(ev.get('n_bins', 15)), int(ev.get('batch_size', 16384))
    row: dict[str, Any] = {}
    if ev.get('report_uncalibrated', False):
        raw = calibration_report(
            model.predict(data.x_test, batch_size=bs, verbose=0), data.y_test, n_bins, with_bins=False
        )
        row.update({f'test_raw_{k}': v for k, v in raw.items() if k != 'n_bins'})
    if ev.get('calibrate', True):
        model = backend.prepare_inference(model, np.asarray(data.x_calib, np.float32), batch_size=bs)
    rep = calibration_report(model.predict(data.x_test, batch_size=bs, verbose=0), data.y_test, n_bins)
    row.update({f'test_{k}': v for k, v in rep.items() if k != 'bins'})
    row.update(backend.cost(model))
    row['bins'] = rep['bins']
    return row, model


def phase_test(cfg, backend, load_checkpoint, data: Data) -> list[dict]:
    from trustqat.core import read_records

    save_path = Path(cfg.save_path)
    records = {}
    if (save_path / 'records.jsonl').exists():
        records = {r.checkpoint: r for r in read_records(save_path / 'records.jsonl') if r.checkpoint}
    (save_path / 'models').mkdir(exist_ok=True)

    rows = []
    for ckpt in _checkpoints(save_path, backend.checkpoint_suffix):
        model = load_checkpoint(cfg, backend, ckpt)
        res, model = evaluate(cfg, backend, model, data)
        backend.save(model, save_path / 'models' / ckpt.name)
        rec = records.get(str(ckpt.relative_to(save_path)))
        row = {
            'checkpoint': ckpt.name,
            'epoch': rec.epoch if rec else None,
            'val_accuracy': rec.metrics.get('val_accuracy') if rec else None,
            'val_ece': rec.metrics.get('val_ece') if rec else None,
            **res,
        }
        rows.append(row)
        cost = ' '.join(f'{k}={row[k]:.0f}' for k in backend.cost_names)
        print(f'{ckpt.name}: acc={row["test_acc"]:.4%} ece={row["test_ece"]:.5f} {cost}', flush=True)
        del model
        gc.collect()

    with (save_path / 'test_results.json').open('w') as f:
        json.dump(rows, f, indent=2)
    fields = [k for k in (rows[0] if rows else {}) if k != 'bins']
    with (save_path / 'test_results.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    return rows


def phase_export(cfg, backend, data: Data, rtl_samples: int = 0, only: list[str] | None = None) -> list[dict]:
    from trustqat.backends import export_capability, require_capability

    exp = dict(plain(cfg).get('export') or {})
    target = exp.pop('target', 'da4ml')
    require_capability(backend, export_capability(target))
    save_path = Path(cfg.save_path)
    models = sorted((save_path / 'models').glob(f'*{backend.checkpoint_suffix}'))
    if only:
        models = [m for m in models if m.name in set(only) or m.stem in set(only)]
    if not models:
        raise FileNotFoundError(f'no inference-ready models in {save_path / "models"}; run the test phase first')

    results = []
    for path in models:
        model = backend.load(path)
        out_dir = save_path / target / path.stem
        row: dict = {'model': path.name, 'project': str(out_dir)}
        if target == 'da4ml' and hasattr(backend, 'build_rtl'):
            rtl = backend.build_rtl(model, out_dir, compile=rtl_samples > 0, **exp)
            pipe = getattr(rtl, '_pipe', None)
            if pipe is not None:
                row['da_est_lut'] = round(float(getattr(pipe, 'cost', float('nan'))))
                row['da_est_ff'] = round(float(getattr(pipe, 'reg_bits', float('nan'))))
            if rtl_samples > 0:
                n = min(rtl_samples, len(data.x_test))
                x = np.ascontiguousarray(data.x_test[:n]).astype(np.float32)
                pred_rtl = rtl.predict(x)
                pred_keras = model.predict(x, batch_size=16384, verbose=0)
                row['rtl_samples'] = n
                row['rtl_acc'] = float(np.mean(np.argmax(pred_rtl, axis=1) == np.asarray(data.y_test[:n]).ravel()))
                row['rtl_mismatch'] = int(np.sum(np.any(pred_rtl != pred_keras, axis=1)))
        else:
            backend.export(model, target, out_dir, **exp)
        results.append(row)
        print(json.dumps(row), flush=True)
    with (save_path / 'export_results.json').open('w') as f:
        json.dump(results, f, indent=2)
    return results


def phase_select(cfg) -> dict:
    from trustqat.core import read_records, select_records

    save_path = Path(cfg.save_path)
    with (save_path / 'selection.json').open() as f:
        selection = json.load(f)
    records = read_records(save_path / 'records.jsonl')
    front = select_records(records, selection['objectives'], selection.get('constraints'), require_checkpoint=True)
    tests = {}
    if (save_path / 'test_results.json').exists():
        with (save_path / 'test_results.json').open() as f:
            tests = {r['checkpoint']: r for r in json.load(f)}
    out = []
    for r in sorted(front, key=lambda r: r.epoch):
        name = Path(r.checkpoint).name if r.checkpoint else None
        t = {k: v for k, v in tests.get(name, {}).items() if k != 'bins'}
        out.append({**r.to_dict(), 'test': t})
    chosen = choose_final(out, selection.get('final'))
    result = {**selection, 'front': out, 'chosen': chosen}
    with (save_path / 'selected.json').open('w') as f:
        json.dump(result, f, indent=2)
    for row in out:
        t = row['test']
        mark = '*' if chosen is not None and row['epoch'] == chosen['epoch'] else ' '
        print(
            f'{mark} epoch {row["epoch"]:4d}  val_acc={row["metrics"].get("val_accuracy", float("nan")):.4%} '
            f'val_ece={row["metrics"].get("val_ece", float("nan")):.5f} cost={row["cost"]}  '
            + (f'test_acc={t["test_acc"]:.4%} test_ece={t["test_ece"]:.5f}' if t else '(no test results)'),
            flush=True,
        )
    return result


# ---------------------------------------------------------------------------- CLI
def main(
    argv: list[str] | None,
    *,
    load_data: Callable[[Any], Data],
    build_model: Callable[[Any, Any], Any],
    load_checkpoint: Callable[[Any, Any, Path], Any],
    description: str | None = None,
) -> int:
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-c', '--config', required=True)
    p.add_argument('-r', '--run', nargs='+', default=['train', 'test'], choices=PHASES + ('all',))
    p.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='OmegaConf dot-list overrides')
    p.add_argument('--rtl-samples', type=int, default=0, help='export: simulate N test samples with Verilator')
    p.add_argument('--only', nargs='*', default=None, help='export: only these model files')
    args = p.parse_args(argv)
    phases = PHASES if 'all' in args.run else tuple(args.run)

    cfg = load_config(args.config, args.set)
    if phases == ('select',):
        phase_select(cfg)
        return 0

    from trustqat.backends import get_backend

    backend = get_backend(cfg.backend.name)
    seed_all(int(cfg.seed), bool(cfg.get('seed_keras', False)))
    print('Loading data...', flush=True)
    data = load_data(cfg)

    if 'train' in phases:
        phase_train(cfg, backend, build_model(cfg, backend), data)
    if 'test' in phases:
        phase_test(cfg, backend, load_checkpoint, data)
    if 'export' in phases:
        phase_export(cfg, backend, data, rtl_samples=args.rtl_samples, only=args.only)
    if 'select' in phases:
        phase_select(cfg)
    return 0
