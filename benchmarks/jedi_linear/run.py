"""JEDI-linear benchmark driver: train, test (accuracy + calibration), export (da4ml RTL), select.

Usage (from the repository root, ``KERAS_BACKEND=jax``)::

    python -m benchmarks.jedi_linear -c benchmarks/jedi_linear/configs/trustqat/perminv-n16-f3.yaml -r train test
    python -m benchmarks.jedi_linear -c CONFIG -r export --rtl-samples 10000
    python -m benchmarks.jedi_linear -c CONFIG -r select
    python -m benchmarks.jedi_linear -c CONFIG -r train --set train.epochs=5 save_path=runs/smoke

Phases write into ``save_path``:

``train``   ``records.jsonl`` (every epoch), ``ckpts/`` (current Pareto front), ``pareto.json``,
            ``selection.json``, ``history.json``, ``last.keras``, ``pretrain_acc.txt``
``test``    ``models/`` (inference-ready copies), ``test_results.json`` / ``.csv``
``export``  ``da4ml/<checkpoint>/`` Verilog projects, ``export_results.json``
``select``  ``selected.json``: validation Pareto front joined with test results
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
from pathlib import Path

os.environ.setdefault('KERAS_BACKEND', 'jax')

import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

PHASES = ('train', 'test', 'export', 'select')


def load_config(path: str, overrides: list[str] | None = None):
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _plain(cfg):
    return OmegaConf.to_container(cfg, resolve=True)


def phase_train(cfg, backend, factory, data) -> None:
    from trustqat.keras import KerasTrainer

    from .data import split_train_val

    x_train_val, x_test, y_train_val, y_test = data
    save_path = Path(cfg.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, save_path / 'config.yaml', resolve=True)

    model = factory(_plain(cfg))
    model.summary()
    seed_all(int(cfg.seed))
    init = cfg.train.get('init_weights')
    if init:
        print(f'Loading initial weights from {init}', flush=True)
        model = backend.load_weights(model, Path(init))
    pred = model.predict(x_test, batch_size=2048, verbose=0)
    acc0 = float(np.mean(np.argmax(pred, axis=1) == np.asarray(y_test).ravel()))
    print(f'pre-training test accuracy: {acc0:.2%}', flush=True)
    (save_path / 'pretrain_acc.txt').write_text(f'pre-training test accuracy: {acc0:.2%}\n')

    x_tr, y_tr, x_val, y_val = split_train_val(
        x_train_val, y_train_val, float(cfg.data.get('val_split', 0.1)), cfg.data.get('input_dtype', 'float16')
    )
    trainer = KerasTrainer(backend, _plain(cfg))
    trainer.fit(model, x_tr, y_tr, x_val, y_val, save_path)


def _checkpoints(save_path: Path, suffix: str) -> list[Path]:
    return sorted((save_path / 'ckpts').glob(f'*{suffix}'))


def phase_test(cfg, backend, factory, data) -> list[dict]:
    from trustqat.core import calibration_report, read_records

    x_train_val, x_test, _, y_test = data
    save_path = Path(cfg.save_path)
    n_bins = int(cfg.get('eval', {}).get('n_bins', 15))
    bs = int(cfg.get('eval', {}).get('batch_size', 16384))
    records = {}
    if (save_path / 'records.jsonl').exists():
        records = {r.checkpoint: r for r in read_records(save_path / 'records.jsonl') if r.checkpoint}
    (save_path / 'models').mkdir(exist_ok=True)
    x_calib = np.asarray(x_train_val, np.float32)

    rows = []
    for ckpt in _checkpoints(save_path, backend.checkpoint_suffix):
        model = backend.load_weights(factory(_plain(cfg)), ckpt)
        model = backend.prepare_inference(model, x_calib, batch_size=bs)
        logits = model.predict(x_test, batch_size=bs, verbose=0)
        rep = calibration_report(logits, y_test, n_bins=n_bins)
        cost = backend.cost(model)
        backend.save(model, save_path / 'models' / ckpt.name)
        rec = records.get(str(ckpt.relative_to(save_path)))
        row = {
            'checkpoint': ckpt.name,
            'epoch': rec.epoch if rec else None,
            **{f'test_{k}': v for k, v in rep.items() if k != 'bins'},
            **cost,
            'val_accuracy': rec.metrics.get('val_accuracy') if rec else None,
            'val_ece': rec.metrics.get('val_ece') if rec else None,
            'bins': rep['bins'],
        }
        rows.append(row)
        print(
            f'{ckpt.name}: acc={rep["acc"]:.4%} ece={rep["ece"]:.5f} '
            + ' '.join(f'{k}={v:.0f}' for k, v in cost.items()),
            flush=True,
        )
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


def phase_export(cfg, backend, data, rtl_samples: int = 0, only: list[str] | None = None) -> list[dict]:
    from trustqat.backends import export_capability, require_capability

    exp = dict(_plain(cfg).get('export') or {})
    target = exp.pop('target', 'da4ml')
    require_capability(backend, export_capability(target))
    _, x_test, _, y_test = data
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
                n = min(rtl_samples, len(x_test))
                x = np.ascontiguousarray(x_test[:n]).astype(np.float32)
                pred_rtl = rtl.predict(x)
                pred_keras = model.predict(x, batch_size=16384, verbose=0)
                row['rtl_samples'] = n
                row['rtl_acc'] = float(np.mean(np.argmax(pred_rtl, axis=1) == np.asarray(y_test[:n]).ravel()))
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
    result = {**selection, 'front': out}
    with (save_path / 'selected.json').open('w') as f:
        json.dump(result, f, indent=2)
    for row in out:
        t = row['test']
        print(
            f'epoch {row["epoch"]:4d}  val_acc={row["metrics"].get("val_accuracy", float("nan")):.4%} '
            f'val_ece={row["metrics"].get("val_ece", float("nan")):.5f} cost={row["cost"]}  '
            + (f'test_acc={t["test_acc"]:.4%} test_ece={t["test_ece"]:.5f}' if t else '(no test results)'),
            flush=True,
        )
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    from trustqat.models import get_model_factory

    from . import models  # noqa: F401  (registers the JEDI-linear factory)
    from .data import get_data

    backend = get_backend(cfg.backend.name)
    factory = get_model_factory(cfg.model.name, backend.name)

    seed_all(int(cfg.seed))
    print('Loading data...', flush=True)
    data = get_data(cfg.data.path, int(cfg.data.n_constituents), bool(cfg.data.pt_eta_phi))

    if 'train' in phases:
        phase_train(cfg, backend, factory, data)
    if 'test' in phases:
        phase_test(cfg, backend, factory, data)
    if 'export' in phases:
        phase_export(cfg, backend, data, rtl_samples=args.rtl_samples, only=args.only)
    if 'select' in phases:
        phase_select(cfg)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
