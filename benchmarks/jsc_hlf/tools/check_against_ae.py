"""Re-evaluate the TrustQAT-AE HLF JSC results (paper Table I) with this repository.

For each of the five EBOP points this loads the HGQ checkpoint and the TrustQAT weights
(``trust_results/jsc_openml_ebops<P>/training/best.weights.h5``) from the artifact and
compares the test accuracy and 15-bin ECE

- as trained (no range calibration) with ``training/results.json``, the numbers in the
  paper's Table I, and
- after ``trace_minmax`` on train+val, as used for the RTL, with the artifact's
  ``hardware/<variant>/metadata.json``.

    python -m benchmarks.jsc_hlf.tools.check_against_ae --ae-root /path/to/hgq-ae \
        --data dataset/jsc_hlf_openml_splits.npz [--ae-data /path/to/openml_splits.npz]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('KERAS_BACKEND', 'jax')

import numpy as np  # noqa: E402

POINTS = (530, 1572, 2571, 7229, 13285)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ae-root', required=True, type=Path)
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--ae-data', type=Path, default=None, help='artifact splits (npz) to compare the data with')
    p.add_argument('-o', '--output', type=Path, default=Path('runs/jsc_ae_check.csv'))
    args = p.parse_args(argv)

    from trustqat.backends import get_backend
    from trustqat.core import calibration_report

    from ..data import load_splits
    from ..models import load_hgq_checkpoint

    d = load_splits(args.data)
    if args.ae_data:
        ae = load_splits(args.ae_data)
        for k in d:
            same = d[k].shape == ae[k].shape and np.array_equal(d[k], ae[k])
            diff = 0.0 if same or d[k].shape != ae[k].shape else float(np.max(np.abs(d[k] - ae[k])))
            print(f'data {k}: identical={same} max|diff|={diff:.3g}', flush=True)
    x_calib = np.concatenate([d['x_train'], d['x_val']])
    backend = get_backend('hgq2')

    rows = []
    for point in POINTS:
        root = args.ae_root / 'trust_results' / f'jsc_openml_ebops{point}'
        results = json.loads((root / 'training' / 'results.json').read_text())
        (baseline,) = (args.ae_root / 'models' / 'jsc_openml').glob(f'*-ebops={point}-*.keras')
        for variant, key in (('hgq', 'hgq'), ('trust', 'trust_hgq')):
            model = load_hgq_checkpoint(baseline)
            if variant == 'trust':
                model.load_weights(root / 'training' / 'best.weights.h5')
            raw = calibration_report(
                model.predict(d['x_test'], batch_size=33200, verbose=0), d['y_test'], 15, with_bins=False
            )
            backend.prepare_inference(model, x_calib, batch_size=33200)
            cal = calibration_report(
                model.predict(d['x_test'], batch_size=33200, verbose=0), d['y_test'], 15, with_bins=False
            )
            meta = json.loads((root / 'hardware' / key / 'metadata.json').read_text())
            row = {
                'ebops_point': point,
                'variant': variant,
                'raw_acc': raw['acc'],
                'raw_acc_ae': results[key]['test']['accuracy'],
                'raw_ece': raw['ece'],
                'raw_ece_ae': results[key]['test']['ece'],
                'cal_acc': cal['acc'],
                'cal_acc_ae': meta['keras_accuracy_full'],
                'cal_ece': cal['ece'],
                'cal_ece_ae': meta['keras_ece_15'],
                'ebops': backend.cost(model)['ebops'],
            }
            rows.append(row)
            print(
                f'{point:6d} {variant:5s} as trained: acc {row["raw_acc"]:.4%} (AE {row["raw_acc_ae"]:.4%}) '
                f'ece {row["raw_ece"]:.5f} (AE {row["raw_ece_ae"]:.5f}) | calibrated: acc {row["cal_acc"]:.4%} '
                f'(AE {row["cal_acc_ae"]:.4%}) ece {row["cal_ece"]:.5f} (AE {row["cal_ece_ae"]:.5f})',
                flush=True,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    worst = {k: max(abs(r[k] - r[f'{k}_ae']) for r in rows) for k in ('raw_acc', 'raw_ece', 'cal_acc', 'cal_ece')}
    print('max |difference| vs artifact:', {k: f'{v:.2e}' for k, v in worst.items()}, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
