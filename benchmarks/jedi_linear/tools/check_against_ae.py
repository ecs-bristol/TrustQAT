"""Re-evaluate the TrustQAT-AE checkpoints with this repository and compare with the artifact.

For every row of the artifact's ``ece_hardware_results.csv`` this loads

- the selected TrustQAT checkpoint (``checkpoint`` column, relative to the artifact root), and
- the matched HGQ baseline (``train.init_weights`` of the corresponding config here),

evaluates both with :func:`benchmarks.jedi_linear.run` conventions (``trace_minmax`` on the
train/val set, test-set accuracy and 15-bin ECE) and reports the differences to the
artifact's ``ece_accuracy`` / ``ece_ece`` / ``hgq_accuracy`` / ``hgq_ece``.

    TRUSTQAT_JEDI_DATA=... TRUSTQAT_JEDI_MODELS=... \
    python -m benchmarks.jedi_linear.tools.check_against_ae --ae-root /path/to/TrustQAT-AE -o runs/ae_check.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('KERAS_BACKEND', 'jax')

import numpy as np  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parents[1] / 'configs' / 'trustqat'


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ae-root', required=True, type=Path)
    p.add_argument(
        '--results', type=Path, default=None, help='default: <ae-root>/hardware_ece_da4ml/ece_hardware_results.csv'
    )
    p.add_argument('-o', '--output', type=Path, default=Path('runs/ae_check.csv'))
    p.add_argument('--only', nargs='*', default=None, help='config stems, e.g. perminv-n16-f3')
    args = p.parse_args(argv)

    from benchmarks.common.driver import load_config, seed_all
    from benchmarks.common.driver import plain as _plain
    from trustqat.backends import get_backend
    from trustqat.core import calibration_report
    from trustqat.models import get_model_factory

    from .. import models  # noqa: F401
    from ..data import get_data

    results = args.results or args.ae_root / 'hardware_ece_da4ml' / 'ece_hardware_results.csv'
    with results.open() as f:
        rows = list(csv.DictReader(f))
    groups: dict[tuple[int, bool], list[dict]] = defaultdict(list)
    for row in rows:
        fam = 'perminv' if row['model'] == 'Permutation-Invariant' else 'nonperminv'
        row['stem'] = f'{fam}-n{row["particles"]}-f{row["features"]}'
        if args.only and row['stem'] not in args.only:
            continue
        groups[(int(row['particles']), row['features'] == '3')].append(row)

    backend = get_backend('hgq2')
    out_rows = []
    for (n, pt_eta_phi), grp in sorted(groups.items()):
        cfg0 = load_config(str(CONFIG_DIR / f'{grp[0]["stem"]}.yaml'))
        seed_all(int(cfg0.seed))
        x_train_val, x_test, _, y_test = get_data(cfg0.data.path, n, pt_eta_phi)
        x_calib = np.asarray(x_train_val, np.float32)
        for row in grp:
            cfg = load_config(str(CONFIG_DIR / f'{row["stem"]}.yaml'))
            factory = get_model_factory(cfg.model.name, backend.name)
            res = {'config': row['stem']}
            for tag, path in (('trust', args.ae_root / row['checkpoint']), ('hgq', Path(cfg.train.init_weights))):
                model = backend.load_weights(factory(_plain(cfg)), path)
                model = backend.prepare_inference(model, x_calib, batch_size=16384)
                rep = calibration_report(
                    model.predict(x_test, batch_size=16384, verbose=0), y_test, 15, with_bins=False
                )
                ae_acc = float(row['ece_accuracy' if tag == 'trust' else 'hgq_accuracy'])
                ae_ece = float(row['ece_ece' if tag == 'trust' else 'hgq_ece'])
                res.update(
                    {
                        f'{tag}_acc': rep['acc'],
                        f'{tag}_ece': rep['ece'],
                        f'{tag}_ebops': backend.cost(model)['ebops'],
                        f'{tag}_acc_ae': ae_acc,
                        f'{tag}_ece_ae': ae_ece,
                        f'{tag}_dacc': rep['acc'] - ae_acc,
                        f'{tag}_dece': rep['ece'] - ae_ece,
                    }
                )
                del model
                gc.collect()
            out_rows.append(res)
            print(
                f'{row["stem"]:22s} trust acc {res["trust_acc"]:.4%} (AE {res["trust_acc_ae"]:.4%})'
                f' ece {res["trust_ece"]:.5f} (AE {res["trust_ece_ae"]:.5f}) | hgq acc {res["hgq_acc"]:.4%}'
                f' (AE {res["hgq_acc_ae"]:.4%}) ece {res["hgq_ece"]:.5f} (AE {res["hgq_ece_ae"]:.5f})',
                flush=True,
            )
        del x_train_val, x_test, x_calib
        gc.collect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    worst = {k: max(abs(r[k]) for r in out_rows) for k in ('trust_dacc', 'trust_dece', 'hgq_dacc', 'hgq_dece')}
    print('max |difference| vs artifact:', {k: f'{v:.2e}' for k, v in worst.items()}, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
