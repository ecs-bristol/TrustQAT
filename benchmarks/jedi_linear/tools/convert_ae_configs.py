"""Convert TrustQAT-AE / JEDI-linear configs to the TrustQAT run-config schema.

    python benchmarks/jedi_linear/tools/convert_ae_configs.py /path/to/TrustQAT-AE/configs benchmarks/jedi_linear/configs

``sweep-*.yaml`` (HGQ baselines trained from scratch) go to ``hgq/``, ``ece-grid-*.yaml``
(the TrustQAT runs of the paper) go to ``trustqat/`` and ``sweep-*-ece.yaml`` to
``examples/``. Paths to the dataset and to the official HGQ checkpoints become
``${oc.env:TRUSTQAT_JEDI_DATA,dataset}`` and ``${oc.env:TRUSTQAT_JEDI_MODELS,official_models}``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml


def _fmt(v) -> str:
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, float):
        s = repr(v)
        return s if ('e' in s or '.' in s) else s + '.0'
    if isinstance(v, str) and v.startswith('${'):
        return f'"{v}"'
    return str(v)


def _float(v) -> float:
    return float(v)


def convert(src: Path) -> tuple[str, str]:
    c = yaml.safe_load(src.read_text())
    perminv = c['model_class'].endswith('uq1')
    fam = 'perminv' if perminv else 'nonperminv'
    n, f = int(c['n_constituents']), 3 if c['pt_eta_phi'] else 16
    stem = f'{fam}-n{n}-f{f}'
    name = src.stem
    if name.startswith('ece-grid-'):
        kind = 'trustqat'
    elif name.endswith('-ece'):
        kind, stem = 'examples', f'{stem}-ece-beta'
    else:
        kind = 'hgq'

    tr = c['train']
    trust = c.get('trust') or {}
    trust_on = bool(trust) and trust.get('enabled', True)
    cdr = tr['cdr_args']

    lines = [
        f'# Converted from TrustQAT-AE configs/{src.name}',
        'seed: 42',
        f'save_path: runs/jedi_linear/{kind}/{stem}',
        'data:',
        '  path: ${oc.env:TRUSTQAT_JEDI_DATA,dataset}',
        f'  n_constituents: {n}',
        f'  pt_eta_phi: {_fmt(bool(c["pt_eta_phi"]))}',
        f'  val_split: {_fmt(float(c.get("val_split", 0.1)))}',
        '  input_dtype: float16',
        'model:',
        '  name: jedi_linear',
        f'  model_class: {c["model_class"]}',
    ]
    for k in ('init_bw_a', 'init_bw_k'):
        lines.append(f'  {k}: {int(c["model"][k])}')
    for k in ('k_bw_l1_reg', 'a_bw_l1_reg'):
        lines.append(f'  {k}: {_float(c["model"][k]):.1e}')
    lines += ['backend:', '  name: hgq2', '  beta:', '    intervals:']
    for ep, beta, interp in c['beta']['intervals']:
        ep_s = '"${train.epochs}"' if isinstance(ep, str) else str(int(ep))
        lines.append(f'      - [{ep_s}, {_float(beta):.1e}, {interp}]')
    lines.append('train:')
    if tr.get('init_weights'):
        w = re.sub(r'^official_models', '${oc.env:TRUSTQAT_JEDI_MODELS,official_models}', tr['init_weights'])
        lines.append(f'  init_weights: "{w}"')
    lines += [
        f'  batch_size: {int(tr["bsz"])}',
        f'  epochs: {int(tr["epochs"])}',
        '  optimizer: {name: adam}',
        '  lr_schedule:',
        '    name: cosine_restarts',
        f'    initial_learning_rate: {_float(cdr["initial_learning_rate"]):.1e}',
        f'    first_decay_steps: {int(cdr["first_decay_steps"])}',
        f'    t_mul: {_float(cdr["t_mul"])}',
        f'    m_mul: {_float(cdr["m_mul"])}',
        f'    alpha: {_float(cdr["alpha"]):.1e}',
        f'    alpha_steps: {int(cdr["alpha_steps"])}',
    ]
    es = tr.get('early_stopping')
    if es and es.get('enabled'):
        lines += ['  early_stopping:', '    enabled: true']
        for k in ('monitor', 'mode', 'patience', 'min_delta', 'restore_best_weights'):
            if k in es:
                v = es[k]
                lines.append(f'    {k}: {_float(v):.1e}' if k == 'min_delta' else f'    {k}: {_fmt(v)}')
    lines.append('  freeze_precision: false')
    if trust_on:
        lines += [
            'trust:',
            '  objectives:',
            f'    - {{name: soft_ece, weight: {_float(trust.get("ece_weight", 1.0))}, n_bins: {int(trust.get("ece_bins", 15))}}}',
            'selection:',
            '  objectives: {val_accuracy: max, cost.ebops: min, val_ece: min}',
            f'  constraints: {{val_accuracy: {{gt: {_float(trust.get("pareto_min_accuracy", 0.5))}}}}}',
        ]
    else:
        lines += [
            'trust:',
            '  objectives: []',
            'selection:',
            '  objectives: {val_accuracy: max, cost.ebops: min}',
            '  constraints: {val_accuracy: {gt: 0.5}}',
        ]
    lines += [
        'eval:',
        '  n_bins: 15',
        '  batch_size: 16384',
        'export:',
        '  target: da4ml',
        '  prj_name: jet_classifier_large',
        '  part_name: xcvu13p-flga2577-2-e',
        '  clock_period: 2.0',
        '  latency_cutoff: 2',
    ]
    return f'{kind}/{stem}.yaml', '\n'.join(lines) + '\n'


def main(src_dir: str, dst_dir: str) -> None:
    dst = Path(dst_dir)
    for src in sorted(Path(src_dir).glob('*.yaml')):
        rel, text = convert(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f'{src.name} -> {out}')


if __name__ == '__main__':
    main(*sys.argv[1:3])
