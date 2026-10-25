"""``trustqat`` command line: list backends/objectives and select checkpoints from records."""

from __future__ import annotations

import argparse
import json
import sys


def _cmd_list_backends(args) -> int:
    from .backends import available_backends, get_backend_class

    for name, target in available_backends().items():
        try:
            cls = get_backend_class(name)
            caps = ', '.join(sorted(cls.capabilities))
            print(f'{name:12s} {cls.framework:6s} api={cls.api_version}  [{caps}]  ({target})')
        except Exception as e:  # noqa: BLE001
            print(f'{name:12s} (not importable: {type(e).__name__}: {e})  ({target})')
    return 0


def _cmd_list_objectives(args) -> int:
    from .objectives import get_objective, list_objectives

    for name in list_objectives():
        obj = get_objective(name)
        print(f'{name:12s} metric={obj.metric_name} ({obj.direction})  defaults={obj.params}')
    return 0


def _parse_objective(spec: str) -> tuple[str, str]:
    key, _, direction = spec.rpartition(':')
    if direction not in ('max', 'min') or not key:
        raise argparse.ArgumentTypeError(f'expected KEY:max or KEY:min, got {spec!r}')
    return key, direction


def _parse_constraint(spec: str) -> tuple[str, str, float]:
    # KEY>=VALUE or KEY<=VALUE
    for op, name in (('>=', 'min'), ('<=', 'max')):
        if op in spec:
            key, value = spec.split(op, 1)
            return key, name, float(value)
    raise argparse.ArgumentTypeError(f'expected KEY>=VALUE or KEY<=VALUE, got {spec!r}')


def _cmd_select(args) -> int:
    from .core.pareto import select_records
    from .core.records import read_records

    records = read_records(args.records)
    objectives = dict(args.objective) if args.objective else None
    if objectives is None:
        from pathlib import Path

        with Path(args.records).with_name('selection.json').open() as f:
            sel = json.load(f)
        objectives, constraints = sel['objectives'], sel.get('constraints') or {}
    else:
        constraints = {}
    for key, op, value in args.constraint or []:
        constraints.setdefault(key, {})[op] = value
    front = select_records(records, objectives, constraints, require_checkpoint=args.with_checkpoint)
    out = [r.to_dict() for r in sorted(front, key=lambda r: r.epoch)]
    json.dump({'objectives': objectives, 'constraints': constraints, 'front': out}, sys.stdout, indent=2)
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog='trustqat', description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('list-backends', help='installed QAT backends and their capabilities').set_defaults(
        fn=_cmd_list_backends
    )
    sub.add_parser('list-objectives', help='registered trustworthiness objectives').set_defaults(
        fn=_cmd_list_objectives
    )
    s = sub.add_parser('select', help='Pareto-select epochs from a records.jsonl file')
    s.add_argument('records')
    s.add_argument('-o', '--objective', action='append', type=_parse_objective, help='KEY:max|min (repeatable)')
    s.add_argument('-c', '--constraint', action='append', type=_parse_constraint, help='KEY>=V or KEY<=V')
    s.add_argument('--with-checkpoint', action='store_true', help='only epochs whose checkpoint was kept')
    s.set_defaults(fn=_cmd_select)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == '__main__':
    raise SystemExit(main())
