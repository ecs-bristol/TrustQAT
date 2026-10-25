import json

from trustqat.cli import main
from trustqat.core import EpochRecord, write_record


def test_select_cli(tmp_path, capsys):
    p = tmp_path / 'records.jsonl'
    for e, (acc, ece, cost) in enumerate([(0.7, 0.02, 100), (0.72, 0.03, 100), (0.69, 0.01, 50)]):
        write_record(p, EpochRecord(e, {'val_accuracy': acc, 'val_ece': ece}, {'ebops': cost}, f'ckpts/{e}.keras'))
    (tmp_path / 'selection.json').write_text(
        json.dumps({'objectives': {'val_accuracy': 'max', 'cost.ebops': 'min'}, 'constraints': {}})
    )
    assert main(['select', str(p)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert [r['epoch'] for r in out['front']] == [1, 2]

    assert main(['select', str(p), '-o', 'val_ece:min', '-c', 'val_accuracy>=0.7']) == 0
    out = json.loads(capsys.readouterr().out)
    assert [r['epoch'] for r in out['front']] == [0]


def test_list_commands(capsys):
    assert main(['list-objectives']) == 0
    assert 'soft_ece' in capsys.readouterr().out
    assert main(['list-backends']) == 0
    assert 'hgq2' in capsys.readouterr().out
