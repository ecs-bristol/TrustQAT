import numpy as np
import pytest

from benchmarks.common.driver import choose_final, resolve_selection


def test_resolve_selection_accuracy_tolerance():
    sel = {
        'objectives': {'val_accuracy': 'max', 'val_ece': 'min'},
        'accuracy_tolerance': 0.005,
        'final': {'min': 'val_ece'},
    }
    out = resolve_selection(sel, init_val_accuracy=0.73)
    assert out['constraints']['val_accuracy']['min'] == pytest.approx(0.725)
    assert 'accuracy_tolerance' not in out and out['final'] == {'min': 'val_ece'}
    assert 'accuracy_tolerance' in sel  # input untouched
    assert resolve_selection(None, None) is None
    with pytest.raises(ValueError):
        resolve_selection(sel, None)


def test_choose_final():
    front = [
        {'epoch': 3, 'metrics': {'val_ece': 0.02, 'val_accuracy': 0.73}, 'cost': {'ebops': 500}},
        {'epoch': 1, 'metrics': {'val_ece': 0.02, 'val_accuracy': 0.72}, 'cost': {'ebops': 400}},
        {'epoch': 7, 'metrics': {'val_ece': 0.05, 'val_accuracy': 0.74}, 'cost': {'ebops': 600}},
    ]
    assert choose_final(front, {'min': 'val_ece'})['epoch'] == 1  # tie -> earliest epoch
    assert choose_final(front, {'max': 'val_accuracy'})['epoch'] == 7
    assert choose_final(front, {'min': 'cost.ebops'})['epoch'] == 1
    assert choose_final(front, None) is None
    with pytest.raises(ValueError):
        choose_final(front, {'best': 'val_ece'})


def test_jsc_splits_are_deterministic():
    pytest.importorskip('sklearn')
    from benchmarks.jsc_hlf.data import make_splits

    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(1000, 16)), rng.integers(0, 5, size=1000)
    a, b = make_splits(x, y), make_splits(x, y)
    assert {k: v.shape for k, v in a.items()} == {
        'x_train': (720, 16),
        'y_train': (720,),
        'x_val': (80, 16),
        'y_val': (80,),
        'x_test': (200, 16),
        'y_test': (200,),
    }
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert a['x_train'].dtype == np.float32 and a['y_train'].dtype == np.int32
    x_tv = np.concatenate([a['x_train'], a['x_val']])
    assert np.allclose(x_tv.mean(0), 0, atol=1e-5) and np.allclose(x_tv.std(0), 1, atol=1e-4)
