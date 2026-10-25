"""Backend conformance: two toy backends (proving the API is not HGQ-shaped) and HGQ2."""

import importlib.util

import pytest

from trustqat.testing import check_backend


def test_toy_keras_fixed_backend(toy_data, tmp_path):
    from toy_backends.keras_fixed import ToyKerasFixedBackend, tiny_model

    x, y = toy_data
    report = check_backend(ToyKerasFixedBackend(), tiny_model, x, y, tmp_path)
    assert report['trust_gradient_to_precision'] == 'skipped'
    assert report['trainer_records'] == 'ok'


@pytest.mark.skipif(importlib.util.find_spec('torch') is None, reason='torch not installed')
def test_toy_torch_backend(toy_data, tmp_path):
    from toy_backends.torch_fixed import TinyNet, ToyTorchBackend

    x, y = toy_data
    report = check_backend(ToyTorchBackend(), TinyNet, x, y, tmp_path)
    assert report['trust_gradient_to_precision'] == 'ok'
    assert report['freeze_precision'] == 'ok'


def hgq2_tiny_model(n_in: int = 8, n_classes: int = 3):
    """Small HGQ2 model with bit-width regularisers and EBOPs penalty switched off."""
    import keras
    from hgq.config import LayerConfigScope, QuantizerConfigScope
    from hgq.layers import QDense
    from hgq.regularizers import MonoL1

    zero = MonoL1(0.0)
    with (
        QuantizerConfigScope(default_q_type='kbi', overflow_mode='SAT_SYM', b0=6, br=zero, ir=zero),
        QuantizerConfigScope(default_q_type='kif', place='datalane', overflow_mode='SAT', i0=3, f0=6, fr=zero, ir=zero),
        LayerConfigScope(beta0=0.0),
    ):
        inp = keras.Input((n_in,))
        x = QDense(16, activation='relu')(inp)
        out = QDense(n_classes)(x)
    return keras.Model(inp, out)


@pytest.mark.hgq2
@pytest.mark.skipif(importlib.util.find_spec('hgq') is None, reason='HGQ2 not installed')
def test_hgq2_backend_conformance(toy_data, tmp_path):
    from trustqat.backends import get_backend

    x, y = toy_data
    backend = get_backend('hgq2')
    report = check_backend(backend, hgq2_tiny_model, x, y, tmp_path)
    assert report == {
        'interface': 'ok',
        'trust_gradient_to_params': 'ok',
        'trust_gradient_to_precision': 'ok',
        'cost': 'ok',
        'save_load': 'ok',
        'freeze_precision': 'ok',
        'precision_summary': 'ok',
        'trainer_records': 'ok',
    }


@pytest.mark.hgq2
@pytest.mark.skipif(importlib.util.find_spec('hgq') is None, reason='HGQ2 not installed')
def test_hgq2_precision_summary_and_cost(toy_data):
    from trustqat.backends import get_backend

    x, y = toy_data
    backend = get_backend('hgq2')
    model = hgq2_tiny_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.train_on_batch(x, y)
    s = backend.precision_summary(model)
    kinds = {q.kind for q in s.quantizers}
    assert 'weight' in kinds and 'activation' in kinds
    assert s.totals()['weight']['mean_bits'] > 0
    assert backend.cost(model)['ebops'] > 0
