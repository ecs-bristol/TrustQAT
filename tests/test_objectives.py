import importlib.util

import numpy as np
import pytest

from trustqat.core import expected_calibration_error
from trustqat.objectives import get_objective, list_objectives, objectives_from_config


def _soft_ece_numpy(logits, labels, n_bins=15):
    """Forward value of the minibatch surrogate == ECE of the batch."""
    return expected_calibration_error(logits, labels, n_bins)


def test_registry():
    assert 'soft_ece' in list_objectives()
    obj = get_objective('soft_ece', n_bins=10)
    assert obj.metric_name == 'ece' and obj.direction == 'min' and obj.n_bins == 10
    [(o, w)] = objectives_from_config([{'name': 'soft_ece', 'weight': 2.0, 'n_bins': 15}])
    assert w == 2.0 and o.n_bins == 15
    with pytest.raises(KeyError):
        get_objective('nope')
    with pytest.raises(ValueError):
        objectives_from_config([{'name': 'soft_ece', 'weight': -1}])


# ----------------------------------------------------------------------------- keras
keras = pytest.importorskip('keras')


def test_keras_soft_ece_forward_equals_batch_ece(logits_labels):
    from trustqat.keras import SoftECE

    logits, labels = logits_labels
    v = float(keras.ops.convert_to_numpy(SoftECE(n_bins=15)(labels, logits.astype('float32'))))
    assert v == pytest.approx(_soft_ece_numpy(logits, labels), abs=2e-5)


def test_keras_ece_metric_streams(logits_labels):
    from trustqat.keras import ECE

    logits, labels = logits_labels
    m = ECE(n_bins=15)
    for i in range(0, len(labels), 128):
        m.update_state(labels[i : i + 128], logits[i : i + 128].astype('float32'))
    assert float(m.result()) == pytest.approx(expected_calibration_error(logits, labels, 15), abs=2e-5)
    m.reset_state()
    assert float(m.result()) == 0.0


def test_keras_ece_sample_weight_consistent(logits_labels):
    """Integer sample weights must equal repeating samples."""
    from trustqat.keras import ECE

    logits, labels = logits_labels
    w = np.random.default_rng(0).integers(1, 3, size=len(labels))
    m = ECE(n_bins=15)
    m.update_state(labels, logits.astype('float32'), sample_weight=w.astype('float32'))
    rep = expected_calibration_error(np.repeat(logits, w, axis=0), np.repeat(labels, w), 15)
    assert float(m.result()) == pytest.approx(rep, abs=2e-5)


def test_trust_aware_loss_is_ce_plus_weighted_ece(logits_labels):
    from trustqat.keras import SoftECE, TrustAwareLoss

    logits, labels = logits_labels
    lg = logits.astype('float32')
    total = TrustAwareLoss([{'name': 'soft_ece', 'weight': 2.0, 'n_bins': 15}])
    ce = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    expected = float(ce(labels, lg)) + 2.0 * float(SoftECE(15)(labels, lg))
    assert float(total(labels, lg)) == pytest.approx(expected, rel=1e-5)
    plain = TrustAwareLoss([])
    assert float(plain(labels, lg)) == pytest.approx(float(ce(labels, lg)), rel=1e-6)


def test_trust_aware_loss_serialization():
    from trustqat.keras import TrustAwareLoss

    loss = TrustAwareLoss([{'name': 'soft_ece', 'weight': 0.5, 'n_bins': 10}])
    restored = keras.losses.deserialize(keras.losses.serialize(loss))
    assert isinstance(restored, TrustAwareLoss)
    assert restored.objective_configs == [{'name': 'soft_ece', 'weight': 0.5, 'n_bins': 10}]


@pytest.mark.skipif(keras.backend.backend() != 'jax', reason='gradient check uses jax')
def test_keras_soft_ece_gradient_flows_through_confidence(logits_labels):
    import jax

    from trustqat.keras import SoftECE

    logits, labels = logits_labels
    loss = SoftECE(15)
    g = np.asarray(jax.grad(lambda z: loss(labels, z))(logits.astype('float32')))
    assert np.all(np.isfinite(g)) and np.abs(g).sum() > 0


# ----------------------------------------------------------------------------- torch
@pytest.mark.skipif(importlib.util.find_spec('torch') is None, reason='torch not installed')
def test_torch_soft_ece_matches_numpy_and_has_gradients(logits_labels):
    import torch

    from trustqat.torch import SoftECELoss, TrustAwareLoss

    logits, labels = logits_labels
    z = torch.tensor(logits, dtype=torch.float64, requires_grad=True)
    y = torch.tensor(labels)
    v = SoftECELoss(15)(z, y)
    assert float(v) == pytest.approx(_soft_ece_numpy(logits, labels), abs=1e-6)
    v.backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0

    total = TrustAwareLoss([{'name': 'soft_ece', 'weight': 2.0, 'n_bins': 15}])
    expected = torch.nn.functional.cross_entropy(z, y) + 2.0 * SoftECELoss(15)(z, y)
    assert float(total(z, y)) == pytest.approx(float(expected), rel=1e-9)
