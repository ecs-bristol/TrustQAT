import json

import numpy as np
import pytest

from trustqat.core import (
    EpochRecord,
    ParetoArchive,
    apply_temperature,
    calibration_report,
    expected_calibration_error,
    fit_temperature,
    paired_summary,
    pareto_front,
    read_records,
    select_records,
    softmax,
    validate_record,
    write_record,
)
from trustqat.core.metrics import brier_score, maximum_calibration_error, negative_log_likelihood
from trustqat.core.pareto import dominates


# ----------------------------------------------------------------------------- metrics
def test_ece_hand_computed():
    # two bins used: conf 0.9 (2 samples, 1 correct) and conf 0.6 (2 samples, 2 correct)
    probs = np.array([[0.9, 0.1], [0.9, 0.1], [0.6, 0.4], [0.4, 0.6]])
    labels = np.array([0, 1, 0, 1])
    ece = expected_calibration_error(probs, labels, n_bins=10, from_logits=False)
    expected = 0.5 * abs(0.5 - 0.9) + 0.5 * abs(1.0 - 0.6)
    assert ece == pytest.approx(expected)
    assert maximum_calibration_error(probs, labels, 10, from_logits=False) == pytest.approx(0.4)


def test_ece_matches_artifact_convention(logits_labels):
    """Same binning as TrustQAT-AE evaluate_calibration.py: min(int(c * M), M - 1)."""
    logits, labels = logits_labels
    p = softmax(logits)
    conf, pred = p.max(1), p.argmax(1)
    ids = np.minimum((conf * 15).astype(np.int64), 14)
    ece = 0.0
    for b in range(15):
        m = ids == b
        if m.any():
            ece += m.sum() / len(labels) * abs((pred[m] == labels[m]).mean() - conf[m].mean())
    assert expected_calibration_error(logits, labels, 15) == pytest.approx(ece, abs=1e-12)


def test_perfectly_calibrated_has_zero_ece():
    probs = np.array([[1.0, 0.0]] * 10)
    labels = np.zeros(10, dtype=int)
    assert expected_calibration_error(probs, labels, from_logits=False) == pytest.approx(0.0)


def test_report_and_scores(logits_labels):
    logits, labels = logits_labels
    rep = calibration_report(logits, labels, n_bins=15)
    assert set(rep) >= {'acc', 'ece', 'mce', 'nll', 'brier', 'bins'}
    assert sum(b['count'] for b in rep['bins']) == len(labels)
    assert rep['nll'] == pytest.approx(negative_log_likelihood(logits, labels))
    assert 0 <= brier_score(logits, labels) <= 2


def test_temperature_scaling_recovers_temperature():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(scale=3.0, size=(20000, 4))
    labels = np.array([rng.choice(4, p=p) for p in softmax(true_logits)])
    t = fit_temperature(true_logits * 2.0, labels)  # overconfident by 2x
    assert t == pytest.approx(2.0, rel=0.05)
    assert expected_calibration_error(apply_temperature(true_logits * 2.0, t), labels) < expected_calibration_error(
        true_logits * 2.0, labels
    )


# ----------------------------------------------------------------------------- pareto
def test_dominance_and_front():
    dirs = ['max', 'min']
    assert dominates([0.8, 10], [0.7, 10], dirs)
    assert not dominates([0.8, 10], [0.8, 10], dirs)
    pts = [[0.8, 10], [0.7, 5], [0.7, 12], [0.8, 10], [0.9, 30]]
    assert pareto_front(pts, dirs) == [0, 1, 3, 4]  # duplicates of a front point both stay


def test_select_records_with_constraints():
    recs = [
        EpochRecord(0, {'val_accuracy': 0.60, 'val_ece': 0.01}, {'ebops': 100}),
        EpochRecord(1, {'val_accuracy': 0.75, 'val_ece': 0.02}, {'ebops': 200}),
        EpochRecord(2, {'val_accuracy': 0.74, 'val_ece': 0.03}, {'ebops': 250}),  # dominated by 1
        EpochRecord(3, {'val_accuracy': 0.72, 'val_ece': 0.01}, {'ebops': 300}),
    ]
    obj = {'val_accuracy': 'max', 'val_ece': 'min', 'cost.ebops': 'min'}
    assert [r.epoch for r in select_records(recs, obj)] == [0, 1, 3]
    assert [r.epoch for r in select_records(recs, obj, {'val_accuracy': {'gt': 0.7}})] == [1, 3]


def test_archive_evicts_dominated():
    a = ParetoArchive({'val_accuracy': 'max', 'cost.ebops': 'min'}, {'val_accuracy': {'min': 0.5}})
    r0 = EpochRecord(0, {'val_accuracy': 0.7}, {'ebops': 100})
    r1 = EpochRecord(1, {'val_accuracy': 0.4}, {'ebops': 10})  # violates constraint
    r2 = EpochRecord(2, {'val_accuracy': 0.7}, {'ebops': 100})  # identical: rejected
    r3 = EpochRecord(3, {'val_accuracy': 0.8}, {'ebops': 90})  # dominates r0
    assert a.add(r0) == (True, [])
    assert a.add(r1) == (False, [])
    assert a.add(r2) == (False, [])
    ok, evicted = a.add(r3)
    assert ok and evicted == [r0]
    assert a.members == [r3]


# ----------------------------------------------------------------------------- records
def test_record_roundtrip_and_flat_form(tmp_path):
    p = tmp_path / 'records.jsonl'
    r = EpochRecord(3, {'val_accuracy': 0.7, 'val_ece': 0.01}, {'ebops': 123.0}, 'ckpts/epoch=0003.keras')
    write_record(p, r)
    with p.open('a') as f:
        f.write(json.dumps({'epoch': 4, 'val_accuracy': 0.71, 'cost.ebops': 120}) + '\n')
    a, b = read_records(p)
    assert a == r
    assert b.metrics == {'val_accuracy': 0.71} and b.cost == {'ebops': 120.0}
    assert b.get('cost.ebops') == 120.0


@pytest.mark.parametrize(
    'bad',
    [
        {'metrics': {'a': 1.0}},
        {'epoch': -1, 'metrics': {'a': 1.0}},
        {'epoch': 1, 'metrics': {}},
        {'epoch': 1, 'metrics': {'a': 'x'}},
        {'epoch': 1, 'metrics': {'a': 1.0}, 'cost': {'ebops': -5}},
        {'epoch': 1, 'checkpoint': 3, 'metrics': {'a': 1.0}},
        {'epoch': 1},
    ],
)
def test_invalid_records(bad):
    with pytest.raises(ValueError):
        validate_record(bad)


# ----------------------------------------------------------------------------- compare
def test_paired_summary():
    pairs = [
        ({'ece': 0.01, 'acc': 0.80}, {'ece': 0.005, 'acc': 0.801}),
        ({'ece': 0.02, 'acc': 0.7}, {'ece': 0.03, 'acc': 0.7}),
    ]
    s = paired_summary(pairs)
    assert s['n_pairs'] == 2 and s['n_improved'] == 1
    assert s['median_relative_ece_reduction'] == pytest.approx((0.5 - 0.5) / 2)
