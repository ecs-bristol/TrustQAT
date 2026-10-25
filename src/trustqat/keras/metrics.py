"""Streaming Keras metrics matching :mod:`trustqat.core.metrics`."""

from __future__ import annotations

import keras
from keras import ops

from .objectives import _probabilities, _sparse_labels


@keras.saving.register_keras_serializable(package='trustqat')
class ECE(keras.metrics.Metric):
    """Dataset-level top-label expected calibration error, accumulated over batches."""

    def __init__(self, n_bins: int = 15, from_logits: bool = True, name: str = 'ece', **kwargs):
        super().__init__(name=name, **kwargs)
        if n_bins <= 0:
            raise ValueError('n_bins must be positive.')
        self.n_bins = int(n_bins)
        self.from_logits = bool(from_logits)
        self.bin_total = self.add_weight(shape=(self.n_bins,), initializer='zeros', name='bin_total')
        self.bin_correct = self.add_weight(shape=(self.n_bins,), initializer='zeros', name='bin_correct')
        self.bin_confidence = self.add_weight(shape=(self.n_bins,), initializer='zeros', name='bin_confidence')

    def update_state(self, y_true, y_pred, sample_weight=None):
        probs = _probabilities(y_pred, self.from_logits)
        labels = _sparse_labels(y_true)
        dtype = self.bin_total.dtype

        confidence = ops.cast(ops.max(probs, axis=-1), dtype)
        prediction = ops.cast(ops.argmax(probs, axis=-1), 'int32')
        correct = ops.cast(ops.equal(prediction, labels), dtype)

        bin_ids = ops.cast(confidence * self.n_bins, 'int32')
        bin_ids = ops.maximum(ops.minimum(bin_ids, self.n_bins - 1), 0)
        bins = ops.cast(ops.one_hot(bin_ids, self.n_bins), dtype)

        if sample_weight is not None:
            weights = ops.cast(ops.reshape(sample_weight, (-1,)), dtype)
            bins = bins * ops.expand_dims(weights, -1)

        self.bin_total.assign_add(ops.sum(bins, axis=0))
        self.bin_correct.assign_add(ops.sum(bins * ops.expand_dims(correct, -1), axis=0))
        self.bin_confidence.assign_add(ops.sum(bins * ops.expand_dims(confidence, -1), axis=0))

    def result(self):
        total = ops.sum(self.bin_total)
        eps = ops.cast(keras.backend.epsilon(), self.bin_total.dtype)
        acc = self.bin_correct / (self.bin_total + eps)
        conf = self.bin_confidence / (self.bin_total + eps)
        return ops.sum((self.bin_total / (total + eps)) * ops.abs(acc - conf))

    def reset_state(self):
        for v in (self.bin_total, self.bin_correct, self.bin_confidence):
            v.assign(ops.zeros(v.shape, dtype=v.dtype))

    def get_config(self):
        config = super().get_config()
        config.update({'n_bins': self.n_bins, 'from_logits': self.from_logits})
        return config
