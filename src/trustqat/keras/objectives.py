"""Keras 3 implementations of trustworthiness training terms (backend-agnostic ``keras.ops``)."""

from __future__ import annotations

from typing import Any

import keras
from keras import ops


def _sparse_labels(y_true):
    return ops.cast(ops.reshape(y_true, (-1,)), 'int32')


def _probabilities(y_pred, from_logits: bool):
    return ops.softmax(y_pred, axis=-1) if from_logits else y_pred


@keras.saving.register_keras_serializable(package='trustqat')
class SoftECE(keras.losses.Loss):
    """Minibatch soft-ECE surrogate for sparse categorical targets (paper Eq. 10).

    Hard correctness and bin assignments are stop-gradient signals; gradients flow
    through the confidence term, pulling each occupied bin's mean confidence toward its
    observed batch accuracy. Returns a scalar per batch.
    """

    def __init__(self, n_bins: int = 15, from_logits: bool = True, name: str = 'soft_ece', **kwargs):
        super().__init__(name=name, **kwargs)
        if n_bins <= 0:
            raise ValueError('n_bins must be positive.')
        self.n_bins = int(n_bins)
        self.from_logits = bool(from_logits)

    def call(self, y_true, y_pred):
        probs = _probabilities(y_pred, self.from_logits)
        labels = _sparse_labels(y_true)
        dtype = probs.dtype

        confidence = ops.max(probs, axis=-1)
        prediction = ops.cast(ops.argmax(probs, axis=-1), 'int32')
        correct = ops.stop_gradient(ops.cast(ops.equal(prediction, labels), dtype))

        bin_ids = ops.cast(confidence * self.n_bins, 'int32')
        bin_ids = ops.maximum(ops.minimum(bin_ids, self.n_bins - 1), 0)
        bin_ids = ops.stop_gradient(bin_ids)

        batch_size = ops.cast(ops.shape(confidence)[0], dtype)
        eps = ops.cast(keras.backend.epsilon(), dtype)
        ece = ops.cast(0.0, dtype)
        for bin_id in range(self.n_bins):
            mask = ops.cast(ops.equal(bin_ids, bin_id), dtype)
            count = ops.sum(mask)
            bin_weight = count / (batch_size + eps)
            bin_acc = ops.sum(correct * mask) / (count + eps)
            bin_conf = ops.sum(confidence * mask) / (count + eps)
            ece = ece + bin_weight * ops.abs(bin_acc - bin_conf)
        return ece

    def get_config(self):
        config = super().get_config()
        config.update({'n_bins': self.n_bins, 'from_logits': self.from_logits})
        return config


@keras.saving.register_keras_serializable(package='trustqat')
class TrustAwareLoss(keras.losses.Loss):
    """``L = CE + sum_k weight_k * L_k`` for sparse categorical classification.

    ``objectives`` is a list of ``{"name": ..., "weight": ..., **params}`` entries
    resolved through :mod:`trustqat.objectives`. With an empty list this is plain
    sparse categorical cross-entropy. The cross-entropy term is per sample and the
    trust terms are per batch; the default ``sum_over_batch_size`` reduction therefore
    gives ``mean(CE) + sum_k weight_k * L_k``.
    """

    def __init__(
        self,
        objectives: list[dict[str, Any]] | None = None,
        from_logits: bool = True,
        name: str = 'trust_aware_loss',
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        from ..objectives import objectives_from_config

        self.objective_configs = [dict(o) for o in (objectives or [])]
        self.from_logits = bool(from_logits)
        self._terms = [(obj.keras_loss(), w) for obj, w in objectives_from_config(self.objective_configs)]

    def call(self, y_true, y_pred):
        loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=self.from_logits)
        for term, weight in self._terms:
            loss = loss + weight * term(y_true, y_pred)
        return loss

    def get_config(self):
        config = super().get_config()
        config.update({'objectives': self.objective_configs, 'from_logits': self.from_logits})
        return config
