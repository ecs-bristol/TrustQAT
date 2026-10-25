"""A deliberately minimal fixed-precision Keras QAT backend, used only to test the API.

It has no trainable precision and no cost regulariser, so it exercises the code paths a
QKerasV3-style backend would take. Not for research results.
"""

from __future__ import annotations

from pathlib import Path

import keras
from keras import ops

from trustqat.backends import BaseBackend


def fake_quant(x, bits: int, max_abs: float = 4.0):
    scale = (2 ** (bits - 1) - 1) / max_abs
    q = ops.clip(ops.round(x * scale), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1) / scale
    return x + ops.stop_gradient(q - x)  # straight-through estimator


@keras.saving.register_keras_serializable(package='trustqat_toy')
class FixedQuantDense(keras.layers.Layer):
    def __init__(self, units: int, w_bits: int = 6, a_bits: int = 8, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units, self.w_bits, self.a_bits = units, w_bits, a_bits
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units), initializer='glorot_uniform', name='kernel')
        self.bias = self.add_weight(shape=(self.units,), initializer='zeros', name='bias')

    def call(self, x):
        x = fake_quant(x, self.a_bits)
        y = ops.matmul(x, fake_quant(self.kernel, self.w_bits)) + self.bias
        return self.activation(y)

    def bops(self) -> float:
        return float(self.kernel.shape[0] * self.kernel.shape[1] * self.w_bits * self.a_bits)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            units=self.units,
            w_bits=self.w_bits,
            a_bits=self.a_bits,
            activation=keras.activations.serialize(self.activation),
        )
        return cfg


def tiny_model(n_in: int = 8, n_classes: int = 3):
    inp = keras.Input((n_in,))
    x = FixedQuantDense(16, activation='relu')(inp)
    out = FixedQuantDense(n_classes)(x)
    return keras.Model(inp, out)


class ToyKerasFixedBackend(BaseBackend):
    name = 'toy-keras-fixed'
    framework = 'keras'
    capabilities = frozenset()
    cost_names = ('bops',)
    checkpoint_suffix = '.keras'

    def cost(self, model):
        return {'bops': sum(layer.bops() for layer in model.layers if isinstance(layer, FixedQuantDense))}

    def save(self, model, path: Path):
        model.save(Path(path))

    def load(self, path: Path, cfg=None):
        return keras.models.load_model(Path(path), compile=False)
