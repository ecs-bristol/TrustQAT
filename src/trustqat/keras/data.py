"""Minimal in-memory dataset for Keras training when the backend does not provide one."""

from __future__ import annotations

import math

import keras
import numpy as np


class ArrayDataset(keras.utils.PyDataset):
    def __init__(self, x, y, batch_size: int, shuffle: bool = False, drop_last: bool = False, seed: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.x, self.y = np.asarray(x), np.asarray(y)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._rng = np.random.default_rng(seed)
        self._order = np.arange(len(self.x))

    def __len__(self):
        n = len(self.x) / self.batch_size
        return math.floor(n) if self.drop_last else math.ceil(n)

    def __getitem__(self, idx):
        sl = self._order[idx * self.batch_size : (idx + 1) * self.batch_size]
        return self.x[sl], self.y[sl]

    def on_epoch_end(self):
        if self.shuffle:
            self._rng.shuffle(self._order)
