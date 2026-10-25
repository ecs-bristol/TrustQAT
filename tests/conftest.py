import os
import sys
from pathlib import Path

os.environ.setdefault('KERAS_BACKEND', 'jax')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(256, 8)).astype(np.float32)
    w = rng.normal(size=(8, 3))
    y = np.argmax(x @ w + 0.5 * rng.normal(size=(256, 3)), axis=1).astype(np.int32)
    return x, y


@pytest.fixture
def logits_labels():
    rng = np.random.default_rng(1)
    logits = rng.normal(scale=2.0, size=(1000, 5))
    labels = rng.integers(0, 5, size=1000)
    return logits, labels
