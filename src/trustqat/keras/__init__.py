"""Keras 3 adapter: trust losses and metrics in ``keras.ops``, callbacks and a default trainer.

Works with any Keras backend (JAX, TensorFlow, PyTorch) and any Keras-based QAT library.
"""

from .callbacks import EpochLogger, ParetoCheckpoint
from .metrics import ECE
from .objectives import SoftECE, TrustAwareLoss
from .schedules import cosine_decay_restarts, get_schedule
from .trainer import KerasTrainer

__all__ = [
    'ECE',
    'EpochLogger',
    'KerasTrainer',
    'ParetoCheckpoint',
    'SoftECE',
    'TrustAwareLoss',
    'cosine_decay_restarts',
    'get_schedule',
]
