"""PyTorch adapter (requires ``torch``). Use these losses in any PyTorch QAT training loop,
e.g. with Brevitas, and write :mod:`trustqat.core.records` to reuse TrustQAT selection."""

from .objectives import SoftECELoss, TrustAwareLoss

__all__ = ['SoftECELoss', 'TrustAwareLoss']
