"""PyTorch implementations of trustworthiness training terms (for PyTorch-based QAT backends)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class SoftECELoss(nn.Module):
    """Minibatch soft-ECE surrogate (paper Eq. 10); same maths as :class:`trustqat.keras.SoftECE`."""

    def __init__(self, n_bins: int = 15, from_logits: bool = True, eps: float = 1e-7):
        super().__init__()
        if n_bins <= 0:
            raise ValueError('n_bins must be positive')
        self.n_bins = int(n_bins)
        self.from_logits = from_logits
        self.eps = eps

    def forward(self, outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(outputs, dim=-1) if self.from_logits else outputs
        target = target.reshape(-1).long()
        confidence, prediction = probs.max(dim=-1)
        correct = (prediction == target).to(probs.dtype).detach()
        bin_ids = (confidence.detach() * self.n_bins).long().clamp(0, self.n_bins - 1)

        n = confidence.shape[0]
        ece = confidence.new_zeros(())
        for b in range(self.n_bins):
            mask = (bin_ids == b).to(probs.dtype)
            count = mask.sum()
            bin_acc = (correct * mask).sum() / (count + self.eps)
            bin_conf = (confidence * mask).sum() / (count + self.eps)
            ece = ece + count / (n + self.eps) * (bin_acc - bin_conf).abs()
        return ece


class TrustAwareLoss(nn.Module):
    """``mean(CE) + sum_k weight_k * L_k``; ``objectives`` as in the Keras version."""

    def __init__(self, objectives: list[dict[str, Any]] | None = None):
        super().__init__()
        from ..objectives import objectives_from_config

        terms = objectives_from_config(objectives)
        self.terms = nn.ModuleList([obj.torch_loss() for obj, _ in terms])
        self.weights = [w for _, w in terms]
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.ce(outputs, target.reshape(-1).long())
        for term, w in zip(self.terms, self.weights, strict=True):
            loss = loss + w * term(outputs, target)
        return loss
