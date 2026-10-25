"""A minimal PyTorch QAT backend with one learnable bit-width per layer, used only to test the API.

It shows how a Brevitas-like backend plugs in: ``framework = "torch"``, trainable
(per-tensor) precision and BOPs as the cost. Not for research results.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from trustqat.backends import FREEZE_PRECISION, TRAINABLE_PRECISION, BaseBackend


def _ste_round(x):
    return x + (x.round() - x).detach()


def fake_quant(x, bits, max_abs: float = 4.0):
    """Fake-quantise with a differentiable (continuous) bit-width, as in learned-precision QAT."""
    levels = 2 ** (_ste_round(bits) - 1) - 1
    scale = levels / max_abs
    q = torch.clamp(_ste_round(x * scale), -levels - 1, levels) / scale
    return q


class LearnedBitsLinear(nn.Module):
    def __init__(self, n_in: int, n_out: int, bits0: float = 6.0):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)
        self.bits = nn.Parameter(torch.tensor(float(bits0)))

    def forward(self, x):
        w = fake_quant(self.linear.weight, self.bits.clamp(2.0, 16.0))
        return nn.functional.linear(x, w, self.linear.bias)


class TinyNet(nn.Module):
    def __init__(self, n_in: int = 8, n_classes: int = 3):
        super().__init__()
        self.l1 = LearnedBitsLinear(n_in, 16)
        self.l2 = LearnedBitsLinear(16, n_classes)

    def forward(self, x):
        return self.l2(torch.relu(self.l1(x.float())))


class ToyTorchBackend(BaseBackend):
    name = 'toy-torch-learned-bits'
    framework = 'torch'
    capabilities = frozenset({TRAINABLE_PRECISION, FREEZE_PRECISION})
    cost_names = ('bops',)
    checkpoint_suffix = '.pt'

    def cost(self, model):
        total = 0.0
        for m in model.modules():
            if isinstance(m, LearnedBitsLinear):
                n_out, n_in = m.linear.weight.shape
                total += n_in * n_out * float(m.bits.detach().round()) * 8.0
        return {'bops': total}

    def save(self, model, path: Path):
        torch.save({'state_dict': model.state_dict()}, Path(path))

    def load(self, path: Path, cfg=None):
        model = TinyNet()
        model.load_state_dict(torch.load(Path(path))['state_dict'])
        return model

    def precision_variables(self, model):
        return [m.bits for m in model.modules() if isinstance(m, LearnedBitsLinear)]

    def freeze_precision(self, model):
        for p in self.precision_variables(model):
            p.requires_grad_(False)
