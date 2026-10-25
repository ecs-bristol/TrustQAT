"""Confidence reliability via a piecewise-differentiable minibatch ECE surrogate.

Training term (paper Eq. 10)::

    L_ECE = sum_m |B_m| / N * | sg(acc(B_m)) - conf(B_m) |

Correctness indicators and bin membership are stop-gradient quantities; gradients flow
through the confidences, hence through the logits, the network parameters and (for QAT
backends with trainable quantizers) the precision parameters. Validation and test use
the dataset-level top-label ECE (Eq. 9).
"""

from __future__ import annotations

from numpy.typing import ArrayLike

from ..core.metrics import expected_calibration_error
from .base import TrustObjective, register_objective


@register_objective
class SoftECEObjective(TrustObjective):
    name = 'soft_ece'
    metric_name = 'ece'
    direction = 'min'

    def __init__(self, n_bins: int = 15):
        if int(n_bins) <= 0:
            raise ValueError('n_bins must be positive')
        super().__init__(n_bins=int(n_bins))

    @property
    def n_bins(self) -> int:
        return self.params['n_bins']

    def numpy_metric(self, logits: ArrayLike, labels: ArrayLike) -> float:
        return expected_calibration_error(logits, labels, n_bins=self.n_bins)

    def keras_loss(self):
        from ..keras.objectives import SoftECE

        return SoftECE(n_bins=self.n_bins, from_logits=True)

    def keras_metric(self):
        from ..keras.metrics import ECE

        return ECE(n_bins=self.n_bins, from_logits=True, name=self.metric_name)

    def torch_loss(self):
        from ..torch.objectives import SoftECELoss

        return SoftECELoss(n_bins=self.n_bins)
