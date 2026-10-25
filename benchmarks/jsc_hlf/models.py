"""HLF JSC models for the ``hgq2`` backend.

TrustQAT starts from the HGQ artifact's checkpoints, which store the full architecture
(three ``QEinsumDenseBatchnorm`` layers of 64, 64 and 32 units and a ``QDense`` output),
so the factory loads ``model.checkpoint`` rather than re-defining the network.
"""

from __future__ import annotations

from typing import Any

import keras
from hgq.layers import QEinsumDenseBatchnorm

from trustqat.models import register_model


def load_hgq_checkpoint(path) -> keras.Model:
    """Load an HGQ artifact checkpoint (accepts the older serialised layer names)."""
    return keras.models.load_model(
        path,
        compile=False,
        custom_objects={
            'QEinsumDenseBatchnorm': QEinsumDenseBatchnorm,
            'hgq>QEinsumDenseBatchnorm': QEinsumDenseBatchnorm,
        },
    )


@register_model('jsc_hlf', backend='hgq2')
def build_jsc_hlf(cfg: Any) -> keras.Model:
    return load_hgq_checkpoint(cfg['model']['checkpoint'])
