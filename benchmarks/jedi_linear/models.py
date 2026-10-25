"""JEDI-linear models written with HGQ2 layers, registered for the ``hgq2`` backend.

Adapted from JEDI-linear (Apache-2.0, Copyright 2025 Chang Sun); see NOTICE. The
architecture and quantiser configuration are unchanged from the TrustQAT-AE artifact.

``model.model_class``:

- ``gnn``: order-sensitive JEDI-linear (precision may differ per particle position)
- ``gnn_uq1``: permutation-invariant JEDI-linear (precision shared over particles)
- ``mlp``, ``mlp_mixer``, ``mlp_mixer_uq1``: reference models from JEDI-linear
"""

from __future__ import annotations

from math import log2
from typing import Any

import keras
from hgq.config import LayerConfigScope, QuantizerConfig, QuantizerConfigScope
from hgq.constraints import MinMax
from hgq.layers import QAdd, QEinsumDense, QEinsumDenseBatchnorm, QSum
from hgq.regularizers import MonoL1

from trustqat.models import register_model

if keras.backend.backend() == 'jax':
    import jax

    jax.config.update('jax_default_matmul_precision', 'tensorfloat32')


def _shape(cfg) -> tuple[int, int]:
    data = cfg['data']
    return int(data['n_constituents']), 3 if data['pt_eta_phi'] else 16


def get_mlpm(cfg, uq1: bool = False):
    iq_conf = QuantizerConfig(place='datalane', round_mode='RND')
    iq_default = QuantizerConfig(place='datalane')
    N, n = _shape(cfg)
    heterogeneous_axis = None if not uq1 else (-1,)

    with QuantizerConfigScope(place='datalane', heterogeneous_axis=heterogeneous_axis):
        inp = keras.layers.Input((N, n))
        x1 = QEinsumDenseBatchnorm('bnc,cC->bnC', (N, 16), bias_axes='C', activation='relu', iq_conf=iq_conf)(inp)
        x1 = QEinsumDenseBatchnorm('bnc,cC->bnC', (N, n), bias_axes='C', activation='relu')(x1)
        x2 = QEinsumDenseBatchnorm('bnc,nN->bNc', (N, n), bias_axes='N')(x1)
        x = QAdd(iq_confs=(iq_conf, iq_default))([inp, x2])
        x = QEinsumDenseBatchnorm('bnc,cC->bnC', (N, 16), bias_axes='C', activation='relu')(x)
        x = QEinsumDenseBatchnorm('bnc,cC->bnC', (N, 16), bias_axes='C', activation='relu')(x)
        x = QEinsumDense('bnc,n->bc', 16)(x)
        x = QEinsumDenseBatchnorm('bc,cC->bC', 16, bias_axes='C', activation='relu')(x)
        x = QEinsumDenseBatchnorm('bc,cC->bC', 16, bias_axes='C', activation='relu')(x)
        x = QEinsumDenseBatchnorm('bc,cC->bC', 16, bias_axes='C', activation='relu')(x)
        out = QEinsumDenseBatchnorm('bc,cC->bC', 5, bias_axes='C')(x)
    return keras.Model(inputs=inp, outputs=out)


def get_mlp(cfg):
    iq_conf = QuantizerConfig(place='datalane', round_mode='RND')
    N, n = _shape(cfg)
    inp = keras.layers.Input((N, n))
    x = keras.layers.Flatten()(inp)
    x = QEinsumDenseBatchnorm('bc->bC', 64, bias_axes='C', activation='relu', iq_conf=iq_conf)(x)
    x = QEinsumDenseBatchnorm('bc->bC', 64, bias_axes='C', activation='relu')(x)
    x = QEinsumDenseBatchnorm('bc->bC', 64, bias_axes='C', activation='relu')(x)
    x = QEinsumDenseBatchnorm('bc->bC', 64, bias_axes='C', activation='relu')(x)
    out = QEinsumDenseBatchnorm('bc->bC', 5, bias_axes='C')(x)
    return keras.Model(inputs=inp, outputs=out)


def get_gnn(cfg, uq1: bool = False):
    N, n = _shape(cfg)
    heterogeneous_axis = None if not uq1 else (-1,)

    with (
        QuantizerConfigScope(place=('weight', 'bias'), overflow_mode='SAT_SYM'),
        QuantizerConfigScope(place='datalane', heterogeneous_axis=heterogeneous_axis),
    ):
        inp = keras.layers.Input((N, n))
        pool_scale = 2.0 ** -round(log2(N))
        x = QEinsumDenseBatchnorm('bnc,cC->bnC', (N, 64), bias_axes='C', activation='relu')(inp)
        s = QEinsumDenseBatchnorm('bnc,cC->bnC', (N, 64), bias_axes='C', activation='relu')(x)
        d = QEinsumDenseBatchnorm('bnc,cC->bnC', (1, 64), bias_axes='C', activation='relu')(
            QSum(axes=1, scale=pool_scale, keepdims=True)(x)
        )
        x = QAdd()([s, d])
        x = QEinsumDenseBatchnorm('bnc,cC->bnC', (N, 64), bias_axes='C', activation='relu')(x)
        x = QSum(axes=1, scale=1 / 16, keepdims=False)(x)
        x = QEinsumDenseBatchnorm('bc,cC->bC', 64, bias_axes='C', activation='relu')(x)
        x = QEinsumDenseBatchnorm('bc,cC->bC', 32, bias_axes='C', activation='relu')(x)
        x = QEinsumDenseBatchnorm('bc,cC->bC', 16, bias_axes='C', activation='relu')(x)
        out = QEinsumDenseBatchnorm('bc,cC->bC', 5, bias_axes='C')(x)
    return keras.Model(inputs=inp, outputs=out)


@register_model('jedi_linear', backend='hgq2')
def build_jedi_linear(cfg: Any):
    """Build a JEDI-linear model from a run config (``data`` and ``model`` sections)."""
    m = cfg['model']
    scope0 = QuantizerConfigScope(
        default_q_type='kbi',
        b0=m['init_bw_k'],
        overflow_mode='wrap',
        i0=0,
        fr=MonoL1(m['k_bw_l1_reg']),
        ir=MonoL1(m['k_bw_l1_reg']),
    )
    scope1 = QuantizerConfigScope(
        default_q_type='kif',
        place='datalane',
        overflow_mode='wrap',
        f0=m['init_bw_a'],
        fr=MonoL1(m['a_bw_l1_reg']),
        ic=MinMax(0, 12),
    )
    scope2 = LayerConfigScope(beta0=0)
    with scope0, scope1, scope2:
        match m['model_class']:
            case 'mlp_mixer':
                return get_mlpm(cfg)
            case 'mlp_mixer_uq1':
                return get_mlpm(cfg, uq1=True)
            case 'mlp':
                return get_mlp(cfg)
            case 'gnn':
                return get_gnn(cfg)
            case 'gnn_uq1':
                return get_gnn(cfg, uq1=True)
            case other:
                raise ValueError(f'unknown model_class {other!r}')
