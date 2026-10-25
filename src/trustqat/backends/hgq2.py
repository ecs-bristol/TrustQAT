"""HGQ2 backend: High Granularity Quantization (per-weight / per-activation trainable precision).

Uses the unmodified upstream ``HGQ2`` package; the trustworthiness terms live in
:mod:`trustqat.keras`, not in HGQ. Calibration gradients reach HGQ's bit-width
parameters through HGQ's own surrogate gradients.

Backend config keys (``backend:`` section of a run config):

``beta``
    HGQ EBOPs regulariser schedule, ``{"intervals": [[epoch, beta, interp], ...]}``
    (``interp`` in ``constant | linear | log``). Omit to leave the layers' beta unchanged.
``dataset_device``
    where :class:`hgq.utils.sugar.Dataset` keeps the data: ``"auto"`` (default, first GPU
    if JAX sees one, else CPU), ``"gpu:0"``, ``"cpu:0"`` or ``"none"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .base import (
    API_VERSION,
    COST_REGULARIZER,
    FREEZE_PRECISION,
    PRECISION_SUMMARY,
    TRAINABLE_PRECISION,
    BaseBackend,
    PrecisionSummary,
    QuantizerStats,
    export_capability,
)

# Hardware defaults of the JEDI-linear / TrustQAT designs (AMD VU13P, 2 ns clock).
DA4ML_DEFAULTS: dict[str, Any] = {
    'part_name': 'xcvu13p-flga2577-2-e',
    'clock_period': 2.0,
    'clock_uncertainty': 0.0,
    'latency_cutoff': 2,
    'solver_options': {'hard_dc': 2},
    'hwconf': (1, -1, -1),
    'prj_name': 'model',
}


def _import_hgq():
    try:
        import hgq  # noqa: F401  (registers HGQ layers for deserialisation)
    except ImportError as e:  # pragma: no cover
        raise ImportError('The hgq2 backend needs HGQ2: pip install "trustqat[hgq2]"') from e
    return hgq


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, 'get'):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class HGQ2Backend(BaseBackend):
    name: ClassVar[str] = 'hgq2'
    framework: ClassVar[str] = 'keras'
    api_version: ClassVar[str] = API_VERSION
    capabilities: ClassVar[frozenset[str]] = frozenset(
        {
            TRAINABLE_PRECISION,
            FREEZE_PRECISION,
            PRECISION_SUMMARY,
            COST_REGULARIZER,
            export_capability('da4ml'),
        }
    )
    cost_names: ClassVar[tuple[str, ...]] = ('ebops',)
    checkpoint_suffix: ClassVar[str] = '.keras'

    def __init__(self):
        _import_hgq()

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _qlayers(model):
        from hgq.layers import QLayerBase

        return [layer for layer in model._flatten_layers() if isinstance(layer, QLayerBase)]

    @staticmethod
    def _fixed_point_quantizers(model):
        from hgq.quantizer.internal.fixed_point_quantizer import FixedPointQuantizerBase

        seen, out = set(), []
        for layer in model._flatten_layers():
            if isinstance(layer, FixedPointQuantizerBase) and id(layer) not in seen:
                seen.add(id(layer))
                out.append(layer)
        return out

    @staticmethod
    def _bitwidth_variables(q) -> list:
        """Gradient-learned bit-width parameters of one quantizer.

        KBI quantizers learn ``_b`` (and ``_i`` unless WRAP), KIF quantizers learn ``_f``
        (and ``_i`` unless WRAP). In WRAP mode ``_i`` tracks the data range instead of being
        learned, and the sign ``_k`` is never learned.
        """
        names = ['_b', '_f']
        if q.overflow_mode != 'WRAP':
            names.append('_i')
        return [getattr(q, n) for n in names if getattr(q, n, None) is not None]

    # ---------------------------------------------------------------- required
    def cost(self, model) -> dict[str, float]:
        from keras import ops

        ebops = 0.0
        for layer in self._qlayers(model):
            if layer.enable_ebops and getattr(layer, '_ebops', None) is not None:
                ebops += float(ops.convert_to_numpy(layer._ebops))
        return {'ebops': ebops}

    def save(self, model, path: Path) -> None:
        model.save(Path(path))

    def load(self, path: Path, cfg: Any = None):
        import keras

        _import_hgq()
        import trustqat.keras  # noqa: F401  (registers TrustQAT losses/metrics)

        return keras.models.load_model(Path(path), compile=False)

    # ---------------------------------------------------------------- hooks
    def training_hooks(self, cfg: Any = None) -> list:
        beta = _get(cfg, 'beta')
        if beta is None:
            return []
        from hgq.utils.sugar import BetaScheduler, PieceWiseSchedule

        intervals = [tuple(v) for v in _get(beta, 'intervals')]
        return [BetaScheduler(PieceWiseSchedule(intervals))]

    def make_dataset(self, x, y, batch_size: int, shuffle: bool, drop_last: bool, cfg: Any = None):
        from hgq.utils.sugar import Dataset

        device = _get(cfg, 'dataset_device', 'auto')
        if device == 'auto':
            device = 'cpu:0'
            try:
                import keras

                if keras.backend.backend() == 'jax':
                    import jax

                    if any(d.platform == 'gpu' for d in jax.devices()):
                        device = 'gpu:0'
            except Exception:  # pragma: no cover
                pass
        return Dataset(x, y, batch_size=batch_size, drop_last=drop_last, device=device, shuffle=shuffle)

    def load_weights(self, model, path: Path):
        """``model.load_weights`` with a fallback for checkpoints from older HGQ2 revisions.

        Some older checkpoints miss quantizer state (e.g. ``i_decay_speed``); in that case
        the full saved model is loaded instead, preferring the inference-ready copy that the
        benchmarks keep in ``../models/<same name>``.
        """
        import keras

        path = Path(path)
        try:
            model.load_weights(path)
            return model
        except ValueError as exc:
            msg = str(exc)
            if 'FixedPointQuantizer' not in msg:
                raise
            for candidate in (path.parent.parent / 'models' / path.name, path):
                if candidate.exists():
                    try:
                        return keras.models.load_model(candidate, compile=False)
                    except Exception:  # noqa: BLE001
                        continue
            raise

    def prepare_inference(self, model, calibration_data=None, batch_size: int = 16384, **kwargs):
        """Calibrate WRAP-mode integer bits on ``calibration_data`` (``hgq.utils.trace_minmax``)."""
        if calibration_data is not None:
            from hgq.utils import trace_minmax

            trace_minmax(model, calibration_data, batch_size=batch_size)
        return model

    # ---------------------------------------------------------------- capabilities
    def precision_variables(self, model) -> list:
        """All gradient-learned bit-width parameters (whether or not currently frozen)."""
        return [v for q in self._fixed_point_quantizers(model) if q.built for v in self._bitwidth_variables(q)]

    def freeze_precision(self, model) -> None:
        """Stop gradient updates of all bit-width parameters (call before ``compile``).

        Integer bits of WRAP-mode activations are not gradient-trained in HGQ; they keep
        tracking the data range, as in normal HGQ training.
        """
        for v in self.precision_variables(model):
            v.trainable = False

    def precision_summary(self, model) -> PrecisionSummary:
        from keras import ops

        stats = []
        for q in self._fixed_point_quantizers(model):
            if not q.built:
                continue
            bits = np.asarray(ops.convert_to_numpy(q.bits), dtype=np.float64)
            flat = np.round(bits.ravel()).astype(int)
            values, counts = np.unique(flat, return_counts=True)
            stats.append(
                QuantizerStats(
                    name=q.path if hasattr(q, 'path') else q.name,
                    kind='weight' if getattr(q, '_is_weight', False) else 'activation',
                    shape=tuple(int(s) for s in bits.shape),
                    n_elements=int(bits.size),
                    bits_min=float(bits.min()) if bits.size else 0.0,
                    bits_mean=float(bits.mean()) if bits.size else 0.0,
                    bits_max=float(bits.max()) if bits.size else 0.0,
                    n_zero=int(np.sum(bits <= 0)),
                    histogram={int(v): int(c) for v, c in zip(values, counts, strict=True)},
                )
            )
        return PrecisionSummary(stats)

    def export(self, model, target: str, out_dir: Path, **kwargs) -> Path:
        """Export to RTL with da4ml (``target="da4ml"``); returns the project directory.

        See :meth:`build_rtl` for the options.
        """
        if target != 'da4ml':
            return super().export(model, target, out_dir, **kwargs)
        self.build_rtl(model, out_dir, **kwargs)
        return Path(out_dir)

    def build_rtl(self, model, out_dir: Path, compile: bool = True, **kwargs):
        """Trace ``model`` with da4ml and write a Verilog project to ``out_dir``.

        ``kwargs`` override :data:`DA4ML_DEFAULTS` (``part_name``, ``clock_period``,
        ``clock_uncertainty``, ``latency_cutoff``, ``solver_options``, ``hwconf``,
        ``prj_name``). With ``compile=True`` the Verilator emulator is built, so the
        returned ``da4ml.codegen.VerilogModel`` can ``predict`` for bit-exact checks.
        The model should be inference-ready (:meth:`prepare_inference`).
        """
        from da4ml.codegen import VerilogModel
        from da4ml.converter import trace_model
        from da4ml.trace import HWConfig, comb_trace

        opts = {**DA4ML_DEFAULTS, **kwargs}
        inp, out = trace_model(model, hwconf=HWConfig(*opts['hwconf']), solver_options=opts['solver_options'])
        solution = comb_trace(inp, out)
        rtl = VerilogModel(
            solution,
            prj_name=opts['prj_name'],
            path=Path(out_dir),
            part_name=opts['part_name'],
            clock_period=opts['clock_period'],
            clock_uncertainty=opts['clock_uncertainty'],
            latency_cutoff=opts['latency_cutoff'],
        )
        if compile:
            rtl.compile()
        return rtl
