# TrustQAT

Trustworthiness-aware quantization-aware training (QAT) for hardware-efficient neural
networks.

Hardware-aware QAT usually trades off two things: predictive performance and
implementation cost. TrustQAT adds a measurable trustworthiness objective to both
**training** and **candidate selection**:

```
L = L_task + Σ_k λ_k · L_trust,k + (backend cost regulariser, e.g. β(t)·EBOPs)
```

Checkpoints are then kept on a validation Pareto front over accuracy, the trust metrics and
hardware cost. Nothing is added to the inference graph: any hardware effect comes
indirectly, through a different precision allocation.

The first objective is **confidence reliability**. It is trained with a
piecewise-differentiable minibatch ECE surrogate and reported as top-label ECE. With a
QAT library whose quantizers are trainable (HGQ2), the calibration gradients reach the
per-weight and per-activation bit-widths.

## Design

```
trustqat/
  core/        numpy only: ECE/MCE/NLL/Brier, Pareto selection, epoch records,
               temperature scaling, paired comparisons
  objectives/  registry of trust objectives (spec + numpy reference metric)
  keras/       keras.ops losses/metrics, Pareto checkpoint callback, default trainer
  torch/       PyTorch losses (for PyTorch QAT libraries such as Brevitas)
  backends/    QAT backend interface + the bundled HGQ2 backend
  models/      model-factory registry, per (benchmark, backend)
  testing/     backend conformance checks
benchmarks/
  common/      shared train / test / export / select driver
  jedi_linear/ JEDI-linear jet tagging (paper Tables II and III)
  jsc_hlf/     HLF jet substructure classification (paper Table I)
```

- **One backend ships, the API is open.** HGQ2 is the only bundled backend. Other QAT
  libraries plug in through the `trustqat.backends` entry-point group, so they need no
  change to this repository. See [docs/writing-a-backend.md](docs/writing-a-backend.md).
- **The core is framework-free.** `trustqat.core`, `trustqat.objectives` and
  `trustqat.backends` import neither Keras, PyTorch nor HGQ; the tests check this.
- **No HGQ fork.** The trust losses are plain `keras.ops` code in `trustqat.keras`, and
  HGQ2 is used unmodified from PyPI.

## Install

```bash
conda env create -f environment.yml   # creates the env "trustqat" (Python 3.12, JAX CUDA 12)
conda activate trustqat               # sets KERAS_BACKEND=jax
pytest -q
```

Or with pip into an existing environment:

```bash
pip install -e ".[hgq2,da4ml,jedi,dev]"   # plus a JAX build for your platform, e.g. "jax[cuda12]"
export KERAS_BACKEND=jax
```

## Quick start

```python
from trustqat.backends import get_backend
from trustqat.keras import KerasTrainer

backend = get_backend('hgq2')
cfg = {
    'train': {
        'batch_size': 1024,
        'epochs': 100,
        'lr_schedule': {
            'name': 'cosine_restarts',
            'initial_learning_rate': 3e-4,
            'first_decay_steps': 100,
            'alpha': 1e-6,
            'alpha_steps': 10,
        },
    },
    'trust': {'objectives': [{'name': 'soft_ece', 'weight': 2.0, 'n_bins': 15}]},
    'selection': {
        'objectives': {'val_accuracy': 'max', 'val_ece': 'min', 'cost.ebops': 'min'},
        'constraints': {'val_accuracy': {'gt': 0.70}},
    },
    'backend': {'beta': {'intervals': [[0, 1e-7, 'log'], [100, 3e-6, 'constant']]}},
}
KerasTrainer(backend, cfg).fit(hgq_model, x_train, y_train, x_val, y_val, 'runs/my_run')
```

A run directory contains:

- `records.jsonl`: one record per epoch (validation metrics plus backend costs)
- `ckpts/`: checkpoints on the current Pareto front
- `pareto.json`: the current front
- `selection.json`: the selection objectives and constraints
- `history.json`
- `last.keras`

Re-select offline with:

```bash
trustqat select runs/my_run/records.jsonl                     # uses selection.json
trustqat select runs/my_run/records.jsonl -o val_ece:min -c "val_accuracy>=0.72"
trustqat list-backends
trustqat list-objectives
```

### Using TrustQAT from another training loop

Any training script (PyTorch, Brevitas, ...) can reuse TrustQAT selection and reporting
without implementing a backend:

1. Add the loss: `from trustqat.torch import TrustAwareLoss`.
2. Write one JSON line per epoch:
   `{"epoch": 7, "metrics": {"val_accuracy": ..., "val_ece": ...}, "cost": {"bops": ...}, "checkpoint": "..."}`.
3. Run `trustqat select`.

## Reproducing the paper results

- **HLF JSC (Table I).** See [benchmarks/jsc_hlf/README.md](benchmarks/jsc_hlf/README.md).
  The five configs fine-tune the HGQ checkpoints at 530 to 13285 EBOPs.
- **JEDI-linear (Tables II and III).** See
  [benchmarks/jedi_linear/README.md](benchmarks/jedi_linear/README.md). The 20 TrustQAT
  configurations and the 20 HGQ baseline sweeps of TrustQAT-AE are in
  `benchmarks/jedi_linear/configs/`.

Both benchmarks share one driver (`benchmarks/common/driver.py`) with the phases `train`,
`test`, `select` and `export`.

## Reproducibility

JAX's default GPU kernels are not bit-deterministic. QAT can amplify last-bit differences
when a value crosses a quantization boundary, so long runs may diverge. For bit-identical
repeated runs, set:

```bash
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
```

This changes the numerics slightly. Leave it unset when reproducing the TrustQAT-AE
artifact, which was trained without it.

## License

Apache-2.0. See `LICENSE` and `NOTICE`. The JEDI-linear benchmark contains code derived
from JEDI-linear (Apache-2.0).
