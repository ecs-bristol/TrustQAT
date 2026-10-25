# Writing a QAT backend

A backend adapts one QAT library to TrustQAT. HGQ2 is the bundled reference
implementation (`src/trustqat/backends/hgq2.py`). Two minimal test backends in
`tests/toy_backends/` show the smallest working versions:

- a fixed-precision Keras backend (QKerasV3-like)
- a PyTorch backend with learned per-tensor bit-widths (Brevitas-like)

## Two levels of integration

| Level | You provide | You get |
|---|---|---|
| Records only | Your own training loop writes `records.jsonl` (see `trustqat.core.records`) and uses `trustqat.keras` or `trustqat.torch` losses | Pareto selection, calibration reports, paired comparisons, `trustqat select` |
| Backend | A class implementing `QATBackend` | The above, plus the default trainer, config-driven runs, freeze-precision ablations, precision summaries, export |

## The interface

```python
from trustqat.backends import BaseBackend, TRAINABLE_PRECISION, FREEZE_PRECISION


class MyBackend(BaseBackend):
    name = 'mylib'
    framework = 'torch'  # or 'keras'
    capabilities = frozenset({TRAINABLE_PRECISION, FREEZE_PRECISION})
    cost_names = ('bops',)  # names of the costs returned by cost()
    checkpoint_suffix = '.pt'

    # required
    def cost(self, model) -> dict[str, float]: ...  # e.g. {'bops': 1.2e6}
    def save(self, model, path): ...
    def load(self, path, cfg=None): ...

    # required only for the declared capabilities
    def precision_variables(self, model): ...  # all learned bit-width params, frozen or not
    def freeze_precision(self, model): ...  # stop their updates

    # def precision_summary(self, model) -> PrecisionSummary: ...   # 'precision_summary'
    # def export(self, model, target, out_dir, **kw): ...           # 'export:<target>'

    # optional hooks (defaults in BaseBackend)
    # def training_hooks(self, cfg): ...        # extra callbacks, e.g. a cost-regulariser schedule
    # def make_dataset(self, x, y, batch_size, shuffle, drop_last, cfg=None): ...
    # def load_weights(self, model, path): ...
    # def prepare_inference(self, model, calibration_data=None, **kw): ...
```

Rules that keep backends interchangeable:

- **Cost names are yours.** The core never assumes `ebops`. Selection configs refer to
  costs as `cost.<name>`.
- **Models are not part of the backend.** Register factories per (benchmark, backend)
  with `@trustqat.models.register_model('jedi_linear', backend='mylib')`.
- **Precision summaries use the generic schema** (`QuantizerStats`). A per-tensor
  quantizer is just `shape == ()`.
- **Models output logits.** Trust objectives assume `from_logits=True`.

## Registering

In your package's `pyproject.toml`:

```toml
[project.entry-points."trustqat.backends"]
mylib = "trustqat_mylib:MyBackend"
```

`trustqat list-backends` should then list it. For notebooks and tests,
`trustqat.backends.register_backend('mylib', MyBackend)` works without packaging.

## Conformance

Run the conformance checks in your CI:

```python
from trustqat.testing import check_backend


def test_conformance(tmp_path):
    report = check_backend(MyBackend(), make_tiny_model, x, y, tmp_path)
```

The suite checks:

- the interface and API version
- the soft-ECE loss updates model parameters, and precision parameters when
  `trainable_precision` is declared
- costs are finite and non-negative
- save then load preserves the outputs
- `freeze_precision` really freezes
- `precision_summary` returns a sane summary
- for Keras backends, a 2-epoch run of the default trainer writes valid records and keeps
  the Pareto checkpoints

Keep the tiny model free of bit-width regularisers, so that the precision check isolates
the trust loss.

## API version

`trustqat.backends.API_VERSION` is `0.1`. While it is 0.x, a backend must match the minor
version, and `get_backend` refuses mismatches with a clear error.
