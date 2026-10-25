# JEDI-linear benchmark

JEDI-linear jet tagging on the HLS4ML LHC jet dataset (5 classes, 150-constituent jets),
with the HGQ2 backend. These are the JEDI-linear experiments of the TrustQAT paper
(CSCloud 2026). The model and data code come from JEDI-linear and the
[TrustQAT-AE](https://github.com/ecs-bristol/TrustQAT-AE) artifact; see `NOTICE`.

## Inputs

1. **Dataset.** Run `bash benchmarks/jedi_linear/prepare_dataset.sh` in a scratch
   directory. It writes `dataset/150c-train.h5` and `dataset/150c-test.h5`.
2. **Official HGQ checkpoints.** The TrustQAT runs start from these. Get them from
   TrustQAT-AE (`tar -xf official_models.tar.gz`).
3. Tell the configs where both are:

   ```bash
   export TRUSTQAT_JEDI_DATA=/path/to/dataset
   export TRUSTQAT_JEDI_MODELS=/path/to/official_models
   ```

   Both default to `dataset` and `official_models` relative to the working directory.

## Configs

| Directory | Contents |
|---|---|
| `configs/trustqat/` | The 20 TrustQAT runs (`ece-grid-*` in TrustQAT-AE): warm start from the official HGQ checkpoint, soft-ECE λ=2, 15 bins, β=0, 100 epochs, lr 3e-5 cosine, early stopping on `val_loss` |
| `configs/hgq/` | The 20 HGQ baseline sweeps (`sweep-*`): from scratch, 7000 epochs, β schedule 2e-8 → 3e-6, no trust objective |
| `configs/examples/` | `nonperminv-n16-f3-ece-beta.yaml`: TrustQAT with a non-zero β (README example of TrustQAT-AE) |

File names: `perminv` is permutation-invariant JEDI-linear (`gnn_uq1`), `nonperminv` is
order-sensitive (`gnn`); `n<N>` is the particle count and `f3`/`f16` the input features.
The paper reports 10 permutation-invariant and 8 order-sensitive pairs (no order-sensitive
N=8), so two of the 20 configs are not in its tables.

To regenerate the configs from TrustQAT-AE:
`python benchmarks/jedi_linear/tools/convert_ae_configs.py <TrustQAT-AE>/configs benchmarks/jedi_linear/configs`.

## Running

From the repository root, in the `trustqat` env:

```bash
CFG=benchmarks/jedi_linear/configs/trustqat/perminv-n16-f3.yaml
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.jedi_linear -c $CFG -r train test select
python -m benchmarks.jedi_linear -c $CFG -r export --rtl-samples 10000   # da4ml RTL + Verilator check
```

`--set key=value` overrides any config entry, e.g.
`--set train.epochs=5 save_path=runs/smoke`.

| Phase | Output in `save_path` |
|---|---|
| `train` | `records.jsonl`, `ckpts/` (Pareto front over val accuracy, val ECE, EBOPs), `pareto.json`, `selection.json`, `history.json`, `last.keras`, `init_metrics.json` |
| `test` | `models/` (inference-ready, after `trace_minmax`), `test_results.json` / `.csv` (accuracy, ECE, MCE, NLL, Brier, EBOPs) |
| `select` | `selected.json`: the validation Pareto front joined with the test results, plus `chosen`, the front checkpoint with the lowest validation ECE (`selection.final`), which is the model the artifact deployed |
| `export` | `da4ml/<checkpoint>/` Verilog projects, `export_results.json` (da4ml LUT/FF estimates, RTL accuracy and mismatches) |

Synthesis: `cd <save_path>/da4ml/<checkpoint> && vivado -mode batch -source build_vivado_prj.tcl`.
The paper used Vivado 2024.2 with a 2.0 ns clock on xcvu13p-flga2577-2-e.

## Differences from TrustQAT-AE

The training maths are the same: the loss, metric, binning, data split, input float16 cast,
LR schedule, β schedule and model all match. What differs:

- **Libraries.** Upstream `HGQ2==0.2.0`, `keras==3.15.1`, `jax==0.12`, `da4ml==0.6.0`,
  Python 3.12, instead of the ECE fork of HGQ2 with keras 3.11.3, jax 0.6.2 and Python 3.11.
- **Checkpoint retention.** Checkpoints are kept by standard Pareto dominance, as described
  in the paper. HGQ's `ParetoFront` callback, used by the artifact, only removes an old
  checkpoint when the new one is strictly better in every objective, so it can keep a few
  weakly dominated ones. Every epoch is in `records.jsonl` either way.
- **Checkpoint names.** Files are named `epoch=NNNN.keras`; the metrics live in
  `records.jsonl` rather than in the file name.

## Checking against TrustQAT-AE

`tools/check_against_ae.py` re-evaluates the artifact's 20 selected TrustQAT checkpoints
and their HGQ baselines with this repository, then compares them with the artifact's
results table:

```bash
python -m benchmarks.jedi_linear.tools.check_against_ae --ae-root /path/to/TrustQAT-AE -o runs/ae_check.csv
```

Result on an RTX 3090 (2026-10-25):

- **Exact matches.** All 20 HGQ baselines and the 10 order-sensitive TrustQAT checkpoints
  reproduce the artifact's test accuracy and ECE (|Δ| < 1e-5).
- **Small differences.** The 10 permutation-invariant TrustQAT checkpoints differ by at
  most 0.04 pp in accuracy and 0.04 pp in ECE. The artifact evaluated these checkpoints as
  saved during training (`load_mode=checkpoint_model`, no `trace_minmax`). This repository
  always recalibrates on the training set before testing, as the artifact did for every
  other model.

The paper's summary statistics are unchanged:

- 15/18 pairs improve
- median relative ECE reduction: 38.9% overall, 54.0% for the order-sensitive models
- median |Δacc|: 0.12 pp

The only exception is the permutation-invariant median, which becomes 25.2% (27.9% in the
paper).
