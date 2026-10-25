# HLF JSC benchmark

Jet substructure classification on the 16 high-level features of the OpenML dataset
`hls4ml_lhc_jets_hlf` (5 classes), with the HGQ2 backend. These are the experiments behind
Table I of the TrustQAT paper (CSCloud 2026). TrustQAT fine-tunes the five HGQ Pareto
checkpoints of the HGQ artifact
([hgq-ae](https://github.com/calad0i/hgq-ae), `models/jsc_openml/`) at 530, 1572, 2571,
7229 and 13285 EBOPs.

## Inputs

1. **Data splits.** Build them with:

   ```bash
   python -m benchmarks.jsc_hlf.data --cache /path/to/openml_cache.h5 --output dataset/jsc_hlf_openml_splits.npz
   ```

   `--cache` is an HDF5 file with `X` and `y`; if it does not exist it is downloaded from
   OpenML and written. The split is the one used by hgq-ae and TrustQAT-AE: 80/20
   train+val/test with `random_state=42`, a `StandardScaler` fitted on train+val, then a
   90/10 train/val split.
2. **HGQ checkpoints.** These are `models/jsc_openml/*.keras` from hgq-ae. They are not
   redistributed here.
3. Tell the configs where both are:

   ```bash
   export TRUSTQAT_JSC_DATA=/path/to/jsc_hlf_openml_splits.npz
   export TRUSTQAT_JSC_MODELS=/path/to/hgq-ae/models/jsc_openml
   ```

## Configs

`configs/trustqat/ebops<P>.yaml` holds one config per HGQ checkpoint. All five use the
settings of TrustQAT-AE (`hgq-ae/code/jsc_hlf/run_trust_hgq.py`):

- **Training:** warm start from the checkpoint, soft-ECE λ=2 with 15 bins, β=0, batch
  33200, 100 epochs, cosine learning rate 3e-5 → 1e-6, early stopping on `val_loss`
  (patience 15), seed 42.
- **Selection:** checkpoints are kept on the validation Pareto front (accuracy, ECE,
  EBOPs) subject to `val_accuracy >= HGQ checkpoint val_accuracy - 0.005`
  (`selection.accuracy_tolerance`). The deployed model is the front checkpoint with the
  lowest validation ECE (`selection.final`).
- **Export:** da4ml with a 1.0 ns clock and `latency_cutoff=1`.

## Running

```bash
for P in 530 1572 2571 7229 13285; do
  python -m benchmarks.jsc_hlf -c benchmarks/jsc_hlf/configs/trustqat/ebops$P.yaml -r train test select
done
python -m benchmarks.jsc_hlf -c benchmarks/jsc_hlf/configs/trustqat/ebops530.yaml -r export --rtl-samples 10000
```

The phases and their outputs are the same as for JEDI-linear (see
`benchmarks/common/driver.py`). `test_results.json` reports each checkpoint twice:

- **`test_raw_*`:** the model as trained. These are the numbers in the paper's Table I.
- **`test_*`:** after `trace_minmax` on train+val. This is the model that da4ml turns into
  RTL.

## Checking against TrustQAT-AE

```bash
python -m benchmarks.jsc_hlf.tools.check_against_ae --ae-root /path/to/hgq-ae \
    --data dataset/jsc_hlf_openml_splits.npz --ae-data /path/to/ae/openml_splits.npz
```

Results on an RTX 3090 (2026-10-25):

- **Data.** The splits are identical to the artifact's.
- **Re-evaluating the artifact's checkpoints.** All 10 models (5 HGQ, 5 TrustQAT)
  reproduce the artifact's test accuracy and ECE exactly, both as trained and after
  calibration.
- **Retraining from scratch with this repository.** 4 of the 5 points (530, 1572, 2571 and
  13285 EBOPs) select the same epoch and reproduce Table I exactly. At 7229 EBOPs the
  training curve matches the artifact to 1e-6 for 28 epochs and then diverges. The selected
  model then has 0.398% ECE after calibration, against 0.247% in the artifact and 0.453%
  for HGQ.
- **Why 7229 diverges.** JAX's default GPU reductions are not bit-deterministic. Two runs of
  this repository differ only in the last float32 bits, yet they also split at epoch 28,
  where a quantization boundary amplifies those bits. With
  `XLA_FLAGS=--xla_gpu_deterministic_ops=true`, repeated runs are bit-identical. The
  artifact was trained without this flag, so its exact numbers are reproduced only without
  it.

| EBOPs | HGQ acc / ECE (as trained) | TrustQAT acc / ECE (as trained) | HGQ ECE (calibrated) | TrustQAT ECE (calibrated) |
|---:|---:|---:|---:|---:|
| 530 | 73.18% / 5.555% | 73.15% / 2.393% | 5.818% | 2.587% |
| 1572 | 74.86% / 1.413% | 74.86% / 0.843% | 1.447% | 1.070% |
| 2571 | 75.59% / 2.087% | 75.65% / 0.482% | 1.855% | 0.488% |
| 7229 | 76.62% / 0.450% | 76.46% / 0.252% | 0.453% | 0.247% |
| 13285 | 76.87% / 0.293% | 76.85% / 0.171% | 0.330% | 0.220% |

These are the artifact's models; the as-trained columns are Table I. TrustQAT lowers ECE
at every point under both conventions, but by different amounts:

| Convention | Relative ECE reduction (range) | Median |
|---|---|---:|
| As trained (Table I) | 40.3% to 76.9% | 44.1% (Table I's rounded values give 44.0%) |
| Calibrated (the models in hardware) | 26.1% to 73.7% | 45.6% |
