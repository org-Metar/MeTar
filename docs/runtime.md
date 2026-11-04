# Runtime reporting and device selection

After [installation](installation.md) and [asset setup](data.md), `metar predict`
and `metar train` report the selected setting, device, numbered steps, progress,
available metrics and completion status on **stderr**. Prediction summaries stay
as JSON on **stdout**, and rankings are written to `--output`.

For example, capture these streams separately from the repository root:

```bash
metar predict bulk-668 --input examples/inputs/bulk_query.npz --device auto --batch-size 256 --output outputs/bulk_668.tsv > run.json 2> runtime.log
```

Use `metar --quiet predict ...` or the scenario's `--quiet` option to suppress
runtime reports while retaining result JSON and errors. `METAR_PROGRESS=0`
suppresses these reports for scripts and their child processes as well.
Training scripts can still emit their existing scientific logs.

A training run has a setting header, real completion counts and short metric
lines. For example, the bulk model reports:

```text
MeTar | Train
  setting: closed-bulk
  device: cuda:0
[Step 1] Check closed-bulk workspace, dependencies and device
[####################] Epoch 1 training episodes: 240/240 (100%)
[Epoch 1/60] train_loss=8.576 | pseudo_val_Hit@10=0.2416 | pseudo_val_MeanAP=0.104
```

Early stopping can finish before the maximum epoch count. The selected-model
summary reports `epochs_run` and `best_epoch`; it does not pretend the remaining
epochs ran. Full result tables remain in the training workspace. Gene-prior
readout workers retain their individual progress logs, whose paths are printed
by the parent process.

## Choose a device

| Option | Behavior |
|---|---|
| `--device auto` | Default for model prediction and training: choose the first visible CUDA GPU when available, otherwise CPU. |
| `--device cpu` | Run on CPU, even when an accelerator is available. |
| `--device cuda:N` | Require visible CUDA device `N`; fail with an explanation if CUDA or that index is unavailable. |

An explicit CUDA request no longer falls back silently to CPU. A CPU-only
PyTorch installation needs a compatible CUDA-enabled wheel before it can use
an NVIDIA GPU; follow the [installation guide](installation.md#nvidia-gpu).
On supported Apple systems, `--device mps` requests MPS explicitly. Device
availability alone does not establish that every training stage supports it.
Consensus aggregates existing score tables and has no device option.

CUDA indices are relative to `CUDA_VISIBLE_DEVICES`. Set it **before** starting
the command. For example, on Linux:

```bash
CUDA_VISIBLE_DEVICES=2 metar predict bulk-668 --input examples/inputs/bulk_query.npz --device cuda:0 --output outputs/bulk_gpu.tsv
```

Here GPU `2` is exposed as the process's first visible device, `cuda:0`.
In PowerShell, set `$env:CUDA_VISIBLE_DEVICES = '2'` before the same `metar`
command. The runtime report records the resolved device; it does not change
GPU visibility.

## Read progress and retain run records

Prediction reports query/feature/target counts and batch completion.
Adapted prediction also reports support encoding; single-cell prediction
reports completed groups and cells used. Consensus reports qualifying and
skipped queries. Training reports epochs or episodes with the loss and
validation metrics already computed by that stage. Reports are throttled;
non-interactive logs contain periodic lines instead of an updating terminal
bar. Reporting does not add evaluation passes or change model selection.

Training retains its original detailed scientific logs on stdout. To keep
only the concise live progress visible while saving those details:

```bash
metar train --stage closed-bulk --workspace ../metar-training --device auto > training.details.log
```

Use `> training.details.log 2> training.progress.log` to retain both streams
without live terminal output.

Install the training extra in the active environment before running a
[prepared private training workspace](training.md):

```bash
uv pip install -e ".[training]"
metar train --stage single-cell --cell-line A549 --workspace ../metar-training-a549 --device cuda:0
```

Keep the exact command, runtime log, Python/PyTorch versions and matching model,
target-table and preprocessing files. For training, retain the workspace's
configuration, metric tables and selection records. Prediction summaries report
execution counts, not accuracy. Device and batch-size changes can produce small
floating-point differences, especially near tied ranks; see
[numerical reproduction limits](reproduction.md#numerical-and-preparation-limits).

## Resolve a failed run

| Report | What to check |
|---|---|
| CUDA requested with a CPU-only build or no usable driver/device | Check `python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"` in the same environment. Install the matching CUDA wheel/driver, or choose `--device cpu` or `auto`. |
| CUDA index unavailable | Check `CUDA_VISIBLE_DEVICES` and use the index within the visible devices. |
| Invalid batch size, empty queries, nonfinite features or mismatched targets | Correct the named input or option. Batch sizes must be positive; queries and target tables must be nonempty and aligned. Follow the scenario's [input guide](usage.md). |
| Empty or unannotated support | Supply separate support responses with at least one valid target per row; invalid rows are rejected rather than removed. See [cross-cell prediction](prediction/cross-cell.md). |
| Missing training inputs or optional modules | Install `.[training]` in the executing environment and check the selected stage's inputs with `--check-only`; see [training setup](training.md). |

Lower `--batch-size` for bulk, full-space, adapted cross-cell or proteomics
prediction when device memory is insufficient. Single-cell loading can still
densify the full input before `--max-cells` sampling; see
[memory troubleshooting](troubleshooting.md#memory-and-result-comparison).
`doctor`, `--check-only` and `--dry-run` inspect readiness or commands; they do
not establish successful inference, training or scientific reproduction.
