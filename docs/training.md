# Training workflows

Most prediction tasks use the [pretrained models](usage.md) without training.
When training is needed, choose the components to update below. The guides
explain both the released data protocol and changes required for a new dataset.

| Start here | Components trained | Reused components | Guide |
|---|---|---|---|
| Single-cell responses | Single-cell response encoder and score scale | Fixed target embeddings; no bulk or graph retraining | [Single-cell encoder](training/single-cell.md) |
| Protein-abundance responses | Protein-abundance encoder, score scale and configured label adapter | Stored bulk target embeddings; no bulk or graph retraining | [Proteomics encoder](training/proteomics.md) |
| A new collection of cell-line tasks | Support-conditioned adapter | Gene prior and bulk response model | [Cell-line adapter](training/cell-line-adapter.md) |
| Rebuild genome-wide target ranking | Bulk response encoder, GNN and target projection | Closed bulk response checkpoint | [Genome-wide branch](training/target-space.md) |
| Rebuild every selected model | Gene prior, bulk response model, adapter and downstream branches | Data schemas and split definitions only | [Full model](training/full-model.md) |

Applying the pretrained adapter to reference compounds from one new cell line
uses the [inference workflow](usage.md).
Combining predictions across cell lines also needs no extra model training.

## Install and place the inputs

Run commands from the repository root in the environment described in
[installation](installation.md). For the NVIDIA GPU commands below, add the
training extra while retaining the CUDA profile:

```bash
uv sync --locked --no-default-groups --group cu126 --extra training
```

For CPU stages, use `uv sync --locked --extra training`. See
[optional dependencies](installation.md#optional-dependencies) to combine extras
with the selected profile.

The code checkout, artifact root and new training directory are separate:

```text
MeTar/                  # this code checkout
metar_artifacts/        # downloaded models and prepared input datasets
metar-training/         # a new training workspace
```

Download the relevant [release archives](data.md#download-what-you-need), then
[place separately obtained public inputs](data.md#place-separately-obtained-public-inputs).
Each setting guide lists its own required files. A single-cell run does not
require proteomics data, and a proteomics run does not require LINCS or Tahoe.

## Prepare a workspace for one stage

For example, train an A549 single-cell encoder while retaining the published
target embeddings:

```bash
python scripts/prepare_training.py --artifacts ../metar_artifacts --output ../metar-training-a549 --stage single-cell --cell-line A549 --reuse-pretrained
```

`--stage` chooses the sources, input data and protocol files to prepare.
`--reuse-pretrained` copies only that stage's required upstream weights or
target embeddings into the new directory. The selected output of the stage
being trained is created by the new run. Sources, metadata and reused weights
are workspace copies; large input datasets are file links by default. Treat
linked inputs as read-only. If the system does not permit input symlinks,
preparation copies the affected inputs, prints
one notice and records `input_mode: copy_fallback` in `training_workspace.json`.
Add `--copy-inputs` to copy all inputs, including when replacing them with your
own data. Compatibility aliases are also copied when links are unavailable;
keep these copies aligned when replacing inputs.

The destination must not exist and must be outside the code and artifact
directories. Preparation refuses to overwrite an existing workspace. Add
`--check-only` to inspect prerequisite files without creating it. Follow the
stage's input guide for feature definitions, preparation and split requirements.

Without `--reuse-pretrained`, no pretrained parameter values are copied.
A downstream stage cannot train until its upstream checkpoints or target
table are supplied. To generate them in the same workspace, prepare
`--stage all` without reuse and run the stages in order, as shown in the
[full-model guide](training/full-model.md).

Training uses the source scripts' basenames inside the workspace. On Windows,
choose a short destination such as `C:/mt-train` and
enable long-path support if needed. Linux is the recommended environment
for the complete training chain.

The bulk cache builder creates a temporary graph link, using a temporary copy
when links are unavailable. It removes only the compatibility file it created
and preserves a matching pre-existing file.

### Compatibility input paths

Let `WORKSPACE` denote the prepared training directory. Its generated
compatibility root is `COMPAT_ROOT = WORKSPACE/compat/home/lwn/claude/`.
Training sources use the following paths relative to `COMPAT_ROOT`:

| Input | Path relative to `COMPAT_ROOT` |
|---|---|
| Bulk profiles and metadata | `bulk/data/` |
| Cell-line split records | `metar/` |
| Single-cell profiles and labels | `sc/data/` |
| Single-cell protocol | `manuscript/write/sc_agg_upgrade_20260321/derived_tables/` |

These paths are created inside `WORKSPACE` to match the training sources. When
replacing copied inputs, update the canonical files and their corresponding
compatibility copies together.

## Check, train and find the results

```bash
metar train --stage single-cell --cell-line A549 --workspace ../metar-training-a549 --device cuda:0 --check-only
metar train --stage single-cell --cell-line A549 --workspace ../metar-training-a549 --device cuda:0
```

`--check-only` verifies the selected stage's input files and workspace
isolation. Installed optional modules and the requested device are checked
before actual training starts. `--dry-run` prints its command without checking
inputs. Training runs the selected paper configuration.
Use a new workspace for a new run: existing selected outputs are not replaced.
`--device cpu` is available for compatible stages. The complete training
workflow targets Linux and NVIDIA GPUs; plan CPU compute requirements for each
individual stage before running the full sequence.

New interface checkpoints are saved in the workspace's `checkpoints/selected/`.
Detailed run directories, configuration records and metric tables remain in
the same workspace. Single-cell and proteomics runs also write under
`training_runs/`; their `--output-dir` must remain inside the workspace.
Interface filenames use the release naming scheme; read the new run's
selection record for its actual selected epoch and performance.

The CLI protects the downloaded model directory from writes and rejects
symlinks or junctions that redirect mutable outputs outside the workspace.
When changing data, replace workspace copies and update their
matching labels, feature definitions and split metadata as described in each
guide. Keep downloaded reference artifacts in the separate input store.

## Reproduce the paper's complete training protocol

The selected execution sequence is:

```text
gene-prior → closed-bulk → support-initializer → cross-cell-adapter
                                                       ↓
                                                  full-space
                                                       ├→ single-cell: A549 / SW480 / NCI-H460
                                                       └→ proteomics
```

This sequence rebuilds all reported branches. Genome-wide training reuses the
closed bulk checkpoint and cell-split record; modality training reuses the exported
target embeddings. See [paper reproduction](reproduction.md) for evaluation
and computational figures.
