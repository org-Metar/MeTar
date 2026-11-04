# Train only what your task needs

Most prediction tasks use the [pretrained models](usage.md) without training.
When training is needed, choose the smallest change below. The guides explain
both the released data protocol and what must change for a new dataset.

| Start here | Components trained | Reused components | Guide |
|---|---|---|---|
| Single-cell responses | Single-cell response encoder and score scale | Fixed target embeddings; no bulk or graph retraining | [Single-cell encoder](training/single-cell.md) |
| Protein-abundance responses | Protein-abundance encoder, score scale and configured label adapter | Stored bulk target embeddings; no bulk or graph retraining | [Proteomics encoder](training/proteomics.md) |
| A new collection of cell-line tasks | Support-conditioned adapter | Gene prior and bulk response model | [Cell-line adapter](training/cell-line-adapter.md) |
| Rebuild genome-wide target ranking | Bulk response encoder, GNN and target projection | Closed bulk response checkpoint | [Genome-wide branch](training/target-space.md) |
| Rebuild every selected model | Gene prior, bulk response model, adapter and downstream branches | Data schemas and split definitions only | [Full model](training/full-model.md) |

Applying the pretrained adapter to reference compounds from one new cell line
is an [inference task](usage.md), not a reason to repeat meta-training.
Combining predictions across cell lines also needs no extra model training.

## Install and place the inputs

Run commands from the repository root in the environment described in
[installation](installation.md):

```bash
uv pip install -e ".[training]"
```

The code checkout, artifact root and new training directory are separate:

```text
MeTar/                  # this code checkout
metar_artifacts/        # downloaded models and prepared input datasets
metar-training/         # a new, private training workspace
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
target embeddings into the new directory. It never copies the selected output
of the stage being trained. Sources, metadata and reused weights are private
files; large input datasets are read-only file links by default. Add
`--copy-inputs` to make private dataset copies, including on Windows where
symlinks are unavailable or when replacing inputs with your own data.

The destination must not exist and must be outside the code and artifact
directories. Preparation refuses to overwrite an existing workspace. Add
`--check-only` to inspect prerequisites without creating it. This checks file
presence, not the scientific suitability of the dataset.

Without `--reuse-pretrained`, no pretrained parameter values are copied.
A downstream stage cannot train until its upstream checkpoints or target
table are supplied. To generate them in the same workspace, prepare
`--stage all` without reuse and run the stages in order, as shown in the
[full-model guide](training/full-model.md).

Training retains historical scientific script basenames inside the private
workspace. On Windows, choose a short destination such as `C:/mt-train` and
enable long-path support if needed. Linux is the recommended environment
for the complete training chain.

The bulk cache builder also creates a temporary file symlink. Workspace
preparation's `--copy-inputs` option does not change that separate command.
Building a missing bulk cache on Windows therefore still requires symlink
permission (for example, an enabled Developer Mode); otherwise use Linux/WSL
or an already prepared compatible cache.

## Check, train and find the results

```bash
metar train --stage single-cell --cell-line A549 --workspace ../metar-training-a549 --device cuda:0 --check-only
metar train --stage single-cell --cell-line A549 --workspace ../metar-training-a549 --device cuda:0
```

`--check-only` verifies the selected stage's dependencies and workspace
isolation. `--dry-run` prints its command without checking inputs. Training
runs the selected paper configuration, rather than a hyperparameter search.
Use a new workspace for a new run: existing selected outputs are not replaced.
`--device cpu` is available for compatible stages, but the full training chain
can require substantial compute and has not been rerun end to end on CPU.

New interface checkpoints are saved in the workspace's `checkpoints/selected/`.
Detailed run directories, configuration records and metric tables remain in
the same workspace. Single-cell and proteomics runs also write under
`training_runs/`; their `--output-dir` must remain inside the workspace.
Compatibility filenames retain the release naming scheme; read the new run's
selection record for its actual selected epoch and performance.

The CLI protects the downloaded model directory from writes and rejects
symlinks or junctions that redirect mutable outputs outside the workspace.
When changing data, replace only private workspace files and update their
matching labels, feature definitions and split metadata as described in each
guide. Do not overwrite downloaded reference artifacts.

## Reproduce the paper's complete training protocol

The selected execution sequence is:

```text
gene-prior → closed-bulk → support-initializer → cross-cell-adapter
                                                       ↓
                                                  full-space
                                                       ├→ single-cell: A549 / SW480 / NCI-H460
                                                       └→ proteomics
```

This sequence rebuilds all reported branches. It does not mean every later
stage updates every preceding model: genome-wide training reuses the closed
bulk checkpoint and cell-split record; modality training reuses the exported
target embeddings. See [paper reproduction](reproduction.md) for evaluation
and computational figures.
