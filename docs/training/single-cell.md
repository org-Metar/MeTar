# Train a single-cell response encoder

Use this route to train the single-cell response branch while keeping the
learned target space fixed. To use an existing checkpoint, follow the
[single-cell prediction guide](../prediction/single-cell.md).

## What is trained

The response encoder and score scale are trained. All 16,583 target embeddings
remain fixed. The selected architecture expects 128-coordinate cell inputs
and has configurations for A549, SW480 and NCI-H460.

## Prepare and train

Obtain the model archive and these prepared public files under your
[artifact root](../data.md). LINCS, proteomics and gene-prior training inputs
are not needed for this stage.

| File | Purpose |
|---|---|
| `data/processed/fullspace/round259_target_embedding.csv` | Fixed 16,583 × 256 target table |
| `data/processed/tahoe/tahoe_a549.h5ad` | Prepared A549 cell responses |
| `data/processed/tahoe/tahoe_dti.csv` | Compound–target training labels |
| `data/processed/tahoe/sc_protocol_lock_v3_20260326.json` | Compound and target splits |

```bash
python scripts/prepare_training.py --artifacts ../metar_artifacts --output ../metar-training-a549 --stage single-cell --cell-line A549 --reuse-pretrained
metar train --stage single-cell --cell-line A549 --workspace ../metar-training-a549 --device cuda:0
```

Replace `A549` with `SW480` or `NCI-H460` and supply `tahoe_sw480.h5ad` or
`tahoe_nci_h460.h5ad`. Each run gets its own workspace. For workspace isolation,
Windows path considerations and checks, see [training setup](../training.md).

## Data format

The H5AD `X` matrix has one cell per row and exactly 128 numeric features per
cell. `obs['drug']` identifies the compound; the reader trims whitespace and
converts names to lowercase. The coordinates must follow the expected Tahoe
feature construction, including its projection basis and gene order. The saved
[projection and raw-data recipe](../usage.md#prepare-x128-from-raw-tahoe-plates)
are provided for that purpose.

The label CSV is read with its first column as the row index. It must contain
`drug` and `id` columns; `id` is a gene symbol or serialized list such as
`['EGFR', 'ERBB2']`. Gene names must match the fixed target table.

The protocol JSON defines `fit_drugs`, `val_drugs`, `test_drugs`,
`train_drugs_outer`, `seen_targets`, `holdout_targets`, `exclusive_drugs` and
`mixed_drugs`. It also specifies `target_column: "id"` and
`protocol_tag: "v3_corrected_gene_targets_fixed_cells"`. Only training compounds
and supervised targets contribute training labels; separate compound splits
and held-out targets support evaluation.

## Use your own training data

Prepare with `--copy-inputs`, then replace the selected H5AD and label CSV
inside the workspace. Update its protocol JSON with new compound and target
memberships for the new dataset. Keep training, validation and test compounds
disjoint, and keep held-out target
labels out of the training target list. Update the single-cell data and protocol
[compatibility copies](../training.md#compatibility-input-paths) together with
the canonical files when inputs are copied.

This interface retains the 128-feature architecture, 16,583-target table and
three selected cell-line configurations. Another feature dimension, candidate
space or cell-line configuration requires changes to the model and protocol
configuration in the workspace sources.

## Find the outputs

The run is written under `training_runs/single-cell_A549/` (or the requested
output directory within the workspace), with encoder checkpoint, training history,
validation/test metrics and per-compound results below `train_A549/`.
The interface checkpoint is copied to
`checkpoints/selected/round260_A549_seed2043.pth`.
SW480 and NCI-H460 filenames use their selected seeds, recorded in the run.
Use the new checkpoint with the same target table and coordinate system.
