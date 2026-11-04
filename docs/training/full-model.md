# Train the complete selected model chain

Use this route when rebuilding MeTar's learned response model and target
representations from the prepared public datasets. It trains all selected
branches, including the downstream single-cell and proteomics encoders.
For a smaller change, start with [the training index](../training.md).

## Required inputs

Download the model/schema archive and `training_inputs.tar.gz`, then place
the [prepared public inputs](../data.md#place-separately-obtained-public-inputs).
The model archive supplies protocol and gene-order records. This workflow
learns new checkpoint parameters and target-vector values.

| Dataset or record | Required files under the artifact root |
|---|---|
| Gene-relation prior | `data/public_raw/gene_prior/process.txt` |
| PC3 response schema | `data/processed/closed/r210/pc3_landmark_response_978.tsv.gz` |
| PC3 target schema | `data/processed/closed/r210/pc3_metar_drugcell_scores_668.tsv.gz` |
| LINCS response profiles | `data/processed/lincs/large_data.parquet` |
| Known compound targets | `data/processed/lincs/compoundinfo_beta.txt` |
| Genome-wide graph | `data/processed/fullspace/pertkge_ppi_graph.csv` |
| Bulk splits | `data/processed/lincs/bulk_protocol_lock_v2.json`, `data/processed/fullspace/splits_v3.json` |
| Single-cell responses | Three `tahoe_*.h5ad` files for A549, SW480 and NCI-H460 |
| Single-cell labels/splits | `data/processed/tahoe/tahoe_dti.csv`, `data/processed/tahoe/sc_protocol_lock_v3_20260326.json` |
| Protein abundance | `data/processed/proteomics/proteomicsdb_data.csv` |
| Proteomics configuration/order | Method-lock JSON and aligned target table listed in the [proteomics guide](proteomics.md#prepare-and-train) |

The PC3 TSV files preserve the expected response/target order. The
gene-relation prior uses gene relationships only: drug nodes, drug–target
edges, expression-effect and `HAS_BINDING_TO` edges do not initialize it.
The scratch-input identities are recorded in
[`training_input_contract.json`](../../manifests/training_input_contract.json).

## Prepare a fresh workspace

```bash
python scripts/prepare_training.py --artifacts ../metar_artifacts --output ../metar-training --stage all --check-only
python scripts/prepare_training.py --artifacts ../metar_artifacts --output ../metar-training --stage all
```

Omitting `--stage` also selects `all`. Run without `--reuse-pretrained` to create
the upstream parameters from scratch. Preparation copies affected
inputs when symlinks are unavailable; add `--copy-inputs` to copy all inputs.
The input store and original code are kept separate from mutable training
outputs. See [training setup](../training.md) for the fallback notice and marker.

If no prepared bulk cache was present, build it in this workspace:

```bash
python scripts/prepare_bulk_data.py --package ../metar-training --device cpu
```

## Run the chain

```bash
metar train --stage all --workspace ../metar-training --device cuda:0
```

The stages run sequentially and stop on a nonzero result:

1. `gene-prior`: pretrain the selected gene-relation model.
2. `closed-bulk`: train the bulk response model and support-conditioned
   components for the 668-target setting.
3. `support-initializer`: learn the support-conditioned query initialization.
4. `cross-cell-adapter`: meta-train the cell-line adapter while keeping the
   bulk model fixed.
5. `full-space`: train the genome-wide response/target branch and export its
   target embeddings.
6. `single-cell`: train each of the three selected cell-line encoders against
   the newly exported fixed target embeddings.
7. `proteomics`: train the protein-abundance branch and save its preprocessing.

Use `--dry-run` to display the complete command sequence. Dependency checks
for downstream stages will fail before their upstream checkpoints exist;
check the first stage with `--stage gene-prior --check-only` initially.
Individual stage commands let you continue from completed upstream outputs
in the same workspace. Use a fresh workspace when repeating a completed stage.

## Your own training dataset

This command selects the paper's data contracts, architectures, task
partitions, seeds and hyperparameters. New datasets require the corresponding
schemas and protocol configuration.

For compatible replacements, follow the exact schemas in the
[cell-line](cell-line-adapter.md#input-schema),
[single-cell](single-cell.md) and [proteomics](proteomics.md) guides. Replace
workspace data, matching labels, compatibility copies and split records together. Rebuild
derived caches when their source data change; the released bulk cache
builder's source hashes and expected dataset counts must also be updated for a
new dataset, as described in the [target-space guide](target-space.md#input-and-dataset-changes).
Fit preprocessing on the training subset and retain it for validation, test
and future prediction.

A new feature dimension, target set or cell-line panel requires coordinated
source/protocol changes, including the selected count checks and held-out
evaluation definitions. Report such changes as a new experiment, with its own
dataset and evaluation protocol.

## Output files

All training logs, selected checkpoints and scientific outputs remain in the
new workspace. Interface copies are under `checkpoints/selected/`.
The full-space target vectors and proteomics preprocessor are generated
anew. Their release-compatible filenames are stable entry points; selection
records contain the actual epoch and validation results of this run.

For the complete training sequence, use Linux with an NVIDIA GPU and the
environment described in [installation](../installation.md#nvidia-gpu). CPU
execution is available for compatible individual stages and can require
substantially longer runtimes. Use each run's selection records for its exact
metrics and evaluation units. For released-score evaluation and figure
generation, see [paper reproduction](../reproduction.md).
