# Meta-train the cell-line adapter

Use this route to learn support-conditioned adjustment across a collection
of cell-line tasks while keeping the bulk response model fixed. If you only
want predictions in a new cell line using its labeled reference compounds,
load the pretrained adapter through the [cross-cell prediction guide](../usage.md).
That operation does not require this training stage.

## What changes

`cross-cell-adapter` trains the support-conditioned query adapter through
cell-line episodes. Each episode contains reference compounds with known
targets and separate query compounds from the same cell line. The bulk
response model and gene-prior model are reused and frozen. The selected
configuration initializes the new adapter from the released support
initializer; it does not resume the final released cross-cell adapter.

For a complete rebuild, `support-initializer` trains that initialization first.
The [full-model workflow](full-model.md) runs both stages in order.

## Prepare and run

Place the model archive and the following prepared inputs under the
[artifact root](../data.md):

| Input | Purpose |
|---|---|
| `data/processed/lincs/large_data.parquet` | Bulk response profiles with compound and cell-line identities |
| `data/processed/lincs/compoundinfo_beta.txt` | Known compound–target labels |
| `data/processed/fullspace/pertkge_ppi_graph.csv` | Gene graph used by data preparation |
| `data/processed/lincs/bulk_protocol_lock_v2.json` | Compound and target partitions |
| `data/processed/fullspace/splits_v3.json` | Training and evaluation cell-line partitions |
| Selected gene-prior, closed-bulk and support-initializer checkpoints | Frozen upstream models and adapter initialization |

```bash
python scripts/prepare_training.py --artifacts ../metar_artifacts --output ../metar-training-adapter --stage cross-cell-adapter --reuse-pretrained
metar train --stage cross-cell-adapter --workspace ../metar-training-adapter --device cuda:0
```

Preparation copies only the three required upstream checkpoints into private
runtime paths. The final cross-cell checkpoint is left absent so training
creates a new result. Tahoe and proteomics inputs are not required.
Run with `--check-only` first to inspect dependencies.

## Input schema

`large_data.parquet` has one response profile per row. Required metadata are
`drug`, `cell_line` and `method`; the selected reader keeps `method == 0`.
The remaining numeric expression columns form the response vector and must
contain the expected 978 landmark features in the original order. The
`target` column, if present, is replaced using the compound metadata.
Do not put unrelated numeric metadata among the expression columns.

`compoundinfo_beta.txt` is a tab-separated table with `pert_id` and `target`
columns. Each row associates a compound identifier with a target gene;
multiple rows can provide multiple targets for one compound. `pert_id`
must match `drug` in the response table. Labels must map to the 668 targets
expected by the reused bulk model.

`pertkge_ppi_graph.csv` contains `start` and `end` gene-symbol columns. The
bulk protocol JSON defines compound partitions under `drug_split`, including
`inner_train_drugs`, `val_drugs` and `test_drugs`, and target holdouts.
`splits_v3.json` defines `train_celllines` and `test_celllines`. Training
standardization is estimated from the training drug–cell means.

The selected episodic configuration samples 32 reference compounds and 32
query compounds per training episode, with drug-disjoint sets. Cell lines
need enough distinct labeled compounds to form these tasks. Validation uses
separate cell-line/reference/query partitions; test labels do not form
training episodes.

## Use a different collection of cell-line tasks

Prepare with `--copy-inputs` and replace the private LINCS profile/label
files. Update the workspace's compound and cell-line partitions together.
Keep feature order, preprocessing and the 668-target identities compatible
with the frozen response model. A differently normalized 978-column table
is not automatically compatible just because its shape matches.

The execution layout contains private compatibility copies of the bulk
inputs under `compat/home/lwn/claude/bulk/data/` and the split JSON under
`compat/home/lwn/claude/metar/`. When using copied inputs, update those
copies together with the canonical files. The copied bulk protocol also
appears in the `manifests/` directories of its compatibility preparation
trees. Keep all of these protocol copies consistent.

The selected sources retain the paper's cell-selection rules, including
PC3, A375 and MCF7 in training. A completely new panel that excludes those
cell lines, a changed target set or a changed feature dimension requires
corresponding changes to the selected training sources and protocol checks.
The public CLI selects this documented recipe; it does not expose arbitrary
task configuration or a generic few-shot training interface.

## Outputs

The new interface checkpoint is
`checkpoints/selected/round252_adapter_seed20260524.pth` inside the workspace.
The private cross-cell run directory contains the checkpoint, episode
training history, cell splits, per-cell/per-compound metrics and selection
record. These records identify the actual selected epoch and model.
For prediction, pair the new adapter with the same frozen bulk model and
normalization used for its training.

To rebuild only the support initializer, prepare a separate workspace with
`--stage support-initializer --reuse-pretrained`, then train with
`metar train --stage support-initializer --workspace PATH --device cuda:0`.
It reuses the gene prior and closed bulk model, while creating a fresh
initializer checkpoint.
