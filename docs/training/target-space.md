# Rebuild the genome-wide ranking branch

The pretrained genome-wide model can already rank 16,583 candidate genes,
including genes without supplied drug–target training labels. Use the
[genome-wide prediction guide](../usage.md) for that application. This page
is for retraining its response/target model with the selected training
protocol.

## What is trained

`full-space` initializes its query encoder from the closed bulk model and
trains the query encoder, graph neural network, target projection and score
scale. Known associations for 668 targets supervise the shared model, while
the gene graph supplies representations for the full candidate space.
This is a broader optimization than adding a new assay encoder: the target
embeddings are updated and exported again.

The selected recipe keeps 16,583 genes and 668 supervised targets. Ranking
unlabeled candidates does not require adding them as new output classes.
Changing the graph to an arbitrary new gene set is a model/protocol change,
not a file that can be supplied through a `--target-space` training option.

## Prepare and run

Supply the bulk data and split files listed in the
[cell-line training guide](cell-line-adapter.md#prepare-and-run), the closed
bulk checkpoint from the model archive, and either a prepared
`data/processed/fullspace/h07_prepared_data.pt` cache or the inputs needed
to build it.

```bash
python scripts/prepare_training.py --artifacts ../metar_artifacts --output ../metar-training-fullspace --stage full-space --reuse-pretrained
```

The helper also copies the selected cell-split record from the code checkout.
The cross-cell adapter need not be trained again for this isolated stage.
If a cache was not available, create it in the new workspace before training:

```bash
python scripts/prepare_bulk_data.py --package ../metar-training-fullspace --device cpu
```

The cache builder uses a temporary graph link, falling back to a temporary
copy when symlinks are unavailable. Workspace preparation also copies affected
inputs when link permissions are unavailable; see [training setup](../training.md).

Then train the branch:

```bash
metar train --stage full-space --workspace ../metar-training-fullspace --device cuda:0
```

The helper copies the closed bulk checkpoint as an upstream input and leaves
the final genome-wide checkpoint and target-vector export absent. No
single-cell or proteomics dataset is required.

## Input and dataset changes

The prepared cache carries response matrices, gene order, graph adjacency,
training-label indices and compound/target splits. It is built from the
LINCS Parquet table, compound–target TSV, graph CSV and protocol JSON files
described in the [bulk input schema](cell-line-adapter.md#input-schema).
The source validates the selected 16,583-candidate and 668-label counts.

For new compatible bulk data, use `--copy-inputs`, replace the workspace's
profile/label inputs and all corresponding private compatibility copies,
then update the split records. A cache built for another dataset must not
be retained. The released `prepare_bulk_data.py` verifies the published
source-file hashes and audit counts: it will reject a replacement dataset.
To build a new dataset's cache, update those source/audit contracts in a
private copy of the builder, or produce an equivalent cache with the
expected fields and ordering. Feature order and input normalization must
remain compatible with the reused response checkpoint.

For a genuinely new graph, label set, feature construction or cell-line
protocol, adapt the training-source checks and metadata consistently. The
selected CLI does not provide an automatic arbitrary-graph retraining
pipeline. A full rebuild of the upstream response model is described in
the [full-model guide](full-model.md).

## Outputs

The workspace receives:

- `checkpoints/selected/round259_e13_bce001.pth`: the newly trained model.
- `data/processed/fullspace/round259_target_embedding.csv`: the new target
  embedding export.
- Private run records containing training history, split definitions,
  validation selection and per-profile target-ranking results.

Use the new response checkpoint with its own exported target table.
If downstream single-cell or proteomics models are to use this new target
geometry, retrain their encoders against these exported vectors; an old
modality encoder was trained against the old vectors.
