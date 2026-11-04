# Train a protein-abundance response encoder

Use this route to train the proteomics response branch while reusing the
learned bulk target embeddings. It does not retrain the graph encoder or bulk
response model. To predict with the existing model, follow the
[proteomics prediction guide](../prediction/proteomics.md).

## What is trained

The protein-abundance encoder, score scale and label adapter selected by the
method configuration are trained. The stored bulk target embeddings stay
fixed; the GNN and bulk encoder are not updated.

The runner supports identity, linear and residual label adapters, selected by
`label_adapter_mode` in the copied configuration. Identity leaves the target
vectors unchanged; other modes learn a modality-specific transformation of
the fixed vectors. This differs from retraining the graph-derived target model.

## Prepare and train

Required files under the [artifact root](../data.md) are:

| File | Purpose |
|---|---|
| `data/processed/proteomics/proteomicsdb_data.csv` | Labeled protein-abundance responses |
| `data/processed/fullspace/round259_target_embedding.csv` | Fixed bulk target vectors |
| `data/processed/fullspace/round259_target_embedding_aligned_to_proteomics_gene_order.csv` | Gene-order schema for the selected protocol |
| `code/original_runs/r259_261/proteomics_results/round261_proteomics_round259_embedding_method_lock.json` | Selected training/preprocessing settings |

```bash
python scripts/prepare_training.py --artifacts ../metar_artifacts --output ../metar-training-proteomics --stage proteomics --reuse-pretrained
metar train --stage proteomics --workspace ../metar-training-proteomics --device cuda:0
```

This stage does not require LINCS or Tahoe response inputs. See
[training setup](../training.md) for installation, private workspaces and
dependency checks.

## Data format and preprocessing

The CSV's first column is a row index. Required metadata columns are `drug`
and `target`; `target` contains a Python-style gene-symbol list such as
`['EGFR']`. Every other column must be a numeric abundance feature, not an
extra sample annotation. Rows without any mapped target are removed.
Compound names are normalized to lowercase.

The runner splits by compound, selects high-variance features using training
rows only, applies `log1p` when training values are nonnegative, and learns
centering and scaling from training rows. It saves this preprocessing for
subsequent prediction. Do not pre-standardize new data using statistics from
the complete dataset or mix arbitrary feature units with the selected recipe.

## Use your own training data

Prepare with `--copy-inputs` and replace only the workspace's
`data/processed/proteomics/proteomicsdb_data.csv`. Keep consistent feature
columns and labels matching the target table. The copied method-lock JSON
controls feature count, dropout, epochs, label-adapter mode and other settings;
edit the workspace copy if intentionally changing that configuration.

The selected evaluation recipe reserves up to 50 target genes and splits
compounds into training, validation and test subsets. With a very small custom
label set this may leave no supervised targets, so this is not a small-dataset
fine-tuning recipe.

## Find the outputs

`training_runs/proteomics/` contains the encoder/label-adapter checkpoint,
training history, selection record and per-compound evaluation.
`checkpoints/selected/round261_proteomics_epoch070.pth` is the interface copy;
its release filename does not fix the selected epoch of a new run.
The matching fitted preprocessor is
`data/processed/proteomics/round261_preprocessor.json`; its matching target
table is the aligned CSV in `data/processed/fullspace/`. Keep these three
artifacts together when predicting on your own abundance table.
