# Bulk transcriptomic prediction

Use `bulk-668` to rank the 668 supervised targets from a compound's prepared
bulk transcriptomic response. The pretrained model runs directly, without
query target labels or retraining. For a new cell line with reference
compounds, see [cross-cell prediction](cross-cell.md).

## Download and run

Complete [installation](../installation.md), run commands from the repository
root, and download the selected models:

```bash
python scripts/download_data.py --archive models_and_representations.tar.gz --download --directory ../metar-downloads --package ../metar_artifacts
```

Set [`METAR_ARTIFACT_ROOT`](../data.md#set-the-artifact-root) to the extracted
directory. The default checkpoint is
`checkpoints/selected/round246_whole_model_checkpoint.pth`; it also stores the
target names and order.

```bash
metar predict bulk-668 --input examples/inputs/bulk_query.npz --cell-line HELA --device cpu --output outputs/bulk_668.tsv
```

This bundled input contains two real HELA responses already in the trained
978-feature representation. It demonstrates prediction over 668 targets;
it is not an evaluation restricted to the training cell lines. No result-data
download is needed. See the [example inputs](../../examples/README.md) for
their provenance and selection. `--checkpoint PATH` selects a compatible model
explicitly; batch size defaults to 256.

## Optional: run the PDF-3 application query

The release supplies the prepared PDF-3 query used in the manuscript in
`result_data.tar.gz`. Try it without obtaining an independent assay:

```bash
python scripts/download_data.py --archive models_and_representations.tar.gz --archive result_data.tar.gz --download --directory ../metar-downloads --package ../metar_artifacts
python scripts/prepare_pdf3.py --package ../metar_artifacts --output ../metar-example/pdf3.csv
metar predict bulk-668 --input ../metar-example/pdf3.csv --id-column query_id --cell-line JHH7 --top-k 20 --output ../metar-example/pdf3_targets.tsv
```

The preparer preserves landmark order and refuses to overwrite output. Its
`--from-raw` option starts from processed FPKM in `application_inputs.tar.gz`,
not raw sequencing reads. See [reproduction](../reproduction.md) for the
manuscript analysis.

## Replace the example with your data

For a named feature table of your own, replace the example with:

```bash
metar predict bulk-668 --input inputs/bulk_queries.tsv --id-column query_id --feature-order inputs/landmark_order.txt --cell-line PC3 --device cpu --output outputs/bulk_668.tsv
```

Replace the input, feature-order file and cell-line name. `--cell-line` records
metadata; it does not change the checkpoint or prepare the response.

Use one prepared response per row with a stable, unique query identifier. A
TSV has this structure; ellipses denote omitted columns, not literal input:

```text
query_id	GENE_1	GENE_2	...	GENE_978
compound_a	0.12	-0.34	...	0.56
compound_b	-0.20	0.45	...	0.08
```

Replace `GENE_1` through `GENE_978` with the actual landmark names. These numbers
illustrate the layout, not a usable biological query.

| Item | Requirement |
|---|---|
| Rows | One response per query; include dose/condition in the ID if these are distinct queries |
| Features | Exactly the training 978 landmarks, with finite numeric values |
| Identifiers | Same identifier type and spelling as the training schema; symbols and Ensembl IDs are not automatically interconverted |
| Order | Use `--feature-order` to select/reorder named columns to training order |
| Labels | Query target labels are not required |
| Metadata | Keep numeric metadata such as dose/time out of the feature matrix |

A feature-order text file has one name per line and no header. A schema table
also works; select its column with `--feature-column`. The original
`geneinfo_beta.txt` is filtered to `feature_space == landmark`, preserving row
order. Without an explicit schema, the CLI uses the default
`data/application_inputs/pdf3/geneinfo_beta.txt` if present. If neither exists,
it takes all numeric columns except the ID in their existing order and requires
978; that fallback cannot detect the wrong biological order.

CSV, TSV, `.csv.gz`, `.tsv.gz` and Parquet are supported. `.npy` accepts an
`N x 978` matrix or length-978 vector; `.npz` contains `x` or `features`, plus
optional `query_ids`. Unnamed arrays must already have the correct feature
order: a feature-order file cannot reorder them.

### Match the response preparation

The input is a prepared response, not raw counts, FPKM or an arbitrary
normalized expression matrix. Ordinary LINCS-style bulk prediction needs the
training drug-cell-mean standardization. **The prediction CLI does not compute
this standardization.** Its fitted mean/std state is not separately released
in the pretrained archive; a raw assay must be prepared externally before
running the command.

The existing preparation can be traced in
[`build_round243_tables`](../../training/leakage_safe_support/leakage_safe_support.py#L553),
which constructs the selected training/support/query splits, and in
[`build_drugcell_table`](../../training/support_query_shift/support_query_shift.py#L178)
and [`standardize_tables`](../../training/support_query_shift/support_query_shift.py#L245):

1. Use the matched protocol and training cell/compound splits. For the reference
   training table, average response profiles within each compound/cell-line
   group, retaining the same ordered 978 features.
2. Calculate one mean and population standard deviation per feature from the
   `meta_train` drug-cell means only. The implementation replaces a standard
   deviation below `1e-6` with `1.0`.
3. Apply `(response - training_mean) / training_std` to prepared queries and
   support responses, using those same training-derived values. Do not estimate
   them again from a query or support set.

For the selected pipeline's context and split selection, see
[`load_context`](../../training/closed_bulk/train_closed_bulk.py#L136) and
[`build_round252_context`](../../scripts/benchmark_adaptation.py#L47). These are
the existing data-preparation implementations, not a raw-assay prediction
command. Reconstructing their inputs requires the matched prepared public
datasets described in [data setup](../data.md); the small bundled examples
already include this transformation. Query target labels are not needed for
prediction.

The PDF-3 application follows a separate documented procedure: average treated
and control FPKM separately, compute
`log2((treated_mean_FPKM + 1) / (control_mean_FPKM + 1))`, then align landmarks.
This does not establish a general raw-log2FC rule for other assays. The
[genome-wide branch](genome-wide.md) has a different preparation contract,
despite also accepting 978 features.

## Check inputs and read predictions

```bash
metar doctor --scenario bulk-668 --input inputs/bulk_queries.tsv --id-column query_id --feature-order inputs/landmark_order.txt
```

Doctor checks files, dimensions, IDs and finite values without inference. Shape
alone cannot establish compatible preprocessing. Output defaults to all 668
candidates with `target`, `score` and descending `rank`. `rank=1` is best;
scores are not binding probabilities. Use `--top-k 20` for a shortlist or
`--top-k 0` for [consensus](consensus.md). See the shared
[output reference](../usage.md#read-outputs-and-check-inputs).
