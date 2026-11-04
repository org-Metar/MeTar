# Consensus across cell lines

Combine predictions for the same compound across multiple cell lines using
post-prediction aggregation of score tables. First generate complete
predictions for the same candidate set in every cell line.

## Run the bundled score example

The checkout includes complete direct-prediction scores over 668 targets for
the same two query compounds in HELA and HCC515. Combine them without
downloading model weights:

```bash
metar predict consensus --input examples/inputs/consensus/hela_scores.tsv --input examples/inputs/consensus/hcc515_scores.tsv --method cell_zscore_mean --min-cell-lines 2 --output outputs/consensus.tsv
```

See [example provenance](../../examples/README.md). The corresponding prepared
queries are included if you want to regenerate these scores with
[cross-cell direct prediction](cross-cell.md#1-predict-directly); support inputs
are also included to try adaptation separately.

## Generate predictions for your own compounds

For example, score matched compounds in PC3 and A375 with the same pretrained
668-target model:

```bash
metar predict bulk-668 --input inputs/pc3_queries.tsv --id-column query_id --feature-order inputs/landmark_order.txt --cell-line PC3 --top-k 0 --output outputs/pc3_668.tsv
metar predict bulk-668 --input inputs/a375_queries.tsv --id-column query_id --feature-order inputs/landmark_order.txt --cell-line A375 --top-k 0 --output outputs/a375_668.tsv
metar predict consensus --input outputs/pc3_668.tsv --input outputs/a375_668.tsv --method cell_zscore_mean --min-cell-lines 2 --output outputs/consensus.tsv
```

The first two commands require your prepared responses following the
[bulk guide](bulk.md); those illustrative input tables are not bundled with the
weights. You can instead use complete genome-wide or adapted outputs, keeping
one setting and candidate set across cell lines. The final command accepts one
long-format file containing several cell lines, or repeated `--input` options.
Once predictions exist, consensus needs no additional download.

## Replace the inputs with your predictions

Every input is CSV, TSV, its gzip form, or Parquet, in **long format** with:

| Column | Content |
|---|---|
| `query_id` | Compound/condition ID, identical across the cell lines being combined |
| `cell_line` | Cell-line identifier |
| `target` | Candidate identifier |
| `score` | Finite model score, not a rank substituted for the score |

Complete CLI prediction outputs already have these columns when `--cell-line`
is recorded. `--drug-column` and `--cell-line-column` select alternative ID
columns. Other output columns may remain.

Use one row per query/cell-line/target combination. Duplicates, missing IDs and
nonfinite scores are rejected. Every query/cell-line pair must contain the same
complete target set in long format. Combine 668-target outputs with each other,
or 16,583-target outputs with each other, before applying top-k filtering. Keep
the `candidate_count` column to support truncation checks.

Use consistent compound identities and conditions, with the same ID for a
matched compound in each cell line. All inputs should use the same model
setting and response preparation.

## Choose the aggregation

`cell_zscore_mean`, the default, standardizes the full target-score vector
within each cell line and averages each target's standardized score across
cell lines. A constant vector contributes zeros. `raw_score_mean` averages
original scores instead. Both give equal weight to each included cell line.

`--min-cell-lines 2` requires at least two distinct cell lines for each query.
Queries below this threshold are skipped; the command fails if none qualifies.
An optional `--target-order inputs/target_order.txt` fixes tie ordering. It must
contain the exact observed target set, one name per line without a header (or
select a schema column with `--target-order-column`).

## Validate and interpret

```bash
metar doctor --scenario consensus --input examples/inputs/consensus/hela_scores.tsv --input examples/inputs/consensus/hcc515_scores.tsv --min-cell-lines 2
```

Output contains one ranking per qualifying query, the [usual ranking columns](../usage.md#read-outputs-and-check-inputs),
and `n_cell_lines`, `cell_lines` and `aggregation_method`. Higher aggregate
scores rank first. These are combined model rankings, not activity measurements
in one cell line or calibrated binding probabilities. Use `--top-k 20` on the
final output if desired, after combining complete vectors.
