# Prediction in a new cell line

Apply the shared 668-target model directly (`MeTar_direct`), or use responses
and known targets from separate reference compounds in the new cell line
(`MeTar_adapted`). Adapted prediction applies the pretrained support-conditioned
adapter at inference; it does not retrain model parameters on your query.

## Download the models

Complete [installation](../installation.md), then run from the repository root:

```bash
python scripts/download_data.py --archive models_and_representations.tar.gz --download --directory ../metar-downloads --package ../metar_artifacts
```

Set [`METAR_ARTIFACT_ROOT`](../data.md#set-the-artifact-root) to that directory.
Direct prediction uses `checkpoints/selected/round246_whole_model_checkpoint.pth`;
adapted prediction also uses
`checkpoints/selected/round252_adapter_seed20260524.pth`. `--checkpoint` and
`--adapter-checkpoint` override those defaults.

## 1. Predict directly

```bash
metar predict cross-cell --mode direct --cell-line HELA --input examples/inputs/cross_cell/hela/query.npz --device cpu --output outputs/hela_direct.tsv
```

This runs two bundled real HELA query responses. No support compounds or query
labels are required for direct prediction. With your own data, follow the
[bulk format and preparation](bulk.md#replace-the-example-with-your-data),
including training-matched standardization. `--cell-line` records context; it
does not choose a cell-line-specific bulk checkpoint or transform the assay.

## 2. Use reference compounds from the new cell line

Prepare three files:

| File | Content |
|---|---|
| Query responses | One query per row: unique `query_id` and 978 landmark response features |
| Support responses | One separate reference compound per row: unique `query_id` and the same 978 features |
| Support labels | One row per support ID: `query_id` and known `targets` |

Both response tables must come from the same cell line and follow the same
feature schema and preparation. Keep query and support compounds separate;
renaming the same compound does not make it independent support. Use explicit
IDs, because generated IDs such as `query_0` can overlap between files.

The CLI does not calculate the bulk standardization for either table. Prepare
raw assays externally, using the training-derived per-feature mean/std for both
query and support responses. The exact existing implementation is
[`standardize_tables`](../../training/support_query_shift/support_query_shift.py#L245),
called by [`build_round243_tables`](../../training/leakage_safe_support/leakage_safe_support.py#L553)
after compound/cell-line aggregation and training split selection. See the
[three preparation steps](bulk.md#match-the-response-preparation), including
the small-standard-deviation rule and the limitation that the fitted state is
not separately released. Do not fit those values on the new cell line's
queries or support set. The bundled examples are already transformed.

The support-label TSV looks like this:

```text
query_id	targets
support_a	TARGET_A
support_b	TARGET_B|TARGET_C
```

Replace these schematic IDs and targets with real values. Each support compound
must have at least one target from the checkpoint's 668-target set. Use its
exact target names, not arbitrary synonyms or accessions; an outside target is
an error. Labels align by ID, regardless of physical row order. Separate
multiple targets with `|`. `--target-column` changes the label column name;
`--id-column` applies to all three tables.

```bash
metar predict cross-cell --mode adapted --cell-line HELA --input examples/inputs/cross_cell/hela/query.npz --support-input examples/inputs/cross_cell/hela/support.npz --support-labels examples/inputs/cross_cell/hela/support_labels.tsv --support-weights examples/inputs/cross_cell/hela/support_weights.npy --device cpu --output outputs/hela_adapted.tsv
```

No query target labels are supplied. The adapter adjusts each query
representation using the support responses and known support targets before
scoring the same 668 candidates. Support rows are equally weighted by default.
`--support-weights PATH` optionally accepts a one-dimensional positive finite
`.npy` array or a table containing exactly one numeric column. Weights follow
support-response row order, not label-file order.

## Example inputs and your own assay

The commands above use two real HELA queries and 64 separate support responses
with known targets. A second HCC515 set for the same query compounds is also
provided under `examples/inputs/cross_cell/hcc515/`. See
[example provenance](../../examples/README.md). The inputs are already prepared;
only the model archive is needed to run them. They are small demonstrations,
not the full manuscript evaluation.

For your own named tables, replace those paths and provide the feature schema:

```bash
metar predict cross-cell --mode adapted --cell-line VCAP --input inputs/vcap_queries.tsv --id-column query_id --feature-order inputs/landmark_order.txt --support-input inputs/vcap_support.tsv --support-labels inputs/vcap_support_labels.tsv --device cpu --output outputs/vcap_adapted.tsv
```

For manuscript bulk/cross-cell inputs, see [public input setup](../data.md#place-separately-obtained-public-inputs).
The matched processed LINCS table and metadata are separate inputs; an upstream
download is not automatically the selected processed table. With your own
assay, preserve feature identity/order, response scale, compound separation
and the known-target vocabulary. Do not fit a query-only standardizer and
assume it matches training.

## Check and read the result

```bash
metar doctor --scenario cross-cell --mode adapted --cell-line HELA --input examples/inputs/cross_cell/hela/query.npz --support-input examples/inputs/cross_cell/hela/support.npz --support-labels examples/inputs/cross_cell/hela/support_labels.tsv
```

For direct checks, set `--mode direct` and omit support options. Doctor checks
structure without inference; it cannot establish assay compatibility from
dimensions alone. `--device auto` is the default; `--device cpu` requires CPU
and `--device cuda:0` requires the first visible GPU. See
[runtime reporting](../runtime.md). The
[usual output columns](../usage.md#read-outputs-and-check-inputs) include
`cell_line` and `prediction_mode`; adapted output also records support count
and shift scale. Scores express within-query rankings, not calibrated
probabilities. For [consensus](consensus.md), keep complete long-format output
with `--top-k 0` and a consistent prediction setting across cell lines.
