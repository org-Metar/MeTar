# Prediction guide

Load a pretrained model and use an example or your own prepared responses. Run
commands from the repository root after [installation](installation.md) and
[model download](data.md). Each guide explains its input format, preparation,
example availability, command options and output.

## Select a workflow

| Setting | What you provide | Candidates | Detailed guide |
|---|---|---|---|
| Bulk transcriptomic prediction | Prepared 978-landmark responses | 668 targets | [Bulk](prediction/bulk.md) |
| New cell line, direct or adapted | Query responses; separate support responses and known targets for adaptation | 668 targets | [Cross-cell](prediction/cross-cell.md) |
| Genome-wide target ranking | Prepared 978-landmark responses for the full-space model | 16,583 genes | [Genome-wide](prediction/genome-wide.md) |
| Single-cell transcriptomic prediction | Grouped cells in saved X128 coordinates | 16,583 genes | [Single-cell](prediction/single-cell.md) |
| Proteomic prediction | Abundances covering saved protein features | 16,583 genes | [Proteomics](prediction/proteomics.md) |
| Multiple cell lines | Complete predictions over one matching target set | Same as inputs | [Consensus](prediction/consensus.md) |

Model prediction defaults to `--device auto`, selecting CUDA when available,
otherwise CPU. Use `--device cpu` to require CPU or `--device cuda:0` to require the
first visible GPU; an unavailable explicit CUDA request is an error. Consensus
has no device option. See [runtime reporting](runtime.md) for progress, quiet
mode and logs, or `metar predict SCENARIO --help` for all options.

Choose an output path distinct from your input files and selected model assets.
The CLI rejects paths that would overwrite these files, including links to them.
An existing prediction file at a separate destination can be replaced.

## Start with the bundled examples

Each guide starts with a command using real prepared inputs included in
[`examples/`](../examples/README.md). Download the model archive for model-based
prediction; consensus can use the supplied complete score tables directly.
These small examples demonstrate the interfaces, rather than replacing the
full evaluation datasets. Follow the same guide to replace them with your own
data.

<a id="start-with-a-real-pdf-3-query"></a>

The manuscript's [PDF-3 application query](prediction/bulk.md#optional-run-the-pdf-3-application-query)
is also available separately through the result-data archive.

## Prepare input files

### Bulk tables and NumPy arrays

Use one query per row and 978 ordered landmark response features. Named table
columns can be selected/reordered with `--feature-order`; NumPy arrays must
already follow that order. See the [bulk format and preparation requirements](prediction/bulk.md#replace-the-example-with-your-data).
The [genome-wide branch](prediction/genome-wide.md#prepare-your-response-table)
and 668-target branch have distinct preprocessing contracts.

### Single-cell tables, NPZ and H5AD

Use one cell per row, 128 saved X128 coordinates and a group ID for the query.
The coordinates are not 128 selected genes. See the
[single-cell input formats](prediction/single-cell.md#replace-the-input-with-your-cells).

### Proteomics tables

Use one query per row and exact named abundance features from the saved
preprocessing JSON. See [proteomics input preparation](prediction/proteomics.md#prepare-your-abundance-table).

## 1. Bulk expression: 668 targets

The pretrained scorer ranks 668 supervised targets. Use the [bulk guide](prediction/bulk.md)
for its command, bundled example, feature order and response-scale
requirements. Ordinary LINCS-style responses require the training
drug-cell-mean standardization; its fitted mean/std state is not separately
released. The documented PDF-3 preparation is a separate application, not a
general raw-log2FC rule.

## 2. Genome-wide prediction

The [genome-wide guide](prediction/genome-wide.md) loads the full-space query
model and its 16,583-gene embeddings. It explains response preparation and why
new genes cannot be added by supplying names alone.

## 3. Prediction in a new cell line

### Direct prediction

Use `metar predict cross-cell --mode direct` with prepared responses.
[Full instructions](prediction/cross-cell.md#1-predict-directly).

### Prediction with support compounds

Use `--mode adapted` with separate same-cell-line support responses and known
support targets. [Input formats and command](prediction/cross-cell.md#2-use-reference-compounds-from-the-new-cell-line).
The trained adapter acts at inference; it does not retrain the model.

## 4. Single-cell prediction

Choose A549, SW480 or NCI-H460 and provide grouped X128 cells.
[Full instructions](prediction/single-cell.md).

### Prepare X128 from raw Tahoe plates

Reuse the released fixed gene selection and SVD projection. The original
ordered Tahoe schema and sparse layout are required; refitting SVD changes
coordinates. [Download, check and project the plates](prediction/single-cell.md#prepare-x128-from-the-original-tahoe-plates).

## 5. Proteomics prediction

The [proteomics guide](prediction/proteomics.md) explains loading the saved
encoder/preprocessing, inspecting required features, and providing abundance
values without applying transformations twice.

## 6. Consensus across cell lines

The [consensus guide](prediction/consensus.md) combines complete long-format
predictions from at least two cell lines. It explains compound IDs, matching
target sets, score standardization and averaging. No additional model is fit.

## Read outputs and check inputs

Output suffix selects TSV, CSV, their gzip forms, or Parquet. In default long
format, one row represents a query-target pair:

| Column | Meaning |
|---|---|
| `query_id` | Input query identifier |
| `scenario` | Prediction setting |
| `candidate_count` | Complete candidate-space size, before `--top-k` filtering |
| `target` | Candidate identifier from the model or target table |
| `score` | Model score, sorted descending |
| `rank` | Ranking position, starting at 1 |
| `rank_fraction` | `rank / candidate_count` |
| `rank_percent` | `100 * rank / candidate_count` |

Metadata may include `cell_line`, `cells_used`, support information or consensus
counts. Larger scores rank first; `rank=1` is the highest-scoring candidate.
Scores and rank percentages are not calibrated binding probabilities. Ties
follow candidate order.

`--top-k 0` writes all candidates; `--top-k 20` keeps the first 20 long-format
rows per query. `--output-format wide` writes one query per row and one score
column per target, retaining all targets. Consensus requires complete long
inputs; generate those before shortening a ranking.

Check model assets or a prepared input without inference:

```bash
metar doctor --scenario bulk-668
metar doctor --scenario bulk-668 --input inputs/bulk_queries.tsv --feature-order inputs/landmark_order.txt
metar doctor --scenario single-cell --cell-line A549 --input inputs/a549_x128.h5ad --id-column drugname_drugconc
metar doctor --scenario proteomics --input inputs/proteomics_queries.tsv
```

Doctor checks files, structure and dimensions. It cannot establish that an
assay uses training-compatible preparation or feature identities from shape
alone. Checks may read the full single-cell matrix. Missing model files usually
indicate an incomplete download or incorrect
[`METAR_ARTIFACT_ROOT`](data.md#set-the-artifact-root).

## Python API

```python
from metar import (
    predict_bulk_668,
    predict_cross_cell,
    predict_full_space,
    predict_single_cell,
    predict_proteomics,
    consensus_from_long,
)
```

These functions take prepared arrays/tables and explicit asset paths. The CLI
adds file loading and default paths. They return `PredictionResult` with
`.to_long()`, `.to_wide()` and `.write()`. See
[`src/metar/predict.py`](../src/metar/predict.py) and
[`src/metar/consensus.py`](../src/metar/consensus.py) for signatures.

For training, see [training](training.md). Manuscript evaluation and plotting
are documented separately in [reproduction](reproduction.md).
