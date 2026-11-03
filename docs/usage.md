# Prediction guide

Install MeTar using [installation.md](installation.md), download the required
[models and data](data.md), and set `METAR_ARTIFACT_ROOT` before running these
commands. Run from the repository root with the environment active.

New users can start with the complete [PDF-3 quick start](quickstart.md). For an
existing dataset, select a workflow below, prepare its input representation and
run its command. See [troubleshooting](troubleshooting.md) for common errors.

## Start with a real PDF-3 query

This example uses a prepared query shipped with the model's reference data.
Set `METAR_ARTIFACT_ROOT` as shown in [data setup](data.md#set-the-artifact-root)
before the `metar` commands:

```bash
python scripts/download_data.py --archive models_and_representations.tar.gz --archive result_data.tar.gz --download --directory ../metar-downloads --package ../metar_artifacts
python scripts/prepare_pdf3.py --package ../metar_artifacts --output ../metar-example/pdf3.csv
metar doctor --scenario bulk-668 --input ../metar-example/pdf3.csv
metar predict bulk-668 --input ../metar-example/pdf3.csv --id-column query_id --cell-line JHH7 --top-k 20 --output ../metar-example/pdf3_targets.tsv
```

The preparer preserves the archived query's 978-feature order. This default path
needs no public raw-input download or gene schema. It uses the PDF-3 application's
aligned log2-fold-change contract, rather than processing arbitrary new RNA-seq
counts. The optional `prepare_pdf3.py --from-raw` path requires the processed
`JHH7.csv` from `application_inputs.tar.gz` and a separately obtained canonical
`geneinfo_beta.txt`; it starts from processed FPKM values, not FASTQ.

## Select a workflow

Paths under `inputs/` are files you prepare. The CLI selects the checkpoint,
target embeddings and saved assay preprocessor from `METAR_ARTIFACT_ROOT`.

| Goal | Command | Query representation | Candidates |
|---|---|---|---|
| Bulk expression | `metar predict bulk-668` | 978 prepared response features | 668 targets |
| Genome-wide target ranking | `metar predict full-space` | 978 features prepared for the full-space model | 16,583 genes |
| New cell line without support data | `metar predict cross-cell --mode direct` | Prepared bulk response | 668 targets |
| New cell line with labeled support compounds | `metar predict cross-cell --mode adapted` | Bulk query and separate support responses | 668 targets |
| Single-cell | `metar predict single-cell` | Cell groups in the trained X128 representation | 16,583 genes |
| Proteomics | `metar predict proteomics` | Protein-abundance features required by the saved preprocessor | 16,583 genes |
| Multiple cell lines | `metar predict consensus` | Complete predictions over the same target set | Same as inputs |

Prediction defaults to CPU. Add `--device cuda:0` for a CUDA-enabled environment;
check `torch.cuda.is_available()` first, because the prediction API falls back
to CPU when CUDA is unavailable. Explicit checkpoint, target-table and batching
options are listed by `metar predict SCENARIO --help`.

All prediction branches except consensus use weights from
`models_and_representations.tar.gz`. The single-cell projection is an additional
download when starting from raw Tahoe plates; consensus uses already generated
prediction tables. Model downloads alone do not supply an independent assay's
query table. The [data guide](data.md) lists archive contents and external inputs.

## Prepare input files

### Bulk tables and NumPy arrays

`bulk-668`, `full-space` and `cross-cell` accept:

| Format | Required content |
|---|---|
| TSV, CSV, compressed TSV/CSV or Parquet | One query per row, an optional unique ID column, and 978 response features |
| `.npy` | An `N x 978` matrix or one length-978 vector; IDs are generated automatically |
| `.npz` | Numeric array `x` or `features`; optional string array `query_ids` |

For tables, pass `--id-column query_id` and
`--feature-order inputs/landmark_order.txt`. The feature-order file contains
978 training feature names, one per line with no header. A schema table with a
`gene_symbol` or other supported name column also works; select its column with
`--feature-column`. The original `geneinfo_beta.txt` is filtered to
`feature_space == landmark` while preserving row order.

With a schema, MeTar selects columns by name and restores training order.
Without an explicit schema, it tries the default `geneinfo_beta.txt` in the
artifact root. If that file is absent too, it takes every numeric column except
the ID column, in the table's existing order, and requires exactly 978. Do not
mix numeric metadata such as dose or time into that input. Feature-order files
cannot reorder unnamed NumPy arrays; arrays must already be correctly ordered.
All bulk feature values must be finite, and query IDs must be unique and
aligned with rows. Use explicit IDs for adapted prediction so automatically
generated query/support IDs do not overlap.

### Single-cell tables, NPZ and H5AD

Each row is a cell; a group ID identifies the cells belonging to one query.

| Format | Required content |
|---|---|
| `.h5ad` | `.X` is `cells x 128` X128; the group column is in `.obs` |
| `.npz` | `x` or `features` is `cells x 128`; `query_ids` or `groups` identifies each cell's query |
| TSV, CSV, compressed TSV/CSV or Parquet | A group column followed by 128 numeric X128 features in training order |

Single-cell prediction has no `--feature-order` option. Keep only the group
column and the correctly ordered 128 numeric coordinates in tabular inputs.
Group IDs may repeat. `.npy` is not a public input format for this branch.
A matrix of 128 selected genes is not X128. H5AD loading converts sparse `.X`
to a dense array, so allow sufficient memory for large query files.

### Proteomics tables

Use TSV, CSV, their supported compressed forms, or Parquet, with one row per
query and preferably a unique `query_id`. Column names must cover all
`feature_names` in the preprocessing JSON. Arbitrary protein identifiers are
not interchangeable with these names. Extra columns are ignored; missing
required columns are errors. Target labels are not query inputs.

## 1. Bulk expression: 668 targets

```bash
metar predict bulk-668 --input inputs/bulk_queries.tsv --id-column query_id --feature-order inputs/landmark_order.txt --cell-line PC3 --device cpu --output outputs/bulk_668.tsv
```

The model needs prepared 978-feature responses, not raw read counts or an
arbitrary normalized expression matrix. Ordinary LINCS-style bulk queries
require the training drug-cell-mean standardization. Its fitted mean/std state
is not separately supplied in the pretrained archive; gene alignment alone is
insufficient. Reconstruct the matching preparation from training inputs when
needed, rather than fitting a new transform on an unrelated query set.

The retained PDF-3 application follows its separate documented procedure:
average treated and control FPKM separately, calculate
`log2((treated_mean_FPKM + 1) / (control_mean_FPKM + 1))`, then align the 978
landmarks. This application does not establish a general raw-log2FC input rule
for other assays.

## 2. Genome-wide prediction

```bash
metar predict full-space --input inputs/full_space_queries.tsv --id-column query_id --feature-order inputs/landmark_order.txt --cell-line PC3 --device cpu --output outputs/full_space.tsv
```

This branch uses the prepared bulk response without the additional
standardization required by the 668-target branch. Do not reuse the same matrix
across branches solely because both accept 978 features. The released embedding
table covers 16,583 candidate genes, including targets outside the supervised
set; the command cannot generate embeddings for genes absent from that table.

A custom `--target-table` must have unique target names in its first column and
finite numeric embedding columns matching the checkpoint dimension. Changing
the candidate subset also changes how ranks should be interpreted.

## 3. Prediction in a new cell line

### Direct prediction

```bash
metar predict cross-cell --mode direct --cell-line VCAP --input inputs/vcap_queries.tsv --id-column query_id --feature-order inputs/landmark_order.txt --device cpu --output outputs/vcap_direct.tsv
```

This applies the shared 668-target scorer. `--cell-line` records the query
context; it does not select a new cell-line-specific bulk checkpoint or change
preprocessing automatically.

### Prediction with support compounds

Prepare query responses, support responses from the same cell line, and support
target labels. Query and support compounds must be separate. Use compound IDs
consistently; renaming the same compound does not make it independent support.
Both response tables must follow the same bulk preparation contract.

The label file is tab-separated, with one or more targets per support ID:

```text
query_id	targets
support_a	TARGET_A
support_b	TARGET_B|TARGET_C
```

Replace the example IDs and targets with real values. Every support query needs
at least one valid target in the 668-target set. Labels align by ID, so their
physical row order need not match the support-response table.

```bash
metar predict cross-cell --mode adapted --cell-line VCAP --input inputs/vcap_queries.tsv --id-column query_id --feature-order inputs/landmark_order.txt --support-input inputs/vcap_support.tsv --support-labels inputs/vcap_support_labels.tsv --device cpu --output outputs/vcap_adapted.tsv
```

This applies the trained support-conditioned adapter at inference; it does not
retrain model parameters for the new cell line. Support rows have equal weight
unless `--support-weights` is provided. That option accepts a one-dimensional
positive NumPy array or a table containing exactly one numeric column. Weights
follow support-response row order, not label-file order.

## 4. Single-cell prediction

```bash
metar predict single-cell --input inputs/a549_x128.h5ad --id-column drugname_drugconc --cell-line A549 --max-cells 300 --sampling-seed 20260327 --device cpu --output outputs/a549_single_cell.tsv
```

Supported cell lines are `A549`, `SW480` and `NCI-H460`, each with its own
trained checkpoint. The bulk checkpoint is not applied to individual cells.
By default, at most 300 cells are selected per query, deterministically from its
ID and the sampling seed. Smaller groups use all cells; `--max-cells 0` uses all
cells in every group. Grouping by `drug` instead of `drugname_drugconc` can merge
concentrations and therefore changes query meaning. Public-query sampling also
differs from the fixed pools used for archived manuscript comparisons.

### Prepare X128 from raw Tahoe plates

Download the saved projection and install its optional dependencies:

```bash
python scripts/download_data.py --archive tahoe_projection.tar.gz --download --directory ../metar-downloads --package ../metar_artifacts
uv pip install -e ".[tahoe-preprocessing]"
python scripts/prepare_single_cell.py --input-dir ../raw_tahoe_plates --assets ../metar_artifacts/data/preprocessing/tahoe --output-dir ../prepared/tahoe_x128 --check-only
```

`--check-only` inspects projection assets and plate schemas without loading the
full expression matrix or writing output. If it passes, run the projection and
assemble a cell-line file:

```bash
python scripts/prepare_single_cell.py --input-dir ../raw_tahoe_plates --assets ../metar_artifacts/data/preprocessing/tahoe --output-dir ../prepared/tahoe_x128
python preprocessing/assemble_single_cell.py --raw-dir ../raw_tahoe_plates --x128-dir ../prepared/tahoe_x128 --out-dir ../prepared/cell_lines --work-dir ../prepared/cell_line_work --cell A549
metar predict single-cell --input ../prepared/cell_lines/tahoe_a549.h5ad --id-column drugname_drugconc --cell-line A549 --device cpu --output outputs/a549_single_cell.tsv
```

The preparer requires CSR sparse `.X` and the original, ordered 62,710-gene
Tahoe schema. Dense or CSC matrices, reordered genes and other schemas are
rejected. It reuses the fixed 6,000-gene selection and SVD projection; do not add
another normalization, log transform, centering or scaling step. Selected genes
retain original gene order, rather than variance order.

The saved components are float32 and promoted to float64 for projection,
matching the notebook reload path. This is not a claim of bitwise identity to
its initial live-fit output. Refitting SVD changes coordinates and requires
retraining the single-cell model. The assembler accumulates selected cells in
memory before writing H5AD.

## 5. Proteomics prediction

```bash
metar predict proteomics --input inputs/proteomics_queries.tsv --id-column query_id --device cpu --output outputs/proteomics.tsv
```

Use the same feature identifiers and abundance representation as training. The
saved preprocessing JSON contains 2,000 selected names, training means/stds and
a `log1p` flag. MeTar selects these columns, fills missing/nonfinite values
inside present columns with zero, then applies the saved transforms. A state
requiring `log1p` rejects negative input. Do not apply the same log transform or
standardization twice.

Normally no preprocessing fit is needed. `--reference-data` and
`--write-preprocessor` can reconstruct the training preparation for an
independent check; they do not fit on a new query. An existing selected
preprocessor takes precedence. The reference CSV has an index column, `drug`,
`target`, and raw feature columns; selection and fitting use training rows.
Use the matching target table, whose order is aligned with the proteomics model.

## 6. Consensus across cell lines

Score the same compounds in at least two cell lines using the same candidate
space, retaining complete predictions:

```bash
metar predict bulk-668 --input inputs/pc3_queries.tsv --id-column query_id --feature-order inputs/landmark_order.txt --cell-line PC3 --top-k 0 --output outputs/pc3_668.tsv
metar predict bulk-668 --input inputs/a375_queries.tsv --id-column query_id --feature-order inputs/landmark_order.txt --cell-line A375 --top-k 0 --output outputs/a375_668.tsv
metar predict consensus --input outputs/pc3_668.tsv --input outputs/a375_668.tsv --method cell_zscore_mean --min-cell-lines 2 --output outputs/consensus.tsv
```

Inputs need `query_id`, `cell_line`, `target` and `score`. Use consistent compound
IDs across cell lines. Each query/cell-line pair must contain the same complete
target set, with no duplicate targets. Do not mix 668- and 16,583-target outputs,
use top-k-only tables, or pass wide-format scores.

`cell_zscore_mean` standardizes each cell line's target-score vector before
averaging by target; `raw_score_mean` averages the original scores. Queries below
`--min-cell-lines` are skipped; the command fails if no query qualifies. An
optional `--target-order inputs/target_order.txt` fixes tie ordering. Consensus
does not train or update model weights.

## Read outputs and check inputs

Output suffix selects TSV, CSV, their compressed forms, or Parquet. In the
usual long format, each row is one query-target pair:

| Column | Meaning |
|---|---|
| `query_id` | Input query identifier |
| `scenario` | Prediction workflow recorded by the model interface |
| `candidate_count` | Size of the complete candidate space, before any `--top-k` filtering |
| `target` | Candidate target identifier |
| `score` | Model score, ranked in descending order |
| `rank` | Position in the query's target ranking, starting at 1 |
| `rank_fraction` | `rank / candidate_count` |
| `rank_percent` | `100 * rank / candidate_count` |

Scenario metadata can include `cell_line`, `cells_used` and support information.
Larger scores rank first; `rank=1` is the best target. `rank_percent` is
`100 * rank / candidate_count`, not confidence or binding probability. Ties
follow the target table's order. `--top-k 0` writes all candidates;
`--top-k 20` keeps the first 20 long-format rows per query.
`--output-format wide` writes one query per row and one score column per target.
Consensus requires complete long-format input.

Check artifacts or a specific prepared file before loading the model:

```bash
metar doctor --scenario bulk-668
metar doctor --scenario bulk-668 --input inputs/bulk_queries.tsv --feature-order inputs/landmark_order.txt
metar doctor --scenario single-cell --cell-line A549 --input inputs/a549_x128.h5ad --id-column drugname_drugconc
metar doctor --scenario proteomics --input inputs/proteomics_queries.tsv
```

Doctor checks files and shapes; it does not run inference or establish that an
assay's preprocessing matches training. Input checks may read the complete
single-cell matrix. Missing checkpoints or target tables usually mean the wrong
artifact root or an incomplete download; see [data setup](data.md). A feature
count alone cannot diagnose a preparation mismatch.

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

Prediction functions return `PredictionResult` with scores, query IDs and target
names. Export with `.to_long()`, `.to_wide()` or `.write()`. Python functions take
prepared arrays/tables and explicit artifact paths; signatures are in
`src/metar/`. The CLI adds file loading and artifact-root defaults.

For scratch training, see [training](training.md). For archived comparisons and
numerical limits, see [reproduction](reproduction.md).
