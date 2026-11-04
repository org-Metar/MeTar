# Single-cell transcriptomic prediction

Rank 16,583 candidate genes from a group of cells belonging to one compound
response. The released single-cell checkpoints support `A549`, `SW480` and
`NCI-H460`. Each expects the saved 128-dimensional Tahoe projection, X128,
rather than raw counts or 128 arbitrarily selected genes.

## Download and predict from X128

Complete [installation](../installation.md), including the `single-cell` extra
for H5AD. Run from the repository root:

```bash
python scripts/download_data.py --archive models_and_representations.tar.gz --download --directory ../metar-downloads --package ../metar_artifacts
```

Set [`METAR_ARTIFACT_ROOT`](../data.md#set-the-artifact-root) to the extracted
directory. `--cell-line` selects a checkpoint under `checkpoints/selected/`:

| Cell line | Checkpoint |
|---|---|
| A549 | `round260_A549_seed2043.pth` |
| SW480 | `round260_SW480_seed2041.pth` |
| NCI-H460 | `round260_NCI_H460_seed2042.pth` |

All use `data/processed/fullspace/round259_target_embedding.csv`. A compatible
`--checkpoint` or `--target-table` overrides a default.

```bash
metar predict single-cell --input examples/inputs/single_cell_a549.npz --cell-line A549 --max-cells 0 --device cpu --output outputs/a549_single_cell.tsv
```

The bundled example contains two A549 compound groups with 300 real X128 cells
each. It needs no raw plate download or new projection. See its
[provenance](../../examples/README.md).

## Replace the input with your cells

For a prepared H5AD file of your own:

```bash
metar predict single-cell --input inputs/a549_x128.h5ad --id-column drugname_drugconc --cell-line A549 --device cpu --output outputs/a549_single_cell.tsv
```

Each row is one cell. Its group ID identifies the query whose cells are scored
together. Repeated group IDs are expected; all coordinates must be finite.

| Format | Required structure |
|---|---|
| H5AD | `.X` is `cells x 128` X128; `.obs` contains the group column selected by `--id-column` |
| NPZ | `x` or `features` is `cells x 128`; `query_ids` or `groups` has one group ID per cell |
| CSV/TSV, gzip forms, Parquet | Group column plus 128 numeric X128 columns in saved component order |

For example, a table has `query_id`, `X128_1`, ..., `X128_128` columns, one cell
per row. Coordinate names illustrate the layout: the loader uses numeric
columns in their existing order, not their names. There is no single-cell
`--feature-order`. Keep numeric metadata out of the matrix. `.npy` is not
supported. H5AD loading densifies sparse `.X`; allow sufficient memory.

Choose IDs that preserve the prediction condition. `drugname_drugconc` retains
concentrations; grouping by `drug` may merge them. Query target labels are not
required. By default, at most 300 cells per group are selected deterministically
from its ID and `--sampling-seed`. Smaller groups use all cells;
`--max-cells 0` uses every cell. Grouping/sampling changes may differ from the
manuscript's fixed cell pools.

## Prepare X128 from the original Tahoe plates

The small bundled example already contains X128. For your raw plates, obtain
upstream data separately, or place existing processed files following the
[data guide](../data.md#place-separately-obtained-public-inputs). For the original
raw plate schema, reuse the released projection:

```bash
python scripts/download_data.py --archive tahoe_projection.tar.gz --download --directory ../metar-downloads --package ../metar_artifacts
uv pip install -e ".[tahoe-preprocessing]"
python scripts/prepare_single_cell.py --input-dir ../raw_tahoe_plates --assets ../metar_artifacts/data/preprocessing/tahoe --output-dir ../prepared/tahoe_x128 --check-only
```

`--check-only` examines assets and plate schemas without reading the full
expression matrix or writing projection output. If it passes:

```bash
python scripts/prepare_single_cell.py --input-dir ../raw_tahoe_plates --assets ../metar_artifacts/data/preprocessing/tahoe --output-dir ../prepared/tahoe_x128
python preprocessing/assemble_single_cell.py --raw-dir ../raw_tahoe_plates --x128-dir ../prepared/tahoe_x128 --out-dir ../prepared/cell_lines --work-dir ../prepared/cell_line_work --cell A549
metar predict single-cell --input ../prepared/cell_lines/tahoe_a549.h5ad --id-column drugname_drugconc --cell-line A549 --device cpu --output outputs/a549_single_cell.tsv
```

The preparer requires the original ordered 62,710-gene schema and CSR sparse
`.X`; dense/CSC matrices, reordered genes and other schemas are rejected. It
keeps the fixed 6,000 selected genes in original gene order and applies saved
SVD components. Do not add normalization, log transformation, centering or
scaling. Selected gene order is not variance rank. Refitting SVD changes the
coordinates and requires encoder retraining. The assembler holds selected
cells in memory before writing H5AD.

An independent assay needs the trained representation and compatible model;
any 128-dimensional embedding is insufficient. The original-plate preparer
does not automatically adapt another gene schema or cell line. See
[training](../training.md) when a new modality encoder is needed.

## Check inputs and read output

```bash
metar doctor --scenario single-cell --cell-line A549 --input examples/inputs/single_cell_a549.npz
```

Doctor may read the complete cell matrix. Prediction returns one target ranking
per group and records `cell_line`, `cells_used` and `sampling_seed`. The
[usual ranking columns](../usage.md#read-outputs-and-check-inputs) apply:
`--top-k 0` keeps all candidates and `--top-k 20` keeps the first 20.
`--device auto` is the default; `--device cpu` requires CPU and `--device cuda:0`
requires the first visible GPU. See [runtime reporting](../runtime.md).
