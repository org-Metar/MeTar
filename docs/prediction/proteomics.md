# Proteomic prediction

Use the pretrained protein-abundance encoder to rank 16,583 candidate genes.
Prediction reuses saved feature selection and preprocessing, without query
target labels or a new preprocessing fit.

## Download and predict

After [installation](../installation.md), run from the repository root:

```bash
python scripts/download_data.py --archive models_and_representations.tar.gz --download --directory ../metar-downloads --package ../metar_artifacts
```

Set [`METAR_ARTIFACT_ROOT`](../data.md#set-the-artifact-root) to that directory.
These defaults are used together:

| Asset | Path under the artifact root |
|---|---|
| Model | `checkpoints/selected/round261_proteomics_epoch070.pth` |
| Preprocessing | `data/processed/proteomics/round261_preprocessor.json` |
| Target embeddings | `data/processed/fullspace/round259_target_embedding_aligned_to_proteomics_gene_order.csv` |

```bash
metar predict proteomics --input examples/inputs/proteomics.tsv --id-column query_id --device cpu --output outputs/proteomics.tsv
```

This uses two real abundance responses with the 2,000 required named features,
included in the checkout. Saved preprocessing is applied during prediction.
See [example provenance](../../examples/README.md). Use `--checkpoint`,
`--preprocessor` and `--target-table` for compatible explicit assets. Batch
size defaults to 64; `--device cuda:0` requests a GPU.

## Prepare your abundance table

Replace the bundled path with your table; no model fitting is needed if its
feature definitions and measurement representation match:

```bash
metar predict proteomics --input inputs/proteomics_queries.tsv --id-column query_id --device cpu --output outputs/proteomics.tsv
```

Use one query per row in CSV, TSV, their gzip forms or Parquet. Include a
unique `query_id` and columns covering all names in the preprocessor's
`feature_names` list. The layout is:

```text
query_id	<feature_names[0]>	<feature_names[1]>	...
compound_a	12.4	5.6	...
compound_b	10.1	6.2	...
```

Ellipses stand for omitted columns, not literal input. Read the actual 2,000
names and transform flag from the released JSON:

```python
import json
import os
from pathlib import Path

root = Path(os.environ["METAR_ARTIFACT_ROOT"])
path = root / "data/processed/proteomics/round261_preprocessor.json"
state = json.loads(path.read_text())
print("Selected features:", len(state["feature_names"]))
print("First names:", state["feature_names"][:10])
print("Apply log1p inside MeTar:", state["log1p"])
```

Use exact identifiers: symbols, accessions and assay column names are not
automatically interchangeable. Column order may differ, because preprocessing
selects/reorders by name. Extra columns are ignored; absent required columns
are errors. Query target labels are not input features.

Abundances must use the same representation and units as training. MeTar
selects the 2,000 columns, replaces missing/nonfinite values *inside existing
columns* with zero, applies `log1p` if the saved flag requires it, and
standardizes with saved training means/stds. Negative values are rejected when
`log1p` is required. Supply values before those saved transformations; do not
apply the same log or standardization twice. Missing-value handling does not
make an unrelated assay compatible.

## Example data and custom assays

The checkout's two-query example is a small subset of the matched processed
ProteomicsDB data. For the full input, obtain it separately and follow the
[data guide](../data.md#place-separately-obtained-public-inputs). Use its
abundance columns and choose a unique query ID per row. A new download or
independent assay still needs identifier, unit and preparation alignment;
the CLI does not perform that biological mapping.

For an assay without the trained features or with a different representation,
train a suitable modality encoder rather than renaming features or zero-filling
whole missing protein columns. See [training](../training.md).

## Reconstruct preprocessing only when needed

Normal prediction uses `--preprocessor` or the default released JSON. If neither
is available, `--reference-data` can reconstruct the original preparation from
the matched reference CSV. That CSV has an index in the first column, `drug`,
`target` and original raw abundance columns. Selection and fitting use the
chosen training rows, not your queries. `--write-preprocessor PATH` saves the
reconstruction. An existing selected JSON takes precedence over reference data.

Keep the proteomics-aligned target table paired with its model; do not swap in
the ordinary full-space CSV without matching the expected target ordering.

## Validate and read results

```bash
metar doctor --scenario proteomics --input examples/inputs/proteomics.tsv --id-column query_id
```

Doctor checks files and feature layout, not the biological scale of an
independent assay. Default output ranks all 16,583 candidates per query;
`--top-k 20` keeps a shortlist. Higher scores rank first. Scores and
`rank_percent` are not binding probabilities. See the
[output reference](../usage.md#read-outputs-and-check-inputs).
