# Genome-wide target ranking

Use `full-space` to rank 16,583 candidate genes from a prepared bulk
transcriptomic response, including candidates outside the 668 supervised
targets. Prediction loads the query model and exported target embeddings
together, without query labels or a new graph-training run.

## Download and predict

After [installation](../installation.md), run from the repository root:

```bash
python scripts/download_data.py --archive models_and_representations.tar.gz --download --directory ../metar-downloads --package ../metar_artifacts
```

Set [`METAR_ARTIFACT_ROOT`](../data.md#set-the-artifact-root) to the extracted
directory. The paired defaults are:

| Asset | Path under the artifact root |
|---|---|
| Query model | `checkpoints/selected/round259_e13_bce001.pth` |
| Target embeddings | `data/processed/fullspace/round259_target_embedding.csv` |

```bash
metar predict full-space --input examples/inputs/fullspace_query.npz --device cpu --output outputs/full_space.tsv
```

This uses four prepared queries included in the code checkout; see
[example provenance](../../examples/README.md). No result-data archive is
required. `--checkpoint` and `--target-table` select compatible alternative
assets. The default batch size is 256.

## Prepare your response table

For a named table of your own:

```bash
metar predict full-space --input inputs/full_space_queries.tsv --id-column query_id --feature-order inputs/landmark_order.txt --cell-line PC3 --device cpu --output outputs/full_space.tsv
```

`--cell-line` records output metadata without selecting another model.

Use the [bulk file layout](bulk.md#replace-the-example-with-your-data): one query
per row, unique `query_id`, and the 978 training landmark response features.
CSV/TSV, their gzip forms, Parquet, `.npy` and `.npz` are supported. Named table
columns can be aligned with `--feature-order inputs/landmark_order.txt`, a
text file containing one actual landmark name per line with no header. Unnamed
NumPy arrays must already follow that order. All values must be finite.

This branch uses the prepared bulk response **without the additional
drug-cell-mean standardization used by the 668-target scorer**. Use the
full-space preparation for this model, including its response scale and feature
order.

The small bundled query input is prepared from the manuscript's full-space
input, without the additional 668-target transformation. For the full data,
follow the [data guide](../data.md#place-separately-obtained-public-inputs);
the evaluation cache is derived from matched processed LINCS inputs. The
[PDF-3 application](bulk.md#optional-run-the-pdf-3-application-query) uses a
separate 668-target preparation. Use the matching full-space inputs for
genome-wide inference.

## Candidate identifiers and target tables

The released CSV's first column supplies target names in output; the remaining
columns are finite numeric target embeddings. Names must be unique and the
embedding dimension must match the query checkpoint. Keep this table paired
with its model.

A custom `--target-table` must contain compatible target embeddings for the
selected candidates. Generate embeddings through the representation workflow
before adding genes absent from the release. The candidate set changes rank
denominators, so retain its size when reporting or comparing rankings.

## Validate and read predictions

```bash
metar doctor --scenario full-space --input examples/inputs/fullspace_query.npz
```

Doctor checks files and dimensions without inference. Check preprocessing
against the requirements above. Default output has one row per query/candidate
with `target`,
`score`, `rank` and `candidate_count=16583`. `--top-k 20` keeps a shortlist;
`--top-k 0` retains complete output for [consensus](consensus.md). Scores are
rankings, not binding probabilities. See the [output reference](../usage.md#read-outputs-and-check-inputs).
