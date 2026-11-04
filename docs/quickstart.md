# Reproduce the PDF-3 target prediction

This paper-reproduction example takes one prepared PDF-3 response from the release data and ranks
668 candidate targets with the pretrained bulk model. It needs no separately
downloaded LINCS or Tahoe dataset and does not train a model. For a first
prediction with a small computational example or your own data, start with
[pretrained prediction](../README.md#prediction-workflows).

## 1. Install and open the repository

Complete [installation](installation.md), activate the environment and run the
commands below from the repository root. The `python` and `metar` commands must
belong to that environment.

## 2. Download the model and example data

```bash
python scripts/download_data.py --archive models_and_representations.tar.gz --archive result_data.tar.gz --download --directory ../metar-downloads --package ../metar_artifacts
```

The two compressed archives total about 361 MB. The downloader verifies file
sizes and SHA-256 values, then extracts them under `../metar_artifacts`.
No GitHub login or GitHub CLI installation is required. Keep that directory
separate from the code checkout.

Set its absolute path in the current terminal:

**Linux/macOS (Bash or Zsh):**

```bash
export METAR_ARTIFACT_ROOT="$(cd ../metar_artifacts && pwd)"
```

**Windows PowerShell:**

```powershell
$env:METAR_ARTIFACT_ROOT = (Resolve-Path ../metar_artifacts).Path
```

Set this variable again in each new terminal before running MeTar. It tells the
CLI where to find the pretrained checkpoints, embeddings and preprocessing files.

## 3. Prepare and inspect the query

```bash
python scripts/prepare_pdf3.py --package ../metar_artifacts --output ../metar-example/pdf3.csv
metar doctor --scenario bulk-668 --input ../metar-example/pdf3.csv --id-column query_id
```

The first command writes one query row with a `query_id` column and 978 ordered
landmark features. The second prints a JSON report with `"status": "ok"`,
`"input_checked": true` and `"model_loaded": false` when its file and schema
checks pass. It checks readiness without running inference.

The example already has the model application's aligned log2-fold-change
representation. It is not a raw-count RNA-seq input. For an independent assay,
follow the [bulk input preparation requirements](usage.md#1-bulk-expression-668-targets)
before substituting your own table.

## 4. Rank the candidates

```bash
metar predict bulk-668 --input ../metar-example/pdf3.csv --id-column query_id --cell-line JHH7 --device cpu --top-k 20 --output ../metar-example/pdf3_targets.tsv
```

A successful run prints a JSON summary with `"status": "ok"`, `"queries": 1`
and `"candidate_targets": 668`. The output file contains the 20 highest-scoring
targets for that query; `candidate_targets` remains 668 because all candidates
were scored before selecting the first 20 rows.

View the result:

```bash
python -c "import pandas as pd; p = pd.read_csv('../metar-example/pdf3_targets.tsv', sep='\t'); print(p[['query_id', 'target', 'score', 'rank', 'rank_percent']].to_string(index=False))"
```

| Column | Meaning |
|---|---|
| `query_id` | The response profile being analyzed |
| `target` | Candidate target identifier |
| `score` | Model similarity score; larger values rank first |
| `rank` | Position in the complete target ranking; 1 is best |
| `rank_percent` | `100 * rank / candidate_count`; smaller values rank nearer the top |

These scores are rankings, not binding probabilities. To save every candidate
rather than the first 20, run the same command with `--top-k 0` and a different
output filename. Complete outputs are needed for multi-cell-line consensus.

`prepare_pdf3.py` refuses to overwrite an existing query file. To repeat the
example, reuse the prepared file or choose a new `--output` path.

## 5. Use your own data

Use the [prediction guide](usage.md#select-a-workflow) to choose bulk,
genome-wide, direct/adapted cross-cell, single-cell, proteomics or consensus
prediction. Each workflow specifies its required input representation and
command. For raw Tahoe plates, use the released
[fixed X128 projection](usage.md#prepare-x128-from-raw-tahoe-plates).
For training and computational figure reproduction, see
[training](training.md), [result reproduction](reproduction.md) and
[plotting](plotting.md). If a command fails, start with
[troubleshooting](troubleshooting.md).
