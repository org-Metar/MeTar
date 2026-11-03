# MeTar

MeTar ranks drug targets from perturbation-response data. It supports bulk
expression, prediction in new cell lines with or without support compounds,
single-cell data, proteomics, and consensus across cell lines.

[Installation](docs/installation.md) · [Prediction guide](docs/usage.md) ·
[Models and data](docs/data.md) · [Training](docs/training.md) · [Plotting](docs/plotting.md)

## Install and download the pretrained model

Run from the repository root. This CPU setup uses Python 3.11; see
[installation](docs/installation.md) for CUDA and optional dependencies.

```bash
uv venv --python 3.11
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[single-cell,parquet]"
```

Activate with `source .venv/bin/activate` on Linux/macOS or
`.\.venv\Scripts\Activate.ps1` in Windows PowerShell. Download the public models
and example over HTTPS; a GitHub account or GitHub CLI is not required:

```bash
python scripts/download_data.py --archive models_and_representations.tar.gz --archive result_data.tar.gz --download --directory ../metar-downloads --package ../metar_artifacts
```

Set `METAR_ARTIFACT_ROOT` to the absolute path of that `metar_artifacts` directory:

```bash
# Linux/macOS
export METAR_ARTIFACT_ROOT=/absolute/path/to/metar_artifacts
```

```powershell
# Windows PowerShell
$env:METAR_ARTIFACT_ROOT = 'D:/datasets/metar_artifacts'
```

## Run the included PDF-3 query

The two archives above contain a real, prepared 978-feature PDF-3 query:

```bash
python scripts/prepare_pdf3.py --package ../metar_artifacts --output ../metar-example/pdf3.csv
metar doctor --scenario bulk-668 --input ../metar-example/pdf3.csv
metar predict bulk-668 --input ../metar-example/pdf3.csv --id-column query_id --cell-line JHH7 --top-k 20 --output ../metar-example/pdf3_targets.tsv
```

This uses the retained PDF-3 log2-fold-change input contract. Independent new
assays still need the model's matching preprocessing; 978 named columns alone
do not establish that a query is ready.

## Choose a prediction workflow

Paths under `inputs/` below are files you prepare, not bundled example datasets.
The [prediction guide](docs/usage.md) explains their exact formats and preparation.

```bash
# Bulk expression: 668 targets
metar predict bulk-668 --input inputs/bulk.tsv --id-column query_id --feature-order inputs/landmark_order.txt --cell-line PC3 --output outputs/bulk.tsv

# Genome-wide target ranking, including targets outside the supervised set
metar predict full-space --input inputs/full_space.tsv --id-column query_id --feature-order inputs/landmark_order.txt --output outputs/full_space.tsv

# A new cell line: direct prediction
metar predict cross-cell --mode direct --cell-line VCAP --input inputs/vcap.tsv --id-column query_id --feature-order inputs/landmark_order.txt --output outputs/vcap_direct.tsv

# A new cell line: prediction with separate labeled support compounds
metar predict cross-cell --mode adapted --cell-line VCAP --input inputs/vcap.tsv --id-column query_id --feature-order inputs/landmark_order.txt --support-input inputs/support.tsv --support-labels inputs/support_labels.tsv --output outputs/vcap_adapted.tsv

# Single-cell: A549, SW480 or NCI-H460, using the trained X128 coordinates
metar predict single-cell --input inputs/a549_x128.h5ad --id-column drugname_drugconc --cell-line A549 --output outputs/single_cell.tsv

# Protein abundance; the saved preprocessor selects and transforms its features
metar predict proteomics --input inputs/proteomics.tsv --id-column query_id --output outputs/proteomics.tsv

# Combine complete predictions for the same compounds across cell lines
metar predict consensus --input outputs/pc3.tsv --input outputs/a375.tsv --output outputs/consensus.tsv
```

Prediction defaults to CPU; add `--device cuda:0` for a CUDA-enabled environment.
Keep `--top-k 0` for the complete score vectors needed by consensus. Use
`--top-k 20` for a short report. Scores and rank percentages are not binding
probabilities. Cross-cell adaptation applies the trained adapter during
inference; it does not retrain model parameters.

## Training, plots and result comparison

| Task | Entry point | Guide |
|---|---|---|
| Download data and check readiness | `python scripts/download_data.py --list`, `metar doctor --help` | [Data](docs/data.md) |
| Train from scratch | `python scripts/prepare_training.py --help`, `metar train --help` | [Training](docs/training.md) |
| Draw computational manuscript figures | `python scripts/plot_figures.py list` | [Plotting](docs/plotting.md) |
| Compare with published results | `python scripts/reproduce_results.py --help` | [Reproduction](docs/reproduction.md) |

The Python API is `import metar`. Model code is in `src/metar/`, scientific
training code in `training/`, preprocessing in `preprocessing/`, analysis in
`analysis/`, and numerical figure code in `figures/`. Large checkpoints and
datasets are downloaded separately. Laboratory figure reproduction is outside
the scope of this code package.

Preprocessing and cross-environment numerical limits are described in the
[reproduction guide](docs/reproduction.md). No project-wide MeTar license has
been approved in this snapshot; distributed third-party terms are retained in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
