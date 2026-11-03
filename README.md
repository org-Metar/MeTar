<h1 align="center">MeTar</h1>

<p align="center">
  <strong>Generalizable phenotype-inverse identification of compound targets<br>
  from perturbational omics</strong>
</p>

<p align="center">
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/version-0.2.0-326C88" alt="Package version 0.2.0"></a>
  <a href="docs/installation.md"><img src="https://img.shields.io/badge/Python-3.11%20recommended-3776AB?logo=python&amp;logoColor=white" alt="Python 3.11 recommended"></a>
  <a href="https://github.com/org-Metar/MeTar/releases/tag/artifacts-20261102"><img src="https://img.shields.io/badge/models%20%26%20data-download-2E8B73" alt="Download pretrained models and data"></a>
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="docs/quickstart.md">Quick start</a> ·
  <a href="docs/usage.md">Prediction guide</a> ·
  <a href="docs/training.md">Training</a> ·
  <a href="docs/plotting.md">Figures</a> ·
  <a href="#citation">Citation</a>
</p>

## Overview

MeTar predicts drug targets from drug-induced perturbation responses. It learns
response and target embeddings in a shared embedding space, using known
drug-target associations as supervision. A support-conditioned adapter uses
separate reference compounds with known targets to adjust predictions for a
cell-line context. A graph neural network uses shared parameters to learn
target embeddings across genes, enabling ranking in the genome-wide target
space.

<p align="center">
  <img src="docs/images/training_workflow.png" width="1200" alt="MeTar training workflow: cell-line tasks, response encoder and support-conditioned adapter, PPI graph encoder, shared embeddings, similarity scoring, supervised loss and parameter update.">
</p>

The workflow above shows the model's training strategy. The released interfaces
apply selected pretrained models to the following prediction tasks:

| Task | Command | Candidate space / required input |
|---|---|---|
| Bulk transcriptomic responses | `metar predict bulk-668` | 668 targets; prepared 978-feature response |
| A new cell line, direct prediction | `metar predict cross-cell --mode direct` | 668 targets; no support-compound input |
| A new cell line, adapted prediction | `metar predict cross-cell --mode adapted` | 668 targets; separate support responses and known targets |
| Genome-wide target ranking | `metar predict full-space` | 16,583 genes, including targets outside the supervised set |
| Single-cell responses | `metar predict single-cell` | 16,583 genes; trained X128 representation for A549, SW480 or NCI-H460 |
| Protein abundance | `metar predict proteomics` | 16,583 genes; features matched to the saved preprocessor |
| Consensus across cell lines | `metar predict consensus` | Complete predictions over the same candidate space |

<p align="center">
  <img src="docs/images/application_scenarios.png" width="1200" alt="MeTar application scenarios: rank a new drug's targets, use support compounds in a new cell line, search the genome-wide target space, and work with single-cell or protein-abundance readouts.">
</p>

In the manuscript, direct and support-conditioned prediction are referred to
as **MeTar_direct** and **MeTar_adapted**, respectively.

The overview images are excerpts from the MeTar manuscript figure. See the
[prediction guide](docs/usage.md) for the input preparation required by each
released model. The single-cell and proteomics branches use their own trained
encoders; a bulk checkpoint is not directly applied to these assays.

## Documentation

| Start here | What it covers |
|---|---|
| [Quick start](docs/quickstart.md) | Run the included PDF-3 query on CPU and read the ranked output |
| [Installation](docs/installation.md) | Python environment, PyTorch, optional dependencies and GPU setup |
| [Models and data](docs/data.md) | Downloadable archives, artifact directories and public input preparation |
| [Prediction guide](docs/usage.md) | Input formats and calls for every supported prediction scenario |
| [Training from scratch](docs/training.md) | Required datasets, stage order and isolated training workspaces |
| [Result reproduction](docs/reproduction.md) | Archived evaluation workflows and numerical comparison limits |
| [Computational figures](docs/plotting.md) | Available plots, source tables, diagram assets and rendering commands |
| [Troubleshooting](docs/troubleshooting.md) | Missing artifacts, feature mismatches, X128 inputs and other common errors |

## Installation

Clone the repository and create a Python 3.11 environment with
[uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/org-Metar/MeTar.git
cd MeTar
uv venv --python 3.11
```

Activate it before using the `python` or `metar` commands below:

```bash
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Install the CPU version of PyTorch and MeTar:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[single-cell,parquet]"
metar describe
```

For CUDA, choose the appropriate wheel using the
[PyTorch installation selector](https://pytorch.org/get-started/locally/), then
install MeTar in that environment. Prediction defaults to CPU; add
`--device cuda:0` to request the GPU. See [installation](docs/installation.md)
for optional training, preprocessing, plotting and test dependencies.

## Download pretrained models and data

The [artifact release](https://github.com/org-Metar/MeTar/releases/tag/artifacts-20261102)
contains pretrained checkpoints, target embeddings, fitted preprocessing
parameters and computational reference data. Downloads are public and require
no GitHub login. Code and large data files live in separate directories.

For the runnable PDF-3 example, download and extract these two archives:

```bash
python scripts/download_data.py --archive models_and_representations.tar.gz --archive result_data.tar.gz --download --directory ../metar-downloads --package ../metar_artifacts
```

The downloader verifies size and SHA-256 before extraction. Set
`METAR_ARTIFACT_ROOT` to the extracted artifact directory:

```bash
# Linux / macOS, run from the repository root
export METAR_ARTIFACT_ROOT="$(cd ../metar_artifacts && pwd)"
```

```powershell
# Windows PowerShell, run from the repository root
$env:METAR_ARTIFACT_ROOT = (Resolve-Path ../metar_artifacts).Path
```

| Additional archive | Needed for |
|---|---|
| `application_inputs.tar.gz` | Processed PDF-3 RNA-seq input and target identifiers |
| `tahoe_projection.tar.gz` | Fixed 6,000-gene selection and fitted 128-dimensional single-cell SVD projection |
| `figure_assets.tar.gz` | Native diagram inputs and computational figure assembly |
| `runtime_sources.tar.gz` | Archived training and evaluation runtime sources |
| `training_inputs.tar.gz` | Graph and PC3 inputs for scratch training |

Select an archive with `--archive NAME`, or list all assets with
`python scripts/download_data.py --list`. The [data guide](docs/data.md)
explains which upstream datasets must be obtained separately.

## Quick start: predict targets for PDF-3

After installation, downloading the two example archives and setting
`METAR_ARTIFACT_ROOT`, run:

```bash
python scripts/prepare_pdf3.py --package ../metar_artifacts --output ../metar-example/pdf3.csv
metar doctor --scenario bulk-668 --input ../metar-example/pdf3.csv
metar predict bulk-668 --input ../metar-example/pdf3.csv --id-column query_id --cell-line JHH7 --device cpu --top-k 20 --output ../metar-example/pdf3_targets.tsv
```

This example uses the archived, prepared PDF-3 response with 978 features. The
output is a table of ranked targets, with columns including `query_id`,
`target`, `score`, `rank` and `rank_percent`. Higher scores rank first;
`rank_percent` is a relative rank, not a binding probability. The
[step-by-step tutorial](docs/quickstart.md) explains the files and output.

## Predict with your own data

The paths below are user-prepared inputs. Each branch has a specific feature
order and preprocessing contract; follow the corresponding section of the
[prediction guide](docs/usage.md).

```bash
# Bulk transcriptomics: 668 targets
metar predict bulk-668 --input inputs/bulk.tsv --id-column query_id --feature-order inputs/landmark_order.txt --cell-line PC3 --output outputs/bulk.tsv

# New cell line: direct prediction
metar predict cross-cell --mode direct --cell-line VCAP --input inputs/vcap.tsv --id-column query_id --feature-order inputs/landmark_order.txt --output outputs/vcap_direct.tsv

# New cell line: prediction with separate labeled support compounds
metar predict cross-cell --mode adapted --cell-line VCAP --input inputs/vcap.tsv --id-column query_id --feature-order inputs/landmark_order.txt --support-input inputs/support.tsv --support-labels inputs/support_labels.tsv --output outputs/vcap_adapted.tsv

# Genome-wide ranking
metar predict full-space --input inputs/full_space.tsv --id-column query_id --feature-order inputs/landmark_order.txt --output outputs/full_space.tsv

# Single-cell: trained X128 coordinates, not 128 selected genes
metar predict single-cell --input inputs/a549_x128.h5ad --id-column drugname_drugconc --cell-line A549 --output outputs/single_cell.tsv

# Protein abundance: reuse the saved feature selection and transforms
metar predict proteomics --input inputs/proteomics.tsv --id-column query_id --output outputs/proteomics.tsv

# Consensus: complete predictions for the same compounds across cell lines
metar predict consensus --input outputs/pc3.tsv --input outputs/a375.tsv --output outputs/consensus.tsv
```

For consensus, retain all candidates with `--top-k 0` in the input predictions.
Do not mix 668-target and 16,583-gene outputs. Cross-cell adaptation applies the
trained adapter at inference; it does not retrain the model for the new cell
line. Use `metar predict SCENARIO --help` for available options.

The Python API is also available:

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

Prediction functions return `PredictionResult`, with `.to_long()`, `.to_wide()`
and `.write()` methods. The CLI handles file loading and default artifact paths;
the Python functions accept prepared data and explicit model paths.

## Training and reproducing results

The [training guide](docs/training.md) describes the scientific training chain:

```text
gene-prior -> closed-bulk -> support-initializer -> cross-cell-adapter
                                                                  |
                                                              full-space
                                                               |     |
                                                      single-cell   proteomics
```

Prepare a separate workspace and inspect the commands before starting training:

```bash
uv pip install -e ".[training]"
python scripts/prepare_training.py --artifacts ../metar_artifacts --output ../metar-training --check-only
# After all prerequisites pass:
python scripts/prepare_training.py --artifacts ../metar_artifacts --output ../metar-training
metar train --stage all --workspace ../metar-training --device cuda:0 --dry-run
```

`--check-only` checks readiness and `--dry-run` prints commands; neither trains
models. The complete chain requires the prepared public inputs listed in the
training guide. For evaluation of the released models and comparison with
archived results, see [result reproduction](docs/reproduction.md).

## Computational figures

List supported figures, check their inputs, and render a panel:

```bash
uv pip install -e ".[plotting]"
python scripts/plot_figures.py list
python scripts/plot_figures.py check fig2g
python scripts/plot_figures.py render fig2g --output outputs/fig2g
```

Plotting requires Arial; `--font-dir` can point to an installed font directory.
Numerical source tables are included in `data/plot_sources/`. Some figure
assemblies also require `figure_assets.tar.gz`. See the
[figure guide](docs/plotting.md) for the supported routes. Laboratory figure
reproduction is outside this code package.

## Repository structure

```text
src/metar/           Prediction API, command-line interface and model components
training/            Scientific training stages and selected configurations
preprocessing/       Tahoe projection recipe and single-cell assembly
analysis/            Computational analysis and evaluation code
scripts/             Data preparation, evaluation and plotting entry points
figures/             Computational figure code and plot catalog
data/plot_sources/   Numerical source tables used by the figure code
manifests/           Model selections, artifact locations and input contracts
docs/                Tutorials, workflow guides and overview images
tests/               Interface and input-validation tests
comparators/         Third-party comparator sources and their licenses
vendor/              Shared historical runtime helpers
```

Large models and datasets are downloaded separately rather than stored in the
Git working tree.

## Citation

If you use MeTar, cite the software using [CITATION.cff](CITATION.cff), or the
**Cite this repository** button on GitHub. The associated manuscript is:

> Wannian Li, Ying Zhang, Hong Liu, Xun Huang and Qi Liu.
> *Generalizable phenotype-inverse identification of compound targets from
> perturbational omics*. Manuscript, 2026.

```bibtex
@unpublished{li2026metar,
  author = {Li, Wannian and Zhang, Ying and Liu, Hong and Huang, Xun and Liu, Qi},
  title  = {Generalizable phenotype-inverse identification of compound targets from perturbational omics},
  year   = {2026},
  note   = {Manuscript}
}
```

The manuscript citation will be updated when a public preprint or publication
identifier is available.

## Questions and licensing

For usage questions or reproducible problems, open a
[GitHub issue](https://github.com/org-Metar/MeTar/issues) with the command,
package versions and error message. Do not include credentials or private data.

Third-party licenses and provenance are retained in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No project-wide MeTar license
has been approved for this snapshot.
