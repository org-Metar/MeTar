<p align="center">
  <img src="docs/images/metar-mark.svg" width="104" height="104" alt="MeTar">
</p>

<h1 align="center">MeTar</h1>

<p align="center">
  <strong>Generalizable phenotype-inverse identification of compound targets<br>
  from perturbational omics</strong>
</p>

<p align="center">
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/version-0.2.0-244B63?style=flat-square" alt="Version 0.2.0"></a>
  <a href="docs/installation.md"><img src="https://img.shields.io/badge/Python-3.11%20recommended-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python 3.11 recommended"></a>
  <a href="https://github.com/org-Metar/MeTar/releases/tag/artifacts-20261102"><img src="https://img.shields.io/badge/models%20%26%20data-download-287F79?style=flat-square" alt="Download pretrained models and data"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/paper-Submitted-687B89?style=flat-square" alt="Paper submitted"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-287F79?style=flat-square" alt="License Apache 2.0"></a>
</p>

---

<a id="contents"></a>

**[1 · Overview](#overview) / [2 · Setup](#installation) / [3 · Predict](#prediction-workflows) / [4 · Train](#training-and-reproduction) / [5 · Reproduce](#paper-reproduction)**

<a id="overview"></a>

## 1 · Overview

MeTar predicts drug targets from drug-induced perturbation responses. It learns
response and target embeddings in a shared embedding space, using known
drug-target associations as supervision, and ranks candidates by similarity.

Beyond prediction in the training setting, MeTar supports three directions
of generalization:

- **New cell lines.** Apply the shared model directly, or use reference
  compounds with known targets to condition a trained adapter for the target
  cell line.
- **Genome-wide target space.** A graph neural network shares parameters across
  genes, enabling ranking beyond the 668 supervised targets over 16,583 candidates.
- **New assay modalities.** Modality-specific encoders map single-cell
  transcriptomic responses and protein abundance into the learned target space.

<p align="center">
  <img src="docs/images/training_workflow.png" width="1100" alt="MeTar training workflow: cell-line tasks, response encoder and support-conditioned adapter, PPI graph encoder, shared embeddings, similarity scoring, supervised loss and parameter update.">
  <br>
  <sub>Response and target representations are learned together using known drug-target associations.</sub>
</p>

The manuscript calls prediction without reference-compound adjustment
**MeTar_direct** and prediction with it **MeTar_adapted**. Predictions for the
same compound can also be combined across cell lines.

<details>
<summary><strong>View the application scenarios</strong></summary>

<p align="center">
  <img src="docs/images/application_scenarios.png" width="1100" alt="MeTar applications: a new drug, a new cell line with support compounds, genome-wide target ranking, and single-cell or protein-abundance readouts.">
</p>

Both overview images are excerpts from the manuscript figure.

</details>

---

<a id="installation"></a>

## 2 · Setup

### Install

Python **3.11 is recommended**. Create and activate an environment with
[uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/org-Metar/MeTar.git
cd MeTar
uv venv --python 3.11
source .venv/bin/activate
```

<details>
<summary><strong>Windows PowerShell</strong></summary>

Use `.\.venv\Scripts\Activate.ps1` instead of the final activation command.

</details>

Install CPU PyTorch and MeTar:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[single-cell,parquet]"
metar describe
```

For CUDA, choose a wheel from the
[PyTorch installation selector](https://pytorch.org/get-started/locally/).
Prediction and training default to `--device auto`; use `--device cpu` to
require CPU or `--device cuda:0` to require the first visible GPU. See
[runtime reporting and device selection](docs/runtime.md) for progress, logs
and CUDA errors. The [installation guide](docs/installation.md) covers other
dependencies and platform details.

### Download the pretrained models

The small computational examples are included in [`examples/`](examples/README.md).
The model-based examples require the model archive:

```bash
python scripts/download_data.py --archive models_and_representations.tar.gz --download --directory ../metar-downloads --package ../metar_artifacts
export METAR_ARTIFACT_ROOT="$(cd ../metar_artifacts && pwd)"
```

<details>
<summary><strong>Windows PowerShell artifact path</strong></summary>

```powershell
$env:METAR_ARTIFACT_ROOT = (Resolve-Path ../metar_artifacts).Path
```

</details>

Downloads are public; the downloader verifies them before extraction. Keep the
artifact directory separate from the code and set `METAR_ARTIFACT_ROOT` again
in each new terminal. [Models and data](docs/data.md) lists additional downloads
for preprocessing, training and paper reproduction.

---

<a id="quick-start"></a>
<a id="prediction-workflows"></a>

## 3 · Predict with pretrained models

Start with the basic bulk interface, then choose the generalization setting
you need. Run commands from the repository root after setup. Each guide
explains the setting, model selection, input format and how to replace the
example with your own data.

<a id="bulk-transcriptomics"></a>

### 3.1 · Bulk transcriptomics · 668 targets

Rank targets from a prepared 978-feature bulk response without retraining.
The bundled HELA profiles demonstrate the basic interface; a prepared response
from a training cell line such as PC3 uses the same command with its own input
and cell-line name.

```bash
metar predict bulk-668 --input examples/inputs/bulk_query.npz --cell-line HELA --output outputs/bulk.tsv
```

**[Bulk guide →](docs/prediction/bulk.md)** · response preparation, landmark
order, named tables/arrays, and substituting your own query.

<a id="cross-cell-prediction"></a>

### 3.2 · New cell lines · direct or adapted

For a new cell line, apply the model directly or supply separate reference
compounds and their known targets. Adapted prediction uses the trained adapter
at inference with fixed model parameters.

```bash
metar predict cross-cell --mode direct --cell-line HELA --input examples/inputs/cross_cell/hela/query.npz --output outputs/hela_direct.tsv
metar predict cross-cell --mode adapted --cell-line HELA --input examples/inputs/cross_cell/hela/query.npz --support-input examples/inputs/cross_cell/hela/support.npz --support-labels examples/inputs/cross_cell/hela/support_labels.tsv --support-weights examples/inputs/cross_cell/hela/support_weights.npy --output outputs/hela_adapted.tsv
```

**[Cell-line guide →](docs/prediction/cross-cell.md)** · query/reference formats,
target labels, support weights and keeping query compounds separate.

<a id="genome-wide-ranking"></a>

### 3.3 · Genome-wide target space · 16,583 genes

Use the full-space model to rank all candidate genes, including those outside
the supervised target set. This branch has its own response-preparation contract.

```bash
metar predict full-space --input examples/inputs/fullspace_query.npz --output outputs/fullspace.tsv
```

**[Target-space guide →](docs/prediction/genome-wide.md)** · full-space inputs,
target embeddings and the distinction from the 668-target bulk branch.

<a id="single-cell-prediction"></a>

### 3.4 · Single-cell transcriptomics

Use the released assay encoder for A549, SW480 or NCI-H460. The input contains
grouped cells in the saved 128-dimensional representation, **X128**.

```bash
metar predict single-cell --input examples/inputs/single_cell_a549.npz --cell-line A549 --max-cells 0 --output outputs/single_cell.tsv
```

**[Single-cell guide →](docs/prediction/single-cell.md)** · cell/group IDs,
H5AD/NPZ formats, the saved projection and preparing your own cells.

<a id="proteomics-prediction"></a>

### 3.5 · Protein abundance

Use the proteomics encoder with its saved feature selection and preprocessing.
The example has already been converted to the expected abundance representation.

```bash
metar predict proteomics --input examples/inputs/proteomics.tsv --id-column query_id --output outputs/proteomics.tsv
```

**[Proteomics guide →](docs/prediction/proteomics.md)** · protein feature names,
units, missing values and matching the saved transform.

<a id="consensus-prediction"></a>

### 3.6 · Combine predictions across cell lines

Aggregate complete rankings for the same compounds and candidate set. This
example combines two included prediction tables; to use your own, first save
each cell line's complete ranking with `--top-k 0`.

```bash
metar predict consensus --input examples/inputs/consensus/hela_scores.tsv --input examples/inputs/consensus/hcc515_scores.tsv --output outputs/consensus.tsv
```

**[Consensus guide →](docs/prediction/consensus.md)** · matching query IDs,
complete score vectors, candidate order and aggregation methods.

All prediction routes write ranked TSV tables. Higher scores rank first;
`rank_percent` is a relative rank, not a binding probability. Use `--top-k 0`
for all candidates. The [prediction reference](docs/usage.md) covers shared
output fields and the Python API. For errors, see
[troubleshooting](docs/troubleshooting.md).

---

<a id="training-and-reproduction"></a>

## 4 · Training and model reuse

Training workflows range from reusing the learned target space to learning the
complete model. Choose by **which components need new training**. Install the
training extra first:

```bash
uv pip install -e ".[training]"
```

The commands below create separate workspaces and use prepared training data
under `../metar_artifacts`. The linked guides list exactly which inputs are
needed and how to supply compatible data of your own. Add `--check-only` to a
preparation command to list missing inputs without writing. Preparation copies
affected inputs when symlinks are unavailable; use `--copy-inputs` to copy all
inputs. See [training setup](docs/training.md) for workspace details.

### 4.1 · Train a single-cell encoder · reuse target embeddings

Train the single-cell response encoder against the fixed released target
embeddings.

```bash
python scripts/prepare_training.py --artifacts ../metar_artifacts --output ../metar-train-sc --stage single-cell --cell-line A549 --reuse-pretrained
metar train --stage single-cell --cell-line A549 --workspace ../metar-train-sc --device cuda:0
```

**[Single-cell training guide →](docs/training/single-cell.md)** · learned/frozen
components, X128 features, training labels and replacing the cell dataset.

### 4.2 · Train a protein-abundance encoder · reuse target embeddings

Train the proteomics encoder using the stored bulk target vectors. The selected
configuration can also train a target adapter. The GNN remains fixed.

```bash
python scripts/prepare_training.py --artifacts ../metar_artifacts --output ../metar-train-proteomics --stage proteomics --reuse-pretrained
metar train --stage proteomics --workspace ../metar-train-proteomics --device cuda:0
```

**[Proteomics training guide →](docs/training/proteomics.md)** · abundance units,
labels, feature selection, preprocessing and the target adapter.

### 4.3 · Train cell-line adaptation · reuse the bulk model

Retrain the episodic adapter when the adaptation training task itself changes.
For ordinary prediction with support compounds, use section 3.2 instead.

```bash
python scripts/prepare_training.py --artifacts ../metar_artifacts --output ../metar-train-adapter --stage cross-cell-adapter --reuse-pretrained
metar train --stage cross-cell-adapter --workspace ../metar-train-adapter --device cuda:0
```

**[Adapter training guide →](docs/training/cell-line-adapter.md)** · support/query
tasks, known-target labels, splits and upstream models.

### 4.4 · Train the genome-wide branch · initialize from the bulk model

Initialize from the pretrained bulk model, then train the response encoder and
full-space target tower together using the prepared gene graph and bulk
responses. This updates more of the model than the preceding routes.

```bash
python scripts/prepare_training.py --artifacts ../metar_artifacts --output ../metar-train-fullspace --stage full-space --reuse-pretrained
# If the prepared bulk cache is absent:
python scripts/prepare_bulk_data.py --package ../metar-train-fullspace --device cpu
metar train --stage full-space --workspace ../metar-train-fullspace --device cuda:0
```

**[Target-space training guide →](docs/training/target-space.md)** · graph,
candidate order, prepared bulk cache and the inherited training protocol.

### 4.5 · Train the complete model from scratch

Use this route to rebuild the gene prior, bulk model, adaptation and assay
branches. It requires the full prepared datasets and trains all stages.

```bash
python scripts/prepare_training.py --artifacts ../metar_artifacts --output ../metar-training
# If the prepared bulk cache is absent:
python scripts/prepare_bulk_data.py --package ../metar-training --device cpu
metar train --stage all --workspace ../metar-training --device cuda:0
```

**[Full training guide →](docs/training/full-model.md)** · data acquisition,
stage order, checkpoints and running individual stages.

These entry points use the selected scientific protocols. Changes to feature
spaces, candidate sets or split rules require corresponding protocol updates.
See the [training reference](docs/training.md) for workspace checks and outputs.

---

<a id="paper-reproduction"></a>

## 5 · Reproduce the paper

The following workflows are for checking the reported analyses and figures.
They are separate from using the pretrained models for new predictions.

| Reproduction task | Detailed instructions |
|---|---|
| PDF-3 target prediction from the prepared RNA-seq response | [PDF-3 walkthrough](docs/quickstart.md) |
| Model evaluation and comparisons | [Result reproduction](docs/reproduction.md) |
| Computational panels and figure assembly | [Figure reproduction](docs/plotting.md) |

Each guide lists the additional downloads and commands. Plotting uses the
numerical source tables in `data/plot_sources/`; laboratory figure reproduction
is outside this code package.

---

<a id="citation"></a>

## 📝 Citation

Manuscript citation:

```bibtex
@unpublished{metar_submitted,
  author  = {Li, Wannian and Zhang, Ying and Liu, Hong and Huang, Xun and Liu, Qi},
  title   = {Generalizable phenotype-inverse identification of compound targets from perturbational omics},
  year    = {2026},
  note    = {Manuscript submitted for publication}
}
```

Use GitHub's **Cite this repository** button to export the citation. Software
citation metadata is also available in [CITATION.cff](CITATION.cff).

---

### Questions

For usage questions or reproducible problems, open a
[GitHub issue](https://github.com/org-Metar/MeTar/issues) with the command,
package versions and error message. Do not include credentials or private data.

<a id="license"></a>

## 📄 License

MeTar's original source code is licensed under the
[Apache License 2.0](LICENSE).

Third-party code retains its original licenses and notices; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The source-code license does
not change the terms of external datasets, third-party model weights or
manuscript figures, which remain subject to their respective terms.

<p align="right"><a href="#contents">Back to contents ↑</a></p>
