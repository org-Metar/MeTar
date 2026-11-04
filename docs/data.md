# Models and data setup

Keep the code checkout and artifact root in separate directories. Large model
weights, prepared datasets and reference scores are downloaded from the
[GitHub artifact release](https://github.com/org-Metar/MeTar/releases/tag/artifacts-20261102).
The public release can be downloaded anonymously over HTTPS.

## Download what you need

List the assets without installing a download tool or signing in:

```bash
python scripts/download_data.py --list
```

For the small prediction examples in [`examples/`](../examples/README.md),
download the model archive:

```bash
python scripts/download_data.py --archive models_and_representations.tar.gz --download --directory ../metar-downloads --package ../metar_artifacts
```

These examples contain prepared computational inputs and do not require the
complete training datasets. For the [PDF-3 reproduction](quickstart.md), also
download the reference-data archive:

```bash
python scripts/download_data.py --archive models_and_representations.tar.gz --archive result_data.tar.gz --download --directory ../metar-downloads --package ../metar_artifacts
```

For all active assets:

```bash
python scripts/download_data.py --download --directory ../metar-downloads --package ../metar_artifacts
```

The downloader checks each archive's size and SHA-256 before extraction. All
archives are overlaid into the same `--package` directory. Identical existing
files are reused; conflicting content is rejected. Omit `--download` to verify
and extract archives already in `--directory`. Repeat `--archive NAME` to select
individual assets. Public downloads use Python's standard library. For private
releases, or when `--github-user LOGIN` is specified, the downloader uses
[GitHub CLI](https://cli.github.com/) with an existing login. That option selects
a saved credential for this subprocess without changing the default account.

| Archive | Use |
|---|---|
| `models_and_representations.tar.gz` | Pretrained checkpoints, target embeddings, saved proteomics preprocessing and protocol/split records |
| `runtime_sources.tar.gz` | Preparation and evaluation code for the archived model comparison |
| `result_data.tar.gz` | Reference scores, computational source tables and the prepared PDF-3 query |
| `application_inputs.tar.gz` | Processed PDF-3 FPKM table and ordered 668-target metadata for prediction input preparation |
| `figure_assets.tar.gz` | Supported computational figure baselines and native diagram/PPTX inputs |
| `tahoe_projection.tar.gz` | Two NPZ files containing the fixed gene selection and saved SVD projection |
| `training_inputs.tar.gz` | Gene-relation graph and PC3 response/target-header inputs for scratch training |

The active asset list and automatic integrity checks are configured in
[`artifact_downloads.json`](../manifests/artifact_downloads.json).
Laboratory figures and their reproduction are not included in this code package.
The Tahoe preprocessing recipe is in `preprocessing/tahoe_projection.ipynb` in
the code checkout, rather than in the projection archive.

For an already prepared query, the model archive is usually sufficient. Bulk
query tables also need an explicit training feature order or the original
landmark schema. Full manuscript evaluation and scratch training need the
prepared public inputs below; downloading pretrained weights alone does not
prepare these datasets. To train only a selected component, use the
[training workspace options](training.md) to check that component's inputs
instead of collecting every dataset first.

## Set the artifact root

Extracted files share one root:

```text
metar_artifacts/
  checkpoints/selected/
  data/application_inputs/pdf3/
  data/processed/fullspace/
  data/processed/lincs/
  data/processed/tahoe/
  data/processed/proteomics/
  data/preprocessing/tahoe/
  data/plot_assets/
  manifests/
```

Point the CLI to its absolute path:

```bash
# Linux/macOS
export METAR_ARTIFACT_ROOT=/absolute/path/to/metar_artifacts
```

```powershell
# Windows PowerShell
$env:METAR_ARTIFACT_ROOT = 'D:/datasets/metar_artifacts'
```

These environment variables apply to the current terminal and its child
processes. Explicit `--checkpoint`, `--target-table` and `--preprocessor` options
can override defaults; see `metar predict SCENARIO --help`.

## Place separately obtained public inputs

The importer places existing prepared inputs; it does not download or prepare
every upstream public dataset. A file downloaded directly from LINCS, STRING,
Tahoe or ProteomicsDB is not necessarily equivalent to the selected processed
file. Use the exact inputs defined by
[`pretrained_artifact_contract.json`](../manifests/pretrained_artifact_contract.json).

Arrange existing files under a source directory using these relative paths:

| Use | Relative path |
|---|---|
| Bulk/cross-cell | `data/processed/lincs/large_data.parquet` |
| Bulk compound metadata | `data/processed/lincs/compoundinfo_beta.txt` |
| Bulk preparation graph | `data/processed/fullspace/pertkge_ppi_graph.csv` |
| Full-space evaluation cache | `data/processed/fullspace/h07_prepared_data.pt` |
| A549 X128 | `data/processed/tahoe/tahoe_a549.h5ad` |
| SW480 X128 | `data/processed/tahoe/tahoe_sw480.h5ad` |
| NCI-H460 X128 | `data/processed/tahoe/tahoe_nci_h460.h5ad` |
| Single-cell target labels | `data/processed/tahoe/tahoe_dti.csv` |
| Proteomics | `data/processed/proteomics/proteomicsdb_data.csv` |
| Landmark schema | `data/application_inputs/pdf3/geneinfo_beta.txt` |

```bash
python scripts/import_data.py --package ../metar_artifacts --source-root ../public-input-store --mode copy
python scripts/import_data.py --package ../metar_artifacts --source-root ../public-input-store --check-only
```

`--mode symlink` saves disk space when available. `--check-only` examines files
already present in the artifact root and writes nothing; it does not check
source files that have not been placed there.

A missing derived bulk cache is reported as `derive_with_build_h07_cache` and
does not by itself fail import readiness. Build it after placing the bulk inputs:

```bash
uv pip install -e ".[training]"
python scripts/prepare_bulk_data.py --package ../metar_artifacts --device cpu
```

Alternatively, use `--include-derived-source` to import an existing
integrity-matched cache. For raw Tahoe plates, follow the
[X128 preparation steps](usage.md#prepare-x128-from-raw-tahoe-plates) to reuse the
saved projection without fitting a new coordinate system. Scratch-training
inputs and workspace preparation are described in [training.md](training.md).

## Check readiness

```bash
metar doctor --scenario bulk-668
metar doctor --scenario single-cell --cell-line A549
metar doctor --scenario proteomics
metar doctor --help
```

Doctor checks required artifacts and, when given `--input`, the file's structure
and dimensions. It does not run inference or establish preprocessing compatibility
from shape alone. See [usage.md](usage.md) for query formats and
[reproduction.md](reproduction.md) for the full comparison workflow.
