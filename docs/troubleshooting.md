# Troubleshooting

Run commands from the repository root with the environment active. The
[quick start](quickstart.md) is a complete example with supplied data; paths
under `inputs/` in the [prediction guide](usage.md) refer to files you prepare.

## Installation and command discovery

| Symptom | What to check or do |
|---|---|
| `metar` is not found, or `No module named metar` | Activate `.venv`, run `uv pip install -e .` from the repository root, then run `metar describe`. Check `python -c "import sys; print(sys.executable)"` to confirm the interpreter. |
| PowerShell blocks `Activate.ps1` | Use `.\.venv\Scripts\python.exe` and `.\.venv\Scripts\metar.exe` directly; add `--python .venv` to the installation commands to target that environment. |
| H5AD or Parquet dependencies are missing | Install the corresponding extra in the same environment: `uv pip install -e ".[single-cell,parquet]"`. |
| GPU was requested but CUDA is unavailable | Check `python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"`. Install a compatible wheel using the [PyTorch selector](https://pytorch.org/get-started/locally/), or run explicitly with `--device cpu`. |

## Downloads and artifact paths

A message such as `bulk checkpoint is required because the hydrated default is
unavailable` usually means the model is not extracted where the CLI expects it.
Run `metar doctor --scenario bulk-668` and inspect the reported paths. Set
`METAR_ARTIFACT_ROOT` to the extracted `metar_artifacts` directory, not the code
repository, the download directory, or the `checkpoints` subdirectory. See
[data setup](data.md#set-the-artifact-root).

If a download fails, rerun the same command. Existing archives are checked
again. A size or SHA-256 mismatch means that archive is incomplete or differs
from the release: move only the named archive out of the download directory
and download it again. The extractor refuses conflicting existing files;
extract into a new artifact directory when comparing different releases.
The public release supports anonymous HTTPS; no saved GitHub credential is
needed. Assets can also be downloaded from the
[release page](https://github.com/org-Metar/MeTar/releases/tag/artifacts-20261102)
and placed in the selected `--directory` before rerunning without `--download`.

## Query preparation

| Symptom | Resolution |
|---|---|
| Wrong feature count or missing landmark columns | Supply the correct 978-feature schema with `--feature-order` for bulk tables. Exclude numeric metadata such as dose/time. Check names, order and the workflow's preprocessing, rather than padding or dropping arbitrary genes. |
| A bulk input passes `doctor` but results differ from expectations | `doctor` validates files, dimensions and finite values; it cannot identify whether the data use training-matched normalization. Check the [bulk preparation contract](usage.md#1-bulk-expression-668-targets). |
| A raw single-cell H5AD has more than 128 columns | Apply the saved [Tahoe X128 projection](usage.md#prepare-x128-from-raw-tahoe-plates) before inference. Selecting 128 genes or fitting a new SVD does not reproduce the trained coordinates. |
| The Tahoe preparer rejects the gene order or sparse encoding | It requires the original ordered 62,710-gene Tahoe schema and CSR `.X`. Check that the original plate and its expression representation were retained; the supplied preparer does not align arbitrary single-cell datasets. |
| Proteomics features are missing | Use the feature identifiers required by the saved preprocessing JSON and the training-compatible abundance representation. Do not apply its saved log transform or standardization a second time. |
| An input contains nonfinite values | Find the affected rows/features and correct their source preparation before bulk/single-cell prediction. |
| PDF-3 query preparation says the output already exists | Reuse that prepared CSV, or choose a new output filename. The preparer deliberately does not overwrite it. |

## Adaptation and consensus

Adapted cross-cell prediction requires `--support-input` and `--support-labels`.
Use independent support compounds from the same cell line, with IDs matching
between response and label tables. Query IDs and support IDs must be disjoint,
and each support compound needs a target annotation in the selected 668-target
vocabulary. Run `metar doctor --scenario cross-cell --mode adapted` with the
same query, support and label arguments to inspect the input files.

For consensus, retain full long-format predictions using `--top-k 0`.
Every input must use the same compound IDs and complete candidate set. Errors
about truncated vectors, duplicate drug-cell-target scores, or different target
spaces require fixing those input tables; changing the aggregation method will
not fix them. If no compound meets `--min-cell-lines`, check matching IDs and
cell-line labels. See the [consensus example](usage.md#6-consensus-across-cell-lines).

## Memory and result comparison

For bulk or proteomics inference, lower `--batch-size` to reduce model-batch
memory. Single-cell H5AD loading can densify the full `.X` matrix before cell
sampling; create an input containing only the query groups you need if that
load exceeds memory. `--max-cells` controls prediction-time sampling, not the
initial H5AD load.

For numerical differences between systems, use the documented
[result comparison workflow](reproduction.md#numerical-and-preparation-limits).
Fresh predictions, archived scores and the supplied computational plot tables
serve different purposes. Training/reproduction workspace scripts that call
Bash should be run in Linux or WSL; see [training](training.md) and
[result reproduction](reproduction.md).

If the issue persists, open a [GitHub issue](https://github.com/org-Metar/MeTar/issues)
with the command, relevant error, Python/PyTorch versions and `doctor` output.
Share a minimal input schema or non-sensitive example rather than private assay
data or credentials.
