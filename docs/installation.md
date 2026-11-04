# Installation

Python 3.11 is recommended for CPU and NVIDIA GPU workflows. Package metadata
accepts Python 3.10 or newer; third-party comparators have separate environment
requirements. Install the prediction package and model/data downloads
separately.

## 1. Get the code

Install [Git](https://git-scm.com/downloads) and
[uv](https://docs.astral.sh/uv/getting-started/installation/) 0.10.7 or newer,
then open a terminal:

```bash
git clone https://github.com/org-Metar/MeTar.git
cd MeTar
uv venv --python 3.11
```

Run the following commands from this `MeTar` directory. The public repository
and release assets support anonymous access.

## 2. Activate the environment

**Linux/macOS (Bash or Zsh):**

```bash
source .venv/bin/activate
```

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

If your PowerShell policy blocks activation, use the environment's executables
directly, for example `.\.venv\Scripts\python.exe` instead of `python` and
`.\.venv\Scripts\metar.exe` instead of `metar`. The `uv sync` commands below
target the project's `.venv` even without activation.

Activate this environment again whenever you open a new terminal. Check that
`python` points to it:

```bash
python -c "import sys; print(sys.executable)"
```

## 3. Choose an installation profile

The repository's `uv.lock` records both profiles. Each installs MeTar in
editable mode with H5AD (`single-cell`) and Parquet support. Choose one profile
for the environment; the CPU and CUDA groups are mutually exclusive.

### CPU

The default profile uses CPU PyTorch, including on machines with an NVIDIA GPU:

```bash
uv sync --locked
```

### NVIDIA GPU

Use the CUDA 12.6 profile on Linux or Windows with a compatible NVIDIA GPU and
driver. Disable the default groups when selecting `cu126`:

```bash
uv sync --locked --no-default-groups --group cu126
```

The profile chooses the PyTorch build; `--device` chooses the device used by a
command. Check driver compatibility with the official
[PyTorch installation guidance](https://pytorch.org/get-started/locally/).

### Locked PyTorch versions

| Profile | PyTorch build on Linux/Windows | Wheel index |
|---|---|---|
| Default `cpu` | `2.14.1+cpu` | [PyTorch CPU](https://download.pytorch.org/whl/cpu) |
| `cu126` | `2.14.1+cu126` | [PyTorch CUDA 12.6](https://download.pytorch.org/whl/cu126) |

Python 3.11 is the recommended interpreter for these profiles. Package metadata
continues to accept Python 3.10 or newer; wheel availability also depends on the
operating system and architecture. MeTar does not require `torchvision` or
`torchaudio`.

## 4. Verify the installation

```bash
python -c "import sys, torch; print(sys.version); print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('Visible GPUs:', torch.cuda.device_count())"
metar describe
metar predict --help
```

`metar describe` lists the six prediction workflows and available training
stages. It does not require downloaded weights. Prediction and training default
to `--device auto`, which selects CUDA when available, otherwise CPU. Use
`--device cpu` to require CPU or `--device cuda:0` to require the first visible
GPU; an unavailable explicit CUDA request is an error. See
[runtime reporting and device selection](runtime.md) for progress, quiet mode
and CUDA visibility.

Run the lightweight tests with the default CPU profile and development extra:

```bash
uv sync --locked --extra dev
python -m pytest -q
```

The repository's [CI workflow](../.github/workflows/ci.yml) runs the CPU test
suite on Linux and Windows with Python 3.11. A Linux job also checks local
documentation links and distribution contents, builds the wheel and source
archive, and checks the installed wheel outside the checkout. These jobs use
the CPU profile; GPU execution requires a separate CUDA-enabled environment.

## Optional dependencies

| Extra | Use |
|---|---|
| `single-cell` | Read `.h5ad` query files; included in both profiles |
| `parquet` | Read/write Parquet; included in both profiles |
| `training` | Selected scratch-training stages and public data preparation |
| `tahoe-preprocessing` | Replay the saved Tahoe projection and assemble X128 |
| `plotting` | Render computational manuscript figures; Arial is supplied separately |
| `dev` | Run the lightweight test suite |

Add extras with `uv sync`, retaining the same profile flags. For example, add
training dependencies to the CPU profile:

```bash
uv sync --locked --extra training
```

For the CUDA profile, keep `--no-default-groups --group cu126`:

```bash
uv sync --locked --no-default-groups --group cu126 --extra training
```

Replace `training` with `tahoe-preprocessing`, `plotting` or `dev` as needed.
Repeat `--extra` to retain several extras, such as
`--extra training --extra plotting`. Each sync makes `.venv` match the requested
profile and extras, so include every extra you still need. A command without
the CUDA profile flags selects the default CPU profile.

For the Bash-based archived evaluation and training workspace scripts, use Linux
or WSL. The prepared-query prediction commands use the Python CLI.

Continue with [pretrained prediction](../README.md#prediction-workflows) to run
the small examples included in the repository. These examples need only the
model archive; [models and data setup](data.md) explains the download and
artifact path. Each prediction guide explains how to substitute your own data.
[Common problems](troubleshooting.md) and
[reproduction requirements](reproduction.md) cover preparation and environment issues.
