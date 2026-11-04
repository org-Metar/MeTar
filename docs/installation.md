# Installation

Use Python 3.11 for the checked CPU workflow. Package metadata accepts Python
3.10 or newer; the retained third-party comparators have separate environment
requirements. The prediction package and large model/data downloads are installed
separately.

## 1. Get the code

Install [Git](https://git-scm.com/downloads) and
[uv](https://docs.astral.sh/uv/getting-started/installation/), then open a terminal:

```bash
git clone https://github.com/org-Metar/MeTar.git
cd MeTar
uv venv --python 3.11
```

Run the following commands from this `MeTar` directory. You do not need a GitHub
account to clone the public repository or download its release assets.

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
`.\.venv\Scripts\metar.exe` instead of `metar`. To target this environment
explicitly during installation, add `--python .venv` to each `uv pip install`
command below.

Activate this environment again whenever you open a new terminal. Check that
`python` points to it:

```bash
python -c "import sys; print(sys.executable)"
```

## 3. Install PyTorch and MeTar

### CPU

Install the CPU PyTorch wheel, then the package with H5AD and Parquet support:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[single-cell,parquet]"
```

### NVIDIA GPU

Use this route instead of the CPU wheel when GPU execution is needed. Select a
compatible wheel for your platform and driver using the official
[PyTorch installation selector](https://pytorch.org/get-started/locally/).
The CUDA 12.8 commands below are an example, not a separately validated project
runtime:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ".[single-cell,parquet]"
```

## 4. Verify the installation

```bash
python -c "import sys, torch; print(sys.version); print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
metar describe
metar predict --help
```

`metar describe` lists the six prediction workflows and available training
stages. It does not require downloaded weights. Prediction defaults to CPU;
add `--device cuda:0` for the first GPU. The prediction API falls back to CPU
when CUDA is unavailable, so check the printed availability if GPU execution
is required.

The optional tests can be run after installing the development extra:

```bash
uv pip install -e ".[dev]"
python -m pytest -q
```

## Optional dependencies

| Extra | Use |
|---|---|
| `single-cell` | Read `.h5ad` query files |
| `parquet` | Read/write Parquet |
| `training` | Selected scratch-training stages and public data preparation |
| `tahoe-preprocessing` | Replay the saved Tahoe projection and assemble X128 |
| `plotting` | Render computational manuscript figures; Arial is supplied separately |
| `dev` | Run the lightweight test suite |

Add an extra to the active environment without recreating it:

```bash
uv pip install -e ".[tahoe-preprocessing]"
uv pip install -e ".[training]"
uv pip install -e ".[plotting]"
```

For the Bash-based archived evaluation and training workspace scripts, use Linux
or WSL. The prepared-query prediction commands use the Python CLI.

Continue with [pretrained prediction](../README.md#prediction-workflows) to run
the small examples included in the repository. These examples need only the
model archive; [models and data setup](data.md) explains the download and
artifact path. Each prediction guide explains how to substitute your own data.
[Common problems](troubleshooting.md) and
[reproduction limits](reproduction.md) cover preparation and environment issues.
