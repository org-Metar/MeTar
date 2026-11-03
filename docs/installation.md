# Installation

Use Python 3.11 for the checked route. Package metadata accepts Python 3.10 or
newer; the captured third-party comparators have separate environment needs.
All commands below run from the code repository root.

## Get the code

With access to the repository:

```bash
git clone https://github.com/org-Metar/MeTar.git
cd MeTar
```

## CPU

Install [uv](https://docs.astral.sh/uv/), then create an environment and install
the CPU PyTorch wheel before the package:

```bash
uv venv --python 3.11
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[dev,single-cell,parquet]"
```

## NVIDIA GPU

Choose a compatible CUDA wheel for your platform and driver using the
[PyTorch installation selector](https://pytorch.org/get-started/locally/).
The following is a CUDA 12.8 example, not a separately validated project runtime:

```bash
uv venv --python 3.11
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ".[dev,single-cell,parquet]"
```

## Activate and verify

```bash
# Linux/macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
python -c "import sys, torch; print(sys.version); print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
metar describe
metar predict --help
python -m pytest -q
```

Keep this environment active for the remaining `python` and `metar` commands.
Prediction defaults to CPU. Add `--device cuda:0` to request the first GPU;
the current prediction API falls back to CPU when CUDA is unavailable. Check
the printed device availability when GPU execution is required.

## Optional dependencies

| Extra | Use |
|---|---|
| `single-cell` | Read `.h5ad` query files |
| `parquet` | Read/write Parquet |
| `training` | Selected scratch-training stages and public data preparation |
| `tahoe-preprocessing` | Replay the released Tahoe projection and assemble X128 |
| `plotting` | Render manuscript figures; Arial is supplied separately |
| `dev` | Run the lightweight test suite |

Add an extra without recreating the environment, for example:

```bash
uv pip install -e ".[training]"
uv pip install -e ".[plotting]"
```

The code checkout contains no large pretrained weights. Continue with
[data setup](data.md), then the [prediction guide](usage.md). The current runtime and historical
comparison limits are documented in [reproduction](reproduction.md).
