# Computational manuscript figures

Use the plotting runner to list the available figures and their required data:

```bash
python scripts/plot_figures.py list
python scripts/plot_figures.py list --json
python scripts/plot_figures.py check fig2g
python scripts/plot_figures.py check fig3f
```

The package includes computational plots and diagram assembly. Laboratory
figures supplied by the experimental collaborators are not reproduced here.

## Render a figure

Install plotting dependencies and make Arial available:

```bash
uv pip install -e ".[plotting]"
python scripts/plot_figures.py render fig2g --output outputs/fig2g
python scripts/plot_figures.py render fig3f --output outputs/fig3f
python scripts/plot_figures.py render fig7 --output outputs/fig7
```

If Arial is in a separate directory, add `--font-dir /path/to/arial-fonts`.
Arial is required for the configured typography. Use a new output directory
for each render; source and input paths are reserved for their original files.

`list` and `check` use only the Python standard library. `check` verifies the
selected source/table files, Python syntax and required external assets.
Rendering uses the supplied numerical values and drawing functions. Model
training and permutation/bootstrap analyses belong to the upstream result
workflows.

## Data and external assets

Numerical source tables are in `data/plot_sources/`; selected figure code is in
`figures/manuscript/`. The runner supplies paths and output bindings. Use it
instead of executing internal composition modules directly.

Some main-figure assemblies need native diagram exports, editable PPTX or a
vector baseline. Download the separate asset archive and pass its root:

```bash
python scripts/download_data.py --archive figure_assets.tar.gz --download --package ../metar_artifacts
python scripts/plot_figures.py check all --asset-root ../metar_artifacts/data/plot_assets
python scripts/plot_figures.py render fig3 --asset-root ../metar_artifacts/data/plot_assets --output outputs/fig3
```

Exact required filenames and integrity hashes are in
[`figures/plot_catalog.json`](../figures/plot_catalog.json). Missing or mismatched
assets fail the check. The archive contains the computational composition
inputs listed in the catalog, including prepared tables and native diagrams.

## Useful entry points

| Route | Content |
|---|---|
| `fig2g` | Target-family/MOA association table |
| `fig3f` | Cell-line adapted recovery summary |
| `fig7` | Multi-cell consensus analyses from retained tables |
| `fig2` through `fig6` | Main-figure computational plots and native-diagram assembly, as listed by the runner |
| Supported `supp` routes | Supplementary computational plots; use `list` for current identifiers |

For scientific calculations and model-to-score reproduction, follow
[reproduction.md](reproduction.md). Figure rendering uses the supplied source
results; biological interpretation follows the underlying experiment.
