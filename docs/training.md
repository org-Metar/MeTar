# Training from scratch

`metar train` orchestrates the selected scientific training chain in a separate
workspace. It preserves the selected split, seed and hyperparameter contracts;
it is not a generic search or arbitrary-dataset training framework.
Install the [training dependencies](installation.md), download the release
assets and import the required [prepared public inputs](data.md) first.

## Inputs and stage order

```text
gene-prior → closed-bulk → support-initializer → cross-cell-adapter → full-space
                                                                       ├→ single-cell (three cell lines)
                                                                       └→ proteomics
```

The stage names above are the values of `--stage`. `--stage all` runs the chain
including A549, SW480 and NCI-H460 single-cell stages.

In addition to the public bulk/Tahoe/ProteomicsDB inputs, scratch training needs
the following files from `training_inputs.tar.gz`:

| Role | Canonical path under artifact root |
|---|---|
| Gene-relation graph | `data/public_raw/gene_prior/process.txt` |
| Original PC3 978-feature response | `data/processed/closed/r210/pc3_landmark_response_978.tsv.gz` |
| Original 668-target score table/header | `data/processed/closed/r210/pc3_metar_drugcell_scores_668.tsv.gz` |

Hashes are in [`training_input_contract.json`](../manifests/training_input_contract.json).
The model archives alone are insufficient. The graph used for MeTar's target
prior contains gene-relation priors only, excluding drug nodes, DTI,
expression-effect and `HAS_BINDING_TO` edges. Comparator checkpoints do not
initialize this prior.

## Prepare an isolated workspace

The code root, artifact root and training output must be separate directories.
The destination must not exist. Preparation copies executable sources and
metadata and links read-only inputs. It copies no selected pretrained checkpoint
or trained target-vector values; only gene names are read from the aligned
reference table to fix proteomics order.

```bash
uv pip install -e ".[training]"
python scripts/prepare_training.py --artifacts ../metar_artifacts --output ../metar-training --check-only
python scripts/prepare_training.py --artifacts ../metar_artifacts --output ../metar-training
```

`--check-only` lists missing prerequisites without writing a workspace, loading
a model or training. It checks file presence rather than dataset semantics.
Use `--copy-inputs` if symlinks are unavailable, for example on Windows without
symlink privileges; this copies the large inputs explicitly.

If no prebuilt bulk cache was present during preparation, build it in the new
workspace:

```bash
python scripts/prepare_bulk_data.py --package ../metar-training --device cpu
```

## Preview, check and train

```bash
metar train --stage all --workspace ../metar-training --device cuda:0 --dry-run
metar train --stage gene-prior --workspace ../metar-training --device cuda:0 --check-only
metar train --stage all --workspace ../metar-training --device cuda:0
```

`--dry-run` prints commands only; it does not validate their inputs. Stage
`--check-only` inspects dependencies and workspace isolation. Downstream stages
depend on checkpoints created by preceding stages, so their checks are expected
to fail in a workspace where upstream training has not completed.

Run stages individually when appropriate:

```bash
metar train --stage closed-bulk --workspace ../metar-training --device cuda:0
metar train --stage single-cell --cell-line A549 --workspace ../metar-training --device cuda:0
metar train --stage proteomics --workspace ../metar-training --device cuda:0
```

Finish each stage's upstream dependencies first. The full-space branch also
requires the cross-cell protocol audit generated earlier in the chain. Freshly
trained full-space vectors feed the single-cell and proteomics branches.

## Outputs and limits

The earlier stages use fixed private paths inside the generated compatibility
workspace. Single-cell and proteomics accept `--output-dir`; under `--stage all`
their outputs use separate assay/cell-line subdirectories. The CLI refuses a
normal pretrained compatibility tree and refuses to overwrite existing selected
training outputs or escape through symlinks/junctions.

Interface checkpoint copies are written to the workspace's
`checkpoints/selected/`. The new run's logs record the actual selected epoch and
metrics. Canonical filenames retain their release identity, so an epoch number
in a compatibility filename does not establish a newly trained result at that
epoch. The original source and pretrained roots are not training output paths.

The present checks cover preparation, isolation and chaining behavior; the
complete training chain and resulting scientific quality were not verified in
the new CPU runtime. See [reproduction](reproduction.md) for the tested scope
and [the input-preparation guide](usage.md) for step-by-step input preparation.
