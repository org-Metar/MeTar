# Third-party notices

## PertKGE

`comparators/pertkge/code/` is a source-only snapshot of the PertKGE
implementation associated with Ni et al., *Cell Genomics* (2024).

- Source: `https://github.com/myzhengSIMM/PertKGE`
- Captured revision: `f69805eb463d025a305efae10cef7a8cb6a4586b`
- License: MIT License, copyright 2024 AIDD

The license file and a source summary are retained beside the required model
and utility modules. MeTar modifies `src/utils.py` so negative sampling remains
valid when fewer than 3,000 candidates are available. Unused upstream demos,
training entry points, models and datasets are not included.

## SSGCN

`comparators/ssgcn/code/` is a source-only PyTorch snapshot linked by its README
to the following upstream repositories:

- PyTorch source: `https://github.com/myzhengSIMM/SSGCN`
- Captured revision: `a76bddcf54aa0fdd334a2e4eb33d93ddf76c5a8a`
- Original implementation: `https://github.com/boyuezhong/SSGCN`
- License: Apache License 2.0, as declared in both upstream README files

Neither upstream repository contains a standalone license file. A verbatim copy
of the Apache License 2.0 is distributed beside this snapshot as `LICENSE`.
Only the model and utility modules needed by the selected comparison are
retained. Unused training demos, environment snapshots, datasets and saved
models are not included.
