# Reproduce model results

For new-query prediction, follow [usage.md](usage.md). This guide covers
comparison against archived manuscript scores. For scratch training, follow
[training.md](training.md).

## Selected models

[`model_selection.yaml`](../manifests/model_selection.yaml) identifies the model
configurations used for the computational results. Figure composition and
checkpoint selection are separate: updating an illustration does not select a
new checkpoint.

| Component | Selected configuration |
|---|---|
| Gene-relation prior | Dimension 256, fanout 16, epoch 30 |
| Joint 668-target bulk model | Epoch 6 |
| Support-conditioned adapter | Seed 20260524, shift scale 0.18, epoch 10 |
| Full-space encoder | `e13_bce001` |
| Single-cell models | A549 seed 2043, SW480 seed 2041, NCI-H460 seed 2042 |
| Proteomics model | Seed 2026, epoch 70 |
| Consensus | Target-wise cell z-score mean or raw-score mean at prediction time |
| PDF-3 prediction input | Aligned log2FC query into the 668-target model |

The target prior uses gene relations only. Drug nodes, DTI edges,
expression-effect edges and `HAS_BINDING_TO` edges are excluded. Retained
comparator code is for comparator reproduction; its pretrained weights are not
MeTar dependencies.

## Compare pretrained outputs

Download the required [release assets](data.md), place the exact prepared public
inputs and build the bulk cache. Generate the execution workspace with Bash
(for example on Linux or WSL):

```bash
export METAR_ARTIFACT_ROOT=/absolute/path/to/metar_artifacts
bash tools/prepare_prediction_runtime.sh "$METAR_ARTIFACT_ROOT"
python scripts/reproduce_results.py --package "$METAR_ARTIFACT_ROOT" --output ../metar-verification --device cuda:0
```

The generated `compat/` tree binds the artifact root and stays outside Git.
It links inputs and supplies the selected executable overlays without modifying
the source data. Normal prepared-query prediction does not require this tree.

The verifier compares these branches against archived outputs:

| Branch | Archived comparison scope |
|---|---|
| Support-adapted cross-cell | 14 evaluation cell lines; 1,452 drug-cell units |
| Full-space | 4,463 held-out profiles; top-5% outcomes and aggregate success rate `0.3269101501232355` |
| Single-cell | A549, SW480 and NCI-H460; 30 drugs and 300 cells per drug in fixed pools; 90 x 16,583 scores |
| Proteomics | 15 final-test metrics and the training-only preprocessing state |

Report the actual branch results produced by the command. An archived reference
value is not a result from a newly run environment.

## Numerical and preparation limits

- The checked Python 3.11/PyTorch CPU implementation agrees with the corresponding
  original model classes in that environment. Single-cell and proteomics did
  not meet the historical `1e-6` score-comparison tolerance there; matching
  checked top-target sets does not establish historical numerical identity.
  Full training and CUDA reproduction require their own runs.
- Public single-cell queries use deterministic query-dependent sampling. The
  archived comparison uses fixed evaluation cell pools. Saved Tahoe projection
  components follow the notebook reload path and need not be bitwise identical
  to coordinates produced by the initial live fit.
- Near-tied full-space float32 scores can shift an internal rank by one across
  executions. The comparison checks the reported top-5% outcomes and aggregate
  rate, while recording rank-tie differences separately.
- Ordinary 668-target bulk queries require training-derived normalization. A
  separately saved fitted mean/std state is not supplied with the pretrained
  release. The PDF-3 log2FC input is a separate application contract, not a
  replacement transform for arbitrary assays.
- The public-input contract identifies processed files, but does not provide a
  complete upstream download/transformation recipe for every file. The importer
  places existing inputs; it cannot reconstruct missing preprocessing from a
  filename or matrix shape.
- The adapted Figure 7 workflow retains checkpoints and score outputs, but not
  every intermediate query/support tensor needed for direct checkpoint-to-score
  re-export. Downstream aggregation and analysis can use the retained scores;
  re-exporting those scores requires the earlier preparation.

Computational plotting is described in [plotting.md](plotting.md). Laboratory
figure reproduction is not part of this code package.
