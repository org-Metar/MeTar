# PertKGE comparator

Adapted from the PertKGE authors' implementation of *Identifying compound-protein interactions with knowledge graph embedding of perturbation transcriptomics*. The original MIT license is retained in `LICENSE`.

`src/model.py` and `src/utils.py` supply the graph embedding model and data/metric helpers used by the manuscript comparison. The manuscript evaluators implement their matched training/scoring protocol using these modules.

Use the repository's comparator environment and scientific reproduction commands in [Reproduction](../../../docs/reproduction.md). Large comparator inputs and checkpoints are downloaded with the published artifacts.
