# Train a response encoder for another assay

Modality training reuses the learned bulk target space while training an
assay-specific response encoder. Choose the corresponding guide:

| Setting | Trained | Detailed guide |
|---|---|---|
| Single-cell | Response encoder and score scale; target embeddings remain fixed | [Single-cell training](single-cell.md) |
| Protein abundance | Response encoder, score scale and configured label adapter; stored target embeddings remain fixed | [Proteomics training](proteomics.md) |

For workspace setup and shared options, see [training setup](../training.md).
To use existing checkpoints, see [prediction](../usage.md).
