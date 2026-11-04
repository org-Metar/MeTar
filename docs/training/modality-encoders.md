# Train a response encoder for another assay

Modality training reuses the learned bulk target space without repeating bulk
response-model or PPI graph training. Each setting now has its own guide:

| Setting | Trained | Detailed guide |
|---|---|---|
| Single-cell | Response encoder and score scale; target embeddings remain fixed | [Single-cell training](single-cell.md) |
| Protein abundance | Response encoder, score scale and configured label adapter; stored target embeddings remain fixed | [Proteomics training](proteomics.md) |

For private workspace setup and shared options, see [training setup](../training.md).
To use existing checkpoints, see [prediction](../usage.md).

<a id="what-is-trained"></a>
<a id="single-cell-prepare-and-train"></a>
<a id="single-cell-data-format"></a>
<a id="use-your-own-single-cell-training-data"></a>
<a id="single-cell-outputs"></a>

Single-cell commands, cell/label formats, custom inputs and outputs are in the
[single-cell guide](single-cell.md).

<a id="proteomics-prepare-and-train"></a>
<a id="protein-abundance-data-format-and-preprocessing"></a>
<a id="proteomics-outputs"></a>

Proteomics commands, abundance preparation, custom inputs and outputs are in
the [proteomics guide](proteomics.md).
