# Small prediction examples

These are real, small computational examples from the public datasets used by
MeTar. They are ready for the prediction interfaces; they are not random format
fixtures. Download the pretrained models as described in the main
[README](../README.md) and run commands from the repository root. The complete
example inputs occupy less than 1 MB.

| Example | Input | What it contains |
|---|---|---|
| Bulk, 668 targets | `inputs/bulk_query.npz` | Two prepared HELA responses, 978 ordered features each |
| New cell line | `inputs/cross_cell/hela/` | Two query responses and 64 separately annotated support responses |
| Second cell line | `inputs/cross_cell/hcc515/` | The same two query compounds in HCC515, with its own 64 support responses |
| Genome-wide ranking | `inputs/fullspace_query.npz` | Four prepared responses from HCC515 and HELA |
| Single-cell ranking | `inputs/single_cell_a549.npz` | A549 X128 coordinates: two compounds, 300 cells per compound |
| Protein abundance | `inputs/proteomics.tsv` | Two compounds and the 2,000 abundance features selected by the saved preprocessor |
| Multi-cell consensus | `inputs/consensus/` | Complete direct-prediction score vectors for the two shared query compounds in HELA and HCC515 |

## Provenance and preparation

### Bulk and cross-cell responses

The bulk and cross-cell inputs are derived from the LINCS/CMap response table
used for the reported cross-cell evaluation. HELA and HCC515 are held-out cell
lines in that evaluation. Each query file contains `BRD-A05186015` and
`BRD-A15297126`; support compounds and query compounds are disjoint. Responses
were aggregated by compound and cell line, then standardized using the mean and
standard deviation of the model's meta-training response table. The values are
already prepared: do not standardize these example files again.

`bulk_query.npz` is the same HELA input as `cross_cell/hela/query.npz`. Its bulk
command demonstrates the 668-target interface; it is not an example of a
within-training-cell evaluation. For your own data, the [bulk
guide](../docs/prediction/bulk.md) explains the required feature order and scale.

Both `query.npz` and `support.npz` have `x` (float32, rows × 978) and `query_ids`
(one unique compound identifier per row). `support_labels.tsv` has columns
`query_id` and `targets`, with multiple gene symbols separated by `|`.
`support_weights.npy` contains one positive weight per support row, in exactly
the same order. Query labels are not included or passed to prediction.

### Genome-wide responses

The four rows come from the prepared h07 target-holdout evaluation inputs. They
retain that workflow's feature order and preparation, independently of the
closed 668-target standardization above. The four identifiers are
`holdout_test:131`, `holdout_test:132` (HCC515), `holdout_test:1532` and
`holdout_test:1533` (HELA). The NPZ contains `x` (float32, 4 × 978) and
`query_ids`. Use the full-space checkpoint and its matching target embeddings.
No evaluation target annotations are supplied to the predictor.

### Single-cell responses

The A549 example is a subset of the processed Tahoe X128 data used for the
reported single-cell analysis. It contains `5-fluorouracil` and `9-ing-41`, with
300 cells sampled per compound. The NPZ contains `x` (float32, 600 × 128) and
`query_ids`; here the compound identifier repeats for all cells belonging to
the same query. These are projection coordinates, not 128 gene counts. Do not
project them again. Use `--max-cells 0` to use all supplied cells. For raw counts,
follow the saved-projection preparation in the [single-cell
guide](../docs/prediction/single-cell.md).

### Protein abundance

The abundance rows for `omipalisib` and `rgb-286638` come from the held-out
ProteomicsDB-derived evaluation table. Only `query_id` and the 2,000 feature
columns required by the released preprocessor are retained. These values are
the input abundance measurements, before standardization. Pass the table
directly to `metar predict proteomics`; the saved preprocessing state performs
the transformation. Its selected state uses `log1p=false`; do not add a log
transform. Target labels are not included.

### Consensus scores

`hela_scores.tsv` and `hcc515_scores.tsv` contain previously computed direct
predictions for the two query compounds supplied above. Each file contains all
668 candidate scores for each compound, with columns `query_id`, `cell_line`,
`target`, `score` and `candidate_count`. These are genuine model outputs, not
experimental measurements or an independent accuracy evaluation. They make the
consensus command runnable without first performing two predictions. You can
also generate fresh complete scores from the two cell-line inputs; do not use
Top-k-truncated score tables for consensus.

The dataset placement and preparation requirements are listed in the
[data guide](../docs/data.md). These examples contain no PDF-3 or wet-lab
material. Their purpose is to show the input and output workflow; the paper's
evaluation protocols and numerical reproduction are documented separately in
the [reproduction guide](../docs/reproduction.md).
