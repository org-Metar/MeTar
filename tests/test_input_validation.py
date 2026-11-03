from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metar.consensus import consensus_from_long
from metar.io import load_feature_matrix
from metar.result import PredictionResult


@pytest.mark.parametrize("suffix", ["npy", "npz"])
def test_one_array_vector_is_one_query(tmp_path, suffix) -> None:
    path = tmp_path / f"query.{suffix}"
    vector = np.asarray([1.0, 2.0, 3.0])
    if suffix == "npy":
        np.save(path, vector)
    else:
        np.savez(path, x=vector)
    matrix, ids = load_feature_matrix(path, expected_dim=3)
    np.testing.assert_array_equal(matrix, vector[None, :])
    assert ids == ("query_0",)


def test_nonfinite_scores_cannot_silently_become_consensus_zeros() -> None:
    frame = pd.DataFrame(
        {
            "query_id": ["q"] * 4,
            "cell_line": ["a", "a", "b", "b"],
            "target": ["g1", "g2"] * 2,
            "score": [np.nan, 1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(ValueError, match="must be present"):
        consensus_from_long(frame)
    frame.loc[0, "score"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        consensus_from_long(frame)


def test_ranking_rejects_nonfinite_scores() -> None:
    with pytest.raises(ValueError, match="finite"):
        PredictionResult(np.asarray([[np.nan]]), ("g",), ("q",), "test")
