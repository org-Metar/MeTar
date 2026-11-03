from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from metar.consensus import consensus_from_long
from metar.models import (
    ClosedDiagReadout,
    FullSpaceScorer,
    ProteomicsEncoder,
    ProteomicsLabelAdapter,
    QueryGatedSupportAdapter,
    Round259QueryEncoder,
    SingleCellModel,
)
from metar.predict import (
    predict_bulk_668,
    predict_cross_cell,
    predict_full_space,
    predict_proteomics,
    predict_single_cell,
)
from metar.preprocessing import ProteomicsPreprocessor


def save_closed_checkpoint(path: Path) -> tuple[ClosedDiagReadout, tuple[str, ...]]:
    torch.manual_seed(3)
    labels = ("A", "B", "C", "D", "E")
    config = {"hidden": 6, "dropout": 0.0, "extra_layers": 0}
    model = ClosedDiagReadout(torch.randn(len(labels), 4), config, input_dim=3)
    model.eval()
    torch.save(
        {
            "labels": labels,
            "readout_cfg": config,
            "readout_state_dict": model.state_dict(),
        },
        path,
    )
    return model, labels


def test_bulk_668_loads_exact_checkpoint_schema(tmp_path: Path) -> None:
    checkpoint = tmp_path / "closed.pth"
    model, labels = save_closed_checkpoint(checkpoint)
    features = np.asarray([[0.1, -0.2, 0.3], [0.5, 0.2, -0.1]], dtype=np.float32)

    result = predict_bulk_668(features, ("q1", "q2"), checkpoint)

    with torch.no_grad():
        expected = model(torch.tensor(features)).numpy()
    np.testing.assert_allclose(result.scores, expected, rtol=0.0, atol=1.0e-7)
    assert result.targets == labels
    assert list(result.to_long(top_k=2).groupby("query_id").size()) == [2, 2]


def test_cross_cell_uses_selected_adapter_schema(tmp_path: Path) -> None:
    base_path = tmp_path / "closed.pth"
    base, labels = save_closed_checkpoint(base_path)
    config = {
        "hidden_dim": 5,
        "dropout": 0.0,
        "max_delta_norm": 2.5,
        "shift_scale": 0.18,
    }
    adapter = QueryGatedSupportAdapter(base.target_anchor, 5, 0.0, 2.5)
    adapter_path = tmp_path / "adapter.pth"
    torch.save(
        {"cfg": config, "adapter_state_dict": adapter.state_dict()}, adapter_path
    )
    query = np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32)
    support = np.asarray([[0.3, 0.1, -0.2], [0.0, 0.5, 0.2]], dtype=np.float32)
    support_y = np.zeros((2, len(labels)), dtype=np.float32)
    support_y[0, 0] = 1.0
    support_y[1, 2] = 1.0

    result = predict_cross_cell(
        query,
        ("drug",),
        support,
        support_y,
        base_path,
        adapter_path,
        support_ids=("s1", "s2"),
    )

    assert result.scores.shape == (1, len(labels))
    assert result.metadata.loc[0, "shift_scale"] == 0.18

    with np.testing.assert_raises_regex(ValueError, "must be disjoint"):
        predict_cross_cell(
            query,
            ("s1",),
            support,
            support_y,
            base_path,
            adapter_path,
            support_ids=("s1", "s2"),
        )


def test_full_space_uses_round259_query_state_and_exported_targets(
    tmp_path: Path,
) -> None:
    torch.manual_seed(4)
    encoder = Round259QueryEncoder(3, 5, 4)
    state = {f"a01_encoder.{key}": value for key, value in encoder.state_dict().items()}
    state["h07_model.logit_scale_unseen"] = torch.tensor(0.7)
    checkpoint = tmp_path / "full.pth"
    torch.save(
        {"model_state_dict": state, "cfg": {"query_adapter_rank": 0}}, checkpoint
    )
    target_values = np.random.RandomState(1).normal(size=(7, 4)).astype(np.float32)
    target_names = tuple(f"g{index}" for index in range(7))
    features = np.asarray([[0.1, 0.2, -0.3]], dtype=np.float32)

    result = predict_full_space(
        features,
        ("q",),
        checkpoint,
        target_names,
        target_values,
    )
    loaded, _ = FullSpaceScorer.from_checkpoint(checkpoint, target_values)

    with torch.no_grad():
        expected = loaded(torch.tensor(features)).numpy()
    np.testing.assert_allclose(result.scores, expected, rtol=0.0, atol=1.0e-7)


def test_single_cell_groups_cells_and_scores_full_space(tmp_path: Path) -> None:
    torch.manual_seed(5)
    model = SingleCellModel(3, 6, 4, 2)
    checkpoint = tmp_path / "single.pth"
    torch.save(model.state_dict(), checkpoint)
    features = np.random.RandomState(2).normal(size=(6, 3)).astype(np.float32)
    groups = ("d1", "d1", "d1", "d2", "d2", "d2")
    targets = np.random.RandomState(3).normal(size=(8, 4)).astype(np.float32)

    result = predict_single_cell(
        features,
        groups,
        checkpoint,
        tuple(f"g{index}" for index in range(8)),
        targets,
        cell_line="A549",
    )

    assert result.scores.shape == (2, 8)
    assert result.query_ids == ("d1", "d2")
    assert result.metadata["cells_used"].tolist() == [3, 3]


def test_proteomics_applies_locked_preprocessor(tmp_path: Path) -> None:
    torch.manual_seed(6)
    encoder = ProteomicsEncoder(2, 4, 3, 1, 0.0)
    adapter = ProteomicsLabelAdapter(3, "linear")
    target_values = np.random.RandomState(4).normal(size=(5, 3)).astype(np.float32)
    checkpoint = tmp_path / "proteomics.pth"
    torch.save(
        {
            "model_state_dict": encoder.state_dict(),
            "label_adapter_state_dict": adapter.state_dict(),
        },
        checkpoint,
    )
    preprocessor = ProteomicsPreprocessor(
        feature_names=("p1", "p2"),
        mean=(1.0, 2.0),
        std=(2.0, 4.0),
        log1p=False,
    )
    frame = pd.DataFrame({"query_id": ["d"], "p1": [3.0], "p2": [6.0]})

    result = predict_proteomics(
        frame,
        ("d",),
        checkpoint,
        tuple(f"g{index}" for index in range(5)),
        target_values,
        preprocessor,
    )

    assert result.scores.shape == (1, 5)
    assert np.isfinite(result.scores).all()


def test_consensus_matches_candidate_wise_zscore_mean() -> None:
    frame = pd.DataFrame(
        {
            "query_id": ["d1"] * 6,
            "cell_line": ["A", "A", "A", "B", "B", "B"],
            "target": ["g1", "g2", "g3"] * 2,
            "score": [1.0, 2.0, 3.0, 10.0, 8.0, 6.0],
        }
    )

    result = consensus_from_long(frame)

    assert result.query_ids == ("d1",)
    assert result.targets == ("g1", "g2", "g3")
    assert result.metadata.loc[0, "n_cell_lines"] == 2
    assert np.isfinite(result.scores).all()


def test_consensus_rejects_topk_truncated_model_output() -> None:
    frame = pd.DataFrame(
        {
            "query_id": ["d1"] * 4,
            "cell_line": ["A", "A", "B", "B"],
            "target": ["g1", "g2"] * 2,
            "score": [1.0, 2.0, 3.0, 4.0],
            "candidate_count": [3] * 4,
        }
    )

    with np.testing.assert_raises_regex(ValueError, "complete score vectors"):
        consensus_from_long(frame)
