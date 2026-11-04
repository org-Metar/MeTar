"""Prediction execution checks; synthetic tensors do not evaluate model quality."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

import metar.predict as prediction
from metar.consensus import consensus_from_long
from metar.io import load_feature_matrix, load_grouped_cells, load_target_table
from metar.models import (
    ClosedDiagReadout,
    FullSpaceScorer,
    ProteomicsEncoder,
    ProteomicsLabelAdapter,
    ProteomicsScorer,
    QueryGatedSupportAdapter,
    Round259QueryEncoder,
    SingleCellModel,
    load_checkpoint,
)
from metar.preprocessing import ProteomicsPreprocessor
from metar.result import PredictionResult


@pytest.fixture
def closed(tmp_path: Path):
    torch.manual_seed(19)
    labels = ("A", "B", "C", "D", "E")
    cfg = {"hidden": 6, "dropout": 0.0}
    model = ClosedDiagReadout(torch.randn(5, 4), cfg, input_dim=3).eval()
    path = tmp_path / "closed.pth"
    torch.save(
        {
            "labels": labels,
            "readout_cfg": cfg,
            "readout_state_dict": model.state_dict(),
        },
        path,
    )
    return path, model, labels


@pytest.fixture
def adapted(tmp_path: Path, closed):
    base_path, base, labels = closed
    cfg = {"hidden_dim": 5, "dropout": 0.0, "max_delta_norm": 2.5, "shift_scale": 0.18}
    adapter = QueryGatedSupportAdapter(base.target_anchor, 5, 0.0, 2.5).eval()
    adapter_path = tmp_path / "adapter.pth"
    torch.save({"cfg": cfg, "adapter_state_dict": adapter.state_dict()}, adapter_path)
    rng = np.random.RandomState(21)
    query = rng.normal(size=(5, 3)).astype(np.float32)
    support = rng.normal(size=(4, 3)).astype(np.float32)
    support_y = np.eye(len(labels), dtype=np.float32)[:4]
    weights = np.asarray([1.0, 2.0, 0.5, 3.0], dtype=np.float32)
    return base_path, base, adapter_path, adapter, query, support, support_y, weights


def test_bulk_batching_matches_model_and_reports_completion(closed, monkeypatch):
    checkpoint, model, labels = closed
    values = np.random.RandomState(7).normal(size=(5, 3)).astype(np.float32)
    events = []
    monkeypatch.setattr(
        prediction, "report_progress", lambda *a, **kw: events.append((a, kw))
    )
    result = prediction.predict_bulk_668(
        values, tuple(f"q{i}" for i in range(5)), checkpoint, batch_size=2, device="cpu"
    )
    with torch.no_grad():
        expected = model(torch.from_numpy(values)).numpy()
    np.testing.assert_allclose(result.scores, expected, rtol=1e-6, atol=1e-6)
    assert result.targets == labels
    assert [item[0] for item in events] == [(1, 3), (2, 3), (3, 3)]
    assert events[-1][1]["metrics"] == {"queries": 5}


def test_prediction_progress_keeps_stdout_clean(closed, capsys):
    from metar.runtime import RunReporter

    checkpoint, _, _ = closed
    with RunReporter("Prediction") as reporter:
        result = prediction.predict_bulk_668(
            np.ones((2, 3)), ("q1", "q2"), checkpoint, device="cpu", batch_size=1
        )
        reporter.finish(queries=len(result.query_ids), targets=len(result.targets))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Load bulk checkpoint" in captured.err
    assert "2/2 (100%)" in captured.err
    assert "[Summary] queries=2 | targets=5" in captured.err


def test_adapted_batching_preserves_selected_adapter_calculation(adapted):
    base_path, base, adapter_path, adapter, query, support, support_y, weights = adapted
    result = prediction.predict_cross_cell(
        query,
        tuple(f"q{i}" for i in range(5)),
        support,
        support_y,
        base_path,
        adapter_path,
        support_weights=weights,
        batch_size=2,
        device="cpu",
    )
    with torch.no_grad():
        query_q = base.encode(torch.from_numpy(query))
        support_q = base.encode(torch.from_numpy(support))
        delta, gate = adapter(
            query_q, support_q, torch.from_numpy(support_y), torch.from_numpy(weights)
        )
        expected = base.score_embeddings(
            F.normalize(query_q + 0.18 * gate[:, None] * delta, dim=1)
        ).numpy()
    np.testing.assert_allclose(result.scores, expected, rtol=1e-5, atol=1e-6)
    assert result.metadata["support_count"].tolist() == [4] * 5


@pytest.mark.parametrize(
    "invalid",
    [
        "empty",
        "zero_labels",
        "nan_features",
        "nan_labels",
        "negative_labels",
        "zero_weights",
        "id_overlap",
    ],
)
def test_adapted_rejects_unusable_support_without_dropping_rows(adapted, invalid):
    base_path, _, adapter_path, _, query, support, support_y, weights = adapted
    kwargs = {}
    if invalid == "empty":
        support, support_y, weights = support[:0], support_y[:0], weights[:0]
    elif invalid == "zero_labels":
        support_y[1] = 0
    elif invalid == "nan_features":
        support[0, 0] = np.nan
    elif invalid == "nan_labels":
        support_y[0, 0] = np.nan
    elif invalid == "negative_labels":
        support_y[0, 0] = -1
    elif invalid == "zero_weights":
        weights[0] = 0
    else:
        kwargs["support_ids"] = ("q0", "s1", "s2", "s3")
    with pytest.raises(ValueError):
        prediction.predict_cross_cell(
            query,
            tuple(f"q{i}" for i in range(5)),
            support,
            support_y,
            base_path,
            adapter_path,
            support_weights=weights,
            device="cpu",
            **kwargs,
        )


def test_full_space_scores_exported_unseen_targets(tmp_path):
    torch.manual_seed(23)
    encoder = Round259QueryEncoder(3, 5, 4, adapter_rank=2)
    state = {f"a01_encoder.{key}": value for key, value in encoder.state_dict().items()}
    state["h07_model.logit_scale_unseen"] = torch.tensor(0.7)
    checkpoint = tmp_path / "full.pth"
    torch.save({"model_state_dict": state}, checkpoint)
    targets = np.random.RandomState(2).normal(size=(7, 4)).astype(np.float32)
    query = np.random.RandomState(3).normal(size=(3, 3)).astype(np.float32)
    labels = tuple(f"unseen_{i}" for i in range(7))
    result = prediction.predict_full_space(
        query,
        ("q1", "q2", "q3"),
        checkpoint,
        labels,
        targets,
        batch_size=2,
        device="cpu",
    )
    model, _ = FullSpaceScorer.from_checkpoint(checkpoint, targets)
    with torch.no_grad():
        expected = model(torch.from_numpy(query)).numpy()
    np.testing.assert_allclose(result.scores, expected, rtol=1e-6, atol=1e-6)
    assert result.targets == labels


def test_single_cell_group_order_sampling_and_progress(tmp_path, monkeypatch):
    torch.manual_seed(29)
    model = SingleCellModel(3, 6, 4, 2).eval()
    checkpoint = tmp_path / "single.pth"
    torch.save(model.state_dict(), checkpoint)
    features = np.random.RandomState(4).normal(size=(7, 3)).astype(np.float32)
    groups = ("b", "a", "b", "a", "b", "a", "b")
    targets = np.random.RandomState(5).normal(size=(5, 4)).astype(np.float32)
    events = []
    monkeypatch.setattr(
        prediction, "report_progress", lambda *a, **kw: events.append(a)
    )
    result = prediction.predict_single_cell(
        features,
        groups,
        checkpoint,
        tuple("ABCDE"),
        targets,
        cell_line="A549",
        max_cells=2,
        device="cpu",
    )
    assert result.query_ids == ("b", "a")
    assert result.metadata["cells_used"].tolist() == [2, 2]
    assert events == [(1, 2), (2, 2)]
    expected = []
    with torch.no_grad():
        for group in result.query_ids:
            indices = np.flatnonzero(np.asarray(groups) == group)
            chosen = prediction._stable_group_subset(indices, group, 2, 20260327)
            expected.append(
                model.score(
                    torch.from_numpy(features[chosen][None]), torch.from_numpy(targets)
                ).numpy()[0]
            )
    np.testing.assert_allclose(result.scores, expected, rtol=0.0, atol=1e-7)


def test_proteomics_matches_locked_preprocessing_and_scorer(tmp_path):
    torch.manual_seed(31)
    encoder = ProteomicsEncoder(2, 4, 3, 1, 0.0)
    adapter = ProteomicsLabelAdapter(3, "linear")
    checkpoint = tmp_path / "proteomics.pth"
    torch.save(
        {
            "model_state_dict": encoder.state_dict(),
            "label_adapter_state_dict": adapter.state_dict(),
        },
        checkpoint,
    )
    targets = np.random.RandomState(6).normal(size=(5, 3)).astype(np.float32)
    preprocessor = ProteomicsPreprocessor(("p1", "p2"), (1.0, 2.0), (2.0, 4.0), False)
    frame = pd.DataFrame({"p1": [3.0, np.nan, np.inf], "p2": [6.0, 4.0, 2.0]})
    result = prediction.predict_proteomics(
        frame,
        ("a", "b", "c"),
        checkpoint,
        tuple("ABCDE"),
        targets,
        preprocessor,
        batch_size=2,
        device="cpu",
    )
    model, _ = ProteomicsScorer.from_checkpoint(checkpoint, targets)
    with torch.no_grad():
        expected = model(torch.from_numpy(preprocessor.transform(frame))).numpy()
    np.testing.assert_allclose(result.scores, expected, rtol=1e-6, atol=1e-6)
    assert np.isfinite(result.scores).all()


@pytest.mark.parametrize("batch_size", [0, -2, 1.5, True])
def test_invalid_batch_size_fails_before_checkpoint_read(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        prediction.predict_bulk_668(
            np.ones((1, 3)), ("q",), "missing.pth", batch_size=batch_size
        )


@pytest.mark.parametrize(
    "values,ids",
    [
        (np.empty((0, 3)), ()),
        (np.ones((2, 3)), ("q",)),
        (np.full((1, 3), np.nan), ("q",)),
        (np.ones((1, 3)), ("",)),
        (np.ones((2, 3)), ("q", "q")),
    ],
)
def test_invalid_queries_fail_before_checkpoint_read(values, ids):
    with pytest.raises(ValueError):
        prediction.predict_bulk_668(values, ids, "missing.pth", device="cpu")


@pytest.mark.parametrize(
    "targets,values",
    [
        (("A", "B"), np.ones((1, 4))),
        (("A", "A"), np.ones((2, 4))),
        (("A",), np.full((1, 4), np.inf)),
        ((), np.empty((0, 4))),
    ],
)
def test_invalid_targets_fail_before_checkpoint_read(targets, values):
    with pytest.raises(ValueError):
        prediction.predict_full_space(
            np.ones((1, 3)), ("q",), "missing.pth", targets, values, device="cpu"
        )


def test_checkpoint_load_is_cpu_mapped(closed, monkeypatch):
    checkpoint, _, _ = closed
    original = torch.load
    options = []

    def traced(*args, **kwargs):
        options.append((kwargs["map_location"], kwargs["weights_only"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", traced)
    assert "readout_state_dict" in load_checkpoint(checkpoint)
    assert options == [("cpu", False)]


@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "truncated_archive",
        "invalid_pickle",
        "truncated_pickle",
        "wrong_magic",
        "bad_protocol",
    ],
)
def test_invalid_checkpoint_data_has_path_and_recovery_advice(tmp_path, kind):
    checkpoint = tmp_path / f"{kind}.pth"
    if kind == "truncated_archive":
        torch.save({"weights": torch.arange(8)}, checkpoint)
        checkpoint.write_bytes(checkpoint.read_bytes()[:96])
    else:
        checkpoint.write_bytes(
            {
                "empty": b"",
                "invalid_pickle": b"not a checkpoint",
                "truncated_pickle": pickle.dumps({"weights": [1, 2, 3]})[:-2],
                "wrong_magic": pickle.dumps(123),
                "bad_protocol": b"\x80\xff",
            }[kind]
        )
    with pytest.raises(ValueError) as error:
        load_checkpoint(checkpoint)
    assert str(checkpoint) in str(error.value)
    assert "Re-download and extract the correct model assets" in str(error.value)


@pytest.mark.parametrize("payload", [None, [], 123, torch.ones(2)])
def test_nonmapping_checkpoint_has_path_and_recovery_advice(tmp_path, payload):
    checkpoint = tmp_path / "wrong_asset.pth"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="must contain a mapping") as error:
        load_checkpoint(checkpoint)
    assert str(checkpoint) in str(error.value)
    assert "Re-download" in str(error.value)


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("CUDA out of memory"),
        ValueError("unexpected tensor state"),
        TypeError("unexpected call signature"),
        ModuleNotFoundError("an unavailable optional checkpoint dependency"),
    ],
)
def test_checkpoint_loader_does_not_mask_unrelated_errors(monkeypatch, error):
    def failed(*args, **kwargs):
        raise error

    monkeypatch.setattr(torch, "load", failed)
    with pytest.raises(type(error)) as captured:
        load_checkpoint("checkpoint.pth")
    assert captured.value is error


def test_cli_invalid_checkpoint_reports_error_without_traceback(
    tmp_path, monkeypatch, capsys
):
    from metar.cli import DEFAULTS, main

    checkpoint = tmp_path / "truncated.pth"
    checkpoint.write_bytes(b"")
    queries = tmp_path / "queries.npz"
    np.savez(queries, x=np.ones((1, 978), dtype=np.float32))
    monkeypatch.setitem(DEFAULTS, "feature_order", tmp_path / "absent.txt")
    with pytest.raises(SystemExit) as error:
        main(
            [
                "predict",
                "bulk-668",
                "--input",
                str(queries),
                "--checkpoint",
                str(checkpoint),
                "--device",
                "cpu",
                "--output",
                str(tmp_path / "out.tsv"),
            ]
        )
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert str(checkpoint) in captured.err
    assert "Traceback" not in captured.err
    assert not (tmp_path / "out.tsv").exists()


def test_explicit_unavailable_cuda_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert prediction.resolve_device("auto") == torch.device("cpu")
    with pytest.raises((RuntimeError, ValueError), match="(?i)cuda"):
        prediction.resolve_device("cuda:0")


def test_input_loaders_reject_missing_ids_empty_and_malformed_arrays(tmp_path):
    path = tmp_path / "features.npz"
    for values in (np.empty((0, 3)), np.asarray(1.0)):
        np.savez(path, x=values)
        with pytest.raises(ValueError):
            load_feature_matrix(path, expected_dim=3)
    np.savez(path, other=np.ones((1, 3)))
    with pytest.raises(ValueError, match="x.*features"):
        load_feature_matrix(path, expected_dim=3)
    frame = pd.DataFrame({"query_id": [None], "f1": [1.0]})
    csv = tmp_path / "input.csv"
    frame.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="present"):
        load_feature_matrix(csv, expected_dim=1)
    with pytest.raises(ValueError, match="present"):
        load_grouped_cells(csv, expected_dim=1)


def test_target_loader_rejects_missing_target_identifier(tmp_path):
    path = tmp_path / "targets.csv"
    pd.DataFrame({"target": [None], "e1": [1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="present"):
        load_target_table(path)


def test_consensus_normalizes_numeric_labels_and_reports_skips(monkeypatch):
    from metar import consensus

    events = []
    monkeypatch.setattr(
        consensus, "report_progress", lambda *a, **kw: events.append((a, kw))
    )
    frame = pd.DataFrame(
        {
            "query_id": [1, 1, 1, 1, 2, 2],
            "cell_line": [1, 1, 2, 2, 1, 1],
            "target": [10, 20] * 3,
            "score": [1.0, 3.0, 5.0, 7.0, 2.0, 4.0],
            "candidate_count": [2] * 6,
        }
    )
    result = consensus_from_long(frame, method="raw_score_mean")
    assert result.query_ids == ("1",)
    assert result.targets == ("10", "20")
    np.testing.assert_allclose(result.scores, [[3.0, 5.0]])
    assert events[-1] == (
        (2, 2),
        {"label": "Consensus groups", "metrics": {"eligible": 1, "skipped": 1}},
    )


def test_result_accepts_array_like_scores_and_rejects_unknown_format(tmp_path):
    result = PredictionResult([[1.0, 2.0]], ("a", "b"), ("q",), "test")
    assert result.to_long()["target"].tolist() == ["b", "a"]
    with pytest.raises(ValueError, match="output_format"):
        result.write(tmp_path / "out.tsv", output_format="unknown")


def test_preprocessor_rejects_nonfinite_statistics():
    with pytest.raises(ValueError, match="means"):
        ProteomicsPreprocessor(("p",), (np.nan,), (1.0,), False)
