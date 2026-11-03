from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from metar.cli import (
    DEFAULTS,
    _load_proteomics_preprocessor,
    build_parser,
    command_describe,
    main,
)
from metar.models import ClosedDiagReadout
from metar.preprocessing import ProteomicsPreprocessor
from metar.training import archived_training_command


def test_parser_exposes_all_prediction_scenarios() -> None:
    parser = build_parser()
    root_subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    predict_parser = root_subparsers.choices["predict"]
    scenario_subparsers = next(
        action
        for action in predict_parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert set(scenario_subparsers.choices) == {
        "bulk-668",
        "full-space",
        "cross-cell",
        "single-cell",
        "proteomics",
        "consensus",
    }


def test_training_dry_command_uses_selected_full_space_variant(tmp_path) -> None:
    command = archived_training_command("full-space", tmp_path, device="cuda:0")
    assert "e13_bce001" in command
    assert "run_archived_script.py" in command[1]


def test_describe_runs(capsys) -> None:
    assert command_describe(None) == 0
    output = capsys.readouterr().out
    assert "bulk-668" in output
    assert "gene-prior" in output


def test_proteomics_defaults_use_aligned_targets_and_locked_preprocessor(
    tmp_path: Path, monkeypatch
) -> None:
    assert DEFAULTS["proteomics_target_table"].name == (
        "round259_target_embedding_aligned_to_proteomics_gene_order.csv"
    )
    preprocessor_path = tmp_path / "round261_preprocessor.json"
    expected = ProteomicsPreprocessor(
        feature_names=("p1", "p2"),
        mean=(1.0, 2.0),
        std=(0.5, 0.25),
        log1p=True,
        top_k_features=2,
    )
    expected.write(preprocessor_path)
    monkeypatch.setitem(DEFAULTS, "proteomics_preprocessor", preprocessor_path)
    args = argparse.Namespace(
        preprocessor=None,
        reference_data=None,
        write_preprocessor=None,
    )

    observed = _load_proteomics_preprocessor(args, ("g1", "g2"))

    assert observed == expected


def test_cross_cell_direct_cli_does_not_require_support_files(tmp_path: Path) -> None:
    target_count = 5
    config = {"hidden": 6, "dropout": 0.0, "extra_layers": 0}
    model = ClosedDiagReadout(torch.randn(target_count, 4), config, input_dim=978)
    checkpoint = tmp_path / "round246.pth"
    torch.save(
        {
            "labels": [f"g{index}" for index in range(target_count)],
            "readout_cfg": config,
            "readout_state_dict": model.state_dict(),
        },
        checkpoint,
    )
    query = pd.DataFrame(
        np.zeros((1, 978), dtype=np.float32),
        columns=[f"feature_{index}" for index in range(978)],
    )
    query.insert(0, "query_id", "drug_1")
    query_path = tmp_path / "query.tsv"
    output_path = tmp_path / "ranking.tsv"
    query.to_csv(query_path, sep="\t", index=False)

    code = main(
        [
            "predict",
            "cross-cell",
            "--mode",
            "direct",
            "--input",
            str(query_path),
            "--checkpoint",
            str(checkpoint),
            "--cell-line",
            "VCAP",
            "--output",
            str(output_path),
        ]
    )

    assert code == 0
    output = pd.read_csv(output_path, sep="\t")
    assert len(output) == target_count
    assert set(output["cell_line"]) == {"VCAP"}
