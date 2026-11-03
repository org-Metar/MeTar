from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from metar.cli import build_parser, command_train
from metar.training import (
    run_training_stage,
    stage_inputs,
    training_command,
    validate_training_workspace,
)
from metar.training_runtime import _copy_generated, align_proteomics_targets
from metar.training_workspace import (
    EXTERNAL_INPUTS,
    GENE_ORDER_SOURCE,
    MARKER,
    METADATA_INPUTS,
    ROUND_NAMES,
    WRITE_ROOT,
    prepare_training_workspace,
)


def make_inputs(artifacts: Path) -> None:
    for relative in (*EXTERNAL_INPUTS, *METADATA_INPUTS):
        path = artifacts / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n" if path.suffix == ".json" else "tiny input\n")
    target = artifacts / GENE_ORDER_SOURCE
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([[99.0, 98.0], [97.0, 96.0]], index=["SEPT3", "G2"]).to_csv(target)
    frozen = artifacts / "checkpoints/selected/round246_whole_model_checkpoint.pth"
    frozen.parent.mkdir(parents=True)
    frozen.write_bytes(b"selected weights must not be copied or loaded")


def test_preparation_reports_all_missing_inputs_without_writing(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "training"
    report = prepare_training_workspace(
        root, tmp_path / "artifacts", output, check_only=True
    )
    assert report["status"] == "missing_inputs"
    assert any("gene_prior/process.txt" in path for path in report["missing_inputs"])
    assert not output.exists()


def test_training_workspace_copies_only_schema_and_uses_private_outputs(
    tmp_path: Path,
) -> None:
    code = Path(__file__).resolve().parents[1]
    artifacts = tmp_path / "artifacts"
    make_inputs(artifacts)
    output = tmp_path / "training"
    report = prepare_training_workspace(code, artifacts, output, copy_inputs=True)
    assert report["selected_checkpoints_copied"] is False
    assert (output / MARKER).is_file()
    assert not (output / "checkpoints").exists()
    assert not (
        output / "data/processed/fullspace/round259_target_embedding.csv"
    ).exists()
    assert (
        output / "data/processed/proteomics/gene_order.tsv"
    ).read_text() == "gene\nSEPT3\nG2\n"
    copied = (
        output
        / WRITE_ROOT
        / ROUND_NAMES["r230"]
        / "scripts/run_round230_scratch_mlp_metar_target_tower_pc3.py"
    )
    assert "H07_CACHE = Path(" in copied.read_text()
    assert not copied.is_symlink()
    legacy_bulk = output / "code/remote_dependencies/h07"
    assert (legacy_bulk / "train_bulk_sharedsample_indunseenlabel_v1_20260330.py").is_file()
    assert (legacy_bulk / "train_bulk_dualbranch_v2_20260328.py").is_file()
    cell_scripts = output / WRITE_ROOT / ROUND_NAMES["r259_261"] / "train_A549"
    assert (cell_scripts / "run_round260_sc_round259_embedding.py").is_file()
    assert (cell_scripts / "run_sc_label_bag_calibration_multicell_base.py").is_file()
    assert (artifacts / GENE_ORDER_SOURCE).read_text().find("99.0") >= 0
    with pytest.raises(FileExistsError):
        prepare_training_workspace(code, artifacts, output, copy_inputs=True)


def test_training_check_only_never_launches_subprocess(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("subprocess must not run during a dependency check")

    monkeypatch.setattr("metar.training.subprocess.run", fail)
    assert (
        run_training_stage("full-space", tmp_path, device="cpu", check_only=True) == 2
    )
    report = json.loads(capsys.readouterr().out)
    assert "data/processed/fullspace/h07_prepared_data.pt" in report["missing_inputs"]
    assert not report["prepared_workspace"]
    with pytest.raises(RuntimeError, match="isolated workspace"):
        run_training_stage("gene-prior", tmp_path, device="cpu")


def test_alignment_uses_retrained_vectors_with_locked_gene_names(
    tmp_path: Path,
) -> None:
    order = tmp_path / "data/processed/proteomics/gene_order.tsv"
    order.parent.mkdir(parents=True)
    order.write_text("gene\nSEPT3\nG2\n")
    fresh = tmp_path / "data/processed/fullspace/round259_target_embedding.csv"
    fresh.parent.mkdir(parents=True)
    pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=["G2", "3-Sep"]).to_csv(fresh)
    result = pd.read_csv(align_proteomics_targets(tmp_path), index_col=0)
    assert result.index.tolist() == ["SEPT3", "G2"]
    np.testing.assert_array_equal(result.to_numpy(), [[3.0, 4.0], [1.0, 2.0]])


def test_training_command_routes_device_through_isolated_runtime(
    tmp_path: Path,
) -> None:
    command = training_command("full-space", tmp_path, device="cuda:2")
    assert command[1:3] == ["-m", "metar.training_runtime"]
    assert command[command.index("--device") + 1] == "cuda:2"


def test_all_stages_use_distinct_assay_output_directories(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []

    def record(stage, workspace, **kwargs):
        calls.append((stage, kwargs.get("cell_line"), kwargs.get("output_dir")))
        return 0

    monkeypatch.setattr("metar.cli.run_training_stage", record)
    args = build_parser().parse_args(
        [
            "train",
            "--stage",
            "all",
            "--workspace",
            str(tmp_path),
            "--output-dir",
            "training_runs/custom",
            "--dry-run",
        ]
    )
    assert command_train(args) == 0
    outputs = [
        output
        for stage, _cell, output in calls
        if stage in {"single-cell", "proteomics"}
    ]
    assert len(outputs) == len(set(outputs)) == 4


def test_mutable_h07_directory_cannot_redirect_into_input_store(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / MARKER).write_text("{}")
    command = training_command("closed-bulk", root, device="cpu")
    for relative in stage_inputs("closed-bulk", command):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    external = tmp_path / "input_store"
    external.mkdir()
    redirected = (
        root
        / WRITE_ROOT
        / "bulk_sharedsample_indunseenlabel_20260330_h07/derived_tables"
    )
    redirected.parent.mkdir(parents=True)
    try:
        redirected.symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip(
            "directory symlinks unavailable; Windows junctions checked independently"
        )
    with pytest.raises(RuntimeError, match="escapes workspace"):
        validate_training_workspace(root, command, "closed-bulk")
    assert not list(external.iterdir())


def test_generated_copy_rechecks_destination_parent(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    source = root / "fresh.txt"
    source.write_text("fresh")
    external = tmp_path / "input_store"
    external.mkdir()
    (root / "checkpoints").mkdir()
    try:
        (root / "checkpoints/selected").symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks unavailable")
    with pytest.raises(RuntimeError, match="escapes workspace"):
        _copy_generated(source, root / "checkpoints/selected/output.pth", root=root)
    assert not list(external.iterdir())
