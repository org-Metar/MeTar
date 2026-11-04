from __future__ import annotations

import json
import sys
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
    pretrained_inputs,
    TARGET_TABLE,
    workspace_inputs,
)


@pytest.fixture
def tmp_path(tmp_path_factory) -> Path:
    """Allow the archived compatibility basenames on Windows test machines."""
    path = tmp_path_factory.mktemp("training")
    return Path("\\\\?\\" + str(path)) if sys.platform == "win32" else path


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


@pytest.mark.parametrize(
    ("stage", "cell_line"),
    [
        ("single-cell", "A549"),
        ("single-cell", "SW480"),
        ("single-cell", "NCI-H460"),
        ("proteomics", None),
        ("closed-bulk", None),
        ("support-initializer", None),
        ("cross-cell-adapter", None),
        ("full-space", None),
    ],
)
def test_stage_workspace_reuses_only_private_upstream_inputs(
    tmp_path: Path, stage: str, cell_line: str | None,
) -> None:
    from metar.training import LOCKED_OUTPUTS, ROUND259

    code = Path(__file__).resolve().parents[1]
    artifacts = tmp_path / "artifacts"
    external, metadata = workspace_inputs(stage, cell_line)
    reused = pretrained_inputs(stage)
    for relative in (*external, *metadata, *reused):
        path = artifacts / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}" if path.suffix == ".json" else "upstream fixture")
    if stage == "proteomics":
        path = artifacts / GENE_ORDER_SOURCE
        pd.DataFrame([[1.0]], index=["G1"]).to_csv(path)
    original_bytes = {relative: (artifacts / relative).read_bytes() for relative in reused}
    output = tmp_path / "training"
    report = prepare_training_workspace(
        code, artifacts, output, stage=stage, cell_line=cell_line,
        reuse_pretrained=True, copy_inputs=True,
    )
    assert report["prepared_stages"] == [stage]
    for mutable in output.rglob("*.py"):
        if "compat" in mutable.parts:
            assert not mutable.is_symlink()
            assert "/NFS2_home/NFS2_home_1/lwn" not in mutable.read_text(encoding="utf-8")
    assert not (output / "checkpoints/selected").exists()
    if stage in LOCKED_OUTPUTS:
        assert not (output / LOCKED_OUTPUTS[stage]).exists()
    for relative, target in reused.items():
        copied = output / target
        assert copied.is_file() and not copied.is_symlink()
        assert copied.read_bytes() == original_bytes[relative]
        copied.write_text("local change")
        assert (artifacts / relative).read_bytes() == original_bytes[relative]
    if stage in {"single-cell", "proteomics"}:
        assert not (output / "data/processed/lincs").exists()
        assert not (output / "data/public_raw/gene_prior").exists()
        assert (output / ROUND259 / "source_data/round259_target_embedding.csv").is_file()
    if stage == "support-initializer":
        metadata_path = (
            output / WRITE_ROOT / ROUND_NAMES["r246"]
            / "source_data/selected_r246_joint_lm0p5_sc0p15_mn2p5.json"
        )
        selected = json.loads(metadata_path.read_text())
        assert (metadata_path.parent.parent / selected["checkpoint"]).is_file()
    if stage == "full-space":
        split_path = (
            output / WRITE_ROOT / ROUND_NAMES["r252"] / "source_data/cell_splits.json"
        )
        assert split_path.is_file() and not split_path.is_symlink()
    command = training_command(stage, output, device="cpu", cell_line=cell_line)
    # A full-space run needs a derived cache, which is deliberately absent here.
    if stage == "full-space":
        cache = output / "data/processed/fullspace/h07_prepared_data.pt"
        cache.write_bytes(b"cache fixture")
    validate_training_workspace(output, command, stage)


def test_stage_preparation_inspection_does_not_require_other_assays(tmp_path: Path) -> None:
    code = Path(__file__).resolve().parents[1]
    report = prepare_training_workspace(
        code, tmp_path / "artifacts", tmp_path / "output", check_only=True,
        stage="single-cell", cell_line="SW480", reuse_pretrained=True,
    )
    paths = "\n".join(report["missing_inputs"])
    assert "tahoe_sw480.h5ad" in paths and TARGET_TABLE in paths
    assert "proteomicsdb" not in paths and "tahoe_a549" not in paths
    assert "large_data.parquet" not in paths
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    ("stage", "cell_line", "source_name"),
    [
        ("single-cell", "A549", "run_round260_sc_round259_embedding.py"),
        ("single-cell", "SW480", "run_sc_label_bag_calibration_multicell_base.py"),
        ("single-cell", "NCI-H460", "run_sc_label_bag_calibration_multicell_base.py"),
        ("proteomics", None, "run_round261_proteomics_round259_embedding.py"),
    ],
)
def test_stage_check_reports_missing_runtime_source(
    tmp_path: Path, monkeypatch, capsys,
    stage: str, cell_line: str | None, source_name: str,
) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("dependency checks must not start training")

    monkeypatch.setattr("metar.training.subprocess.run", fail)
    code = Path(__file__).resolve().parents[1]
    artifacts = tmp_path / "artifacts"
    external, metadata = workspace_inputs(stage, cell_line)
    for relative in (*external, *metadata, *pretrained_inputs(stage)):
        path = artifacts / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}" if path.suffix == ".json" else "fixture")
    if stage == "proteomics":
        pd.DataFrame([[1.0]], index=["G1"]).to_csv(artifacts / GENE_ORDER_SOURCE)
    output = tmp_path / "training"
    prepare_training_workspace(
        code, artifacts, output, stage=stage, cell_line=cell_line,
        reuse_pretrained=True, copy_inputs=True,
    )
    assert run_training_stage(
        stage, output, device="cpu", cell_line=cell_line, check_only=True,
    ) == 0
    capsys.readouterr()
    command = training_command(stage, output, device="cpu", cell_line=cell_line)
    source = next(
        relative for relative in stage_inputs(stage, command)
        if Path(relative).name == source_name
    )
    (output / source).unlink()
    assert run_training_stage(
        stage, output, device="cpu", cell_line=cell_line, check_only=True,
    ) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["missing_inputs"] == [source]


@pytest.mark.parametrize(
    "options", [{"stage": "single-cell"}, {"reuse_pretrained": True},
                {"stage": "gene-prior", "reuse_pretrained": True}],
)
def test_invalid_stage_preparation_never_creates_directory(tmp_path: Path, options) -> None:
    with pytest.raises(ValueError):
        prepare_training_workspace(tmp_path / "code", tmp_path / "artifacts",
                                   tmp_path / "output", **options)
    assert not (tmp_path / "output").exists()
