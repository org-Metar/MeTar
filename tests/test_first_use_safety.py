"""First-run filesystem checks without loading checkpoints or assay datasets."""

from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from metar.cli import (
    DEFAULT_SINGLE_CELL,
    DEFAULTS,
    _validate_prediction_output,
    build_parser,
    main,
)
from metar.training_workspace import (
    MARKER,
    WRITE_ROOT,
    _link,
    prepare_training_workspace,
    pretrained_inputs,
    workspace_inputs,
)


@pytest.fixture
def tmp_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("first_use")
    return Path("\\\\?\\" + str(path)) if sys.platform == "win32" else path


def test_module_entry_point_prints_description_without_model_assets():
    code = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-m", "metar", "describe"],
        cwd=code,
        env={**os.environ, "PYTHONPATH": str(code / "src")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "bulk-668" in completed.stdout
    assert json.loads(completed.stdout)
    assert completed.stderr == ""


def test_empty_query_file_has_actionable_error(tmp_path, capsys):
    source = tmp_path / "query.npz"
    source.touch()
    with pytest.raises(SystemExit) as error:
        main(
            [
                "predict",
                "bulk-668",
                "--input",
                str(source),
                "--output",
                str(tmp_path / "scores.tsv"),
                "--device",
                "cpu",
            ]
        )
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "empty" in captured.err.lower()
    assert "Traceback" not in captured.err
    assert not (tmp_path / "scores.tsv").exists()


def test_out_of_memory_has_recovery_hint_without_retry(monkeypatch, tmp_path, capsys):
    from metar import cli

    attempts = []

    def exhausted(args):
        attempts.append(args.device)
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(cli, "command_bulk_668", exhausted)
    with pytest.raises(SystemExit) as error:
        main(
            [
                "predict",
                "bulk-668",
                "--input",
                str(tmp_path / "query.npz"),
                "--output",
                str(tmp_path / "scores.tsv"),
                "--device",
                "cpu",
            ]
        )
    assert error.value.code == 2
    assert attempts == ["cpu"]
    captured = capsys.readouterr()
    assert "lower --batch-size" in captured.err
    assert "Traceback" not in captured.err
    assert not (tmp_path / "scores.tsv").exists()


@pytest.mark.parametrize(
    "error_code", [errno.EPERM, errno.EACCES, errno.ENOTSUP, "win1314"]
)
def test_input_link_permission_fallback_copies_bytes(tmp_path, monkeypatch, error_code):
    source = tmp_path / "input.bin"
    source.write_bytes(b"read-only scientific input")
    destination = tmp_path / "workspace/input.bin"
    error = OSError(
        error_code if isinstance(error_code, int) else errno.EIO, "link unavailable"
    )
    if error_code == "win1314":
        error.winerror = 1314

    def denied(*args, **kwargs):
        raise error

    monkeypatch.setattr(Path, "symlink_to", denied)
    assert _link(source, destination) is True
    assert destination.read_bytes() == source.read_bytes()
    assert not destination.is_symlink()
    destination.write_bytes(b"private change")
    assert source.read_bytes() == b"read-only scientific input"


@pytest.mark.parametrize("error_code", [errno.ENOSPC, errno.EIO])
def test_unexpected_link_errors_are_not_hidden_by_copy(
    tmp_path, monkeypatch, error_code
):
    source = tmp_path / "input.bin"
    source.write_bytes(b"scientific input")
    destination = tmp_path / "workspace/input.bin"

    def failed(*args, **kwargs):
        raise OSError(error_code, "unexpected filesystem failure")

    monkeypatch.setattr(Path, "symlink_to", failed)
    with pytest.raises(OSError) as error:
        _link(source, destination)
    assert error.value.errno == error_code
    assert not destination.exists()
    assert source.read_bytes() == b"scientific input"


def test_explicit_input_copy_does_not_attempt_link(tmp_path, monkeypatch):
    source = tmp_path / "input.bin"
    source.write_bytes(b"input")

    def unexpected(*args, **kwargs):
        pytest.fail("--copy-inputs must not try to create symlinks")

    monkeypatch.setattr(Path, "symlink_to", unexpected)
    destination = tmp_path / "copy/input.bin"
    assert _link(source, destination, copy_inputs=True) is False
    assert destination.read_bytes() == b"input"


def test_workspace_fallback_records_mode_and_emits_one_notice(
    tmp_path, monkeypatch, capsys
):
    code = Path(__file__).resolve().parents[1]
    artifacts = tmp_path / "artifacts"
    external, metadata = workspace_inputs("single-cell", "A549")
    for relative in (*external, *metadata, *pretrained_inputs("single-cell")):
        path = artifacts / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture scientific input")
    original = {
        path: path.read_bytes() for path in artifacts.rglob("*") if path.is_file()
    }

    def denied(*args, **kwargs):
        raise OSError(errno.EPERM, "symlink unavailable")

    monkeypatch.setattr(Path, "symlink_to", denied)
    output = tmp_path / "workspace"
    report = prepare_training_workspace(
        code,
        artifacts,
        output,
        stage="single-cell",
        cell_line="A549",
        reuse_pretrained=True,
    )
    assert report["input_mode"] == "copy_fallback"
    assert json.loads((output / MARKER).read_text())["input_mode"] == "copy_fallback"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("input symlinks are unavailable") == 1
    for relative in external:
        canonical = output / relative
        alias = output / "compat/home/lwn/claude/sc/data" / Path(relative).name
        assert (
            canonical.read_bytes()
            == alias.read_bytes()
            == (artifacts / relative).read_bytes()
        )
        assert not canonical.is_symlink() and not alias.is_symlink()
    (output / external[0]).write_bytes(b"private change")
    assert all(path.read_bytes() == data for path, data in original.items())
    assert any((output / WRITE_ROOT).rglob("*.py"))


def test_cli_rejects_query_overwrite_before_loading_models(tmp_path, capsys):
    source = tmp_path / "queries.tsv"
    source.write_bytes(b"valuable original input")
    with pytest.raises(SystemExit) as error:
        main(["predict", "bulk-668", "--input", str(source), "--output", str(source)])
    assert error.value.code == 2
    assert source.read_bytes() == b"valuable original input"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "would overwrite an input or model asset" in captured.err


@pytest.mark.parametrize(
    "scenario,default_name",
    [
        ("bulk-668", "closed_checkpoint"),
        ("bulk-668", "feature_order"),
        ("full-space", "full_checkpoint"),
        ("full-space", "target_table"),
        ("cross-cell", "adapter_checkpoint"),
        ("proteomics", "proteomics_checkpoint"),
        ("proteomics", "proteomics_target_table"),
        ("proteomics", "proteomics_preprocessor"),
        ("proteomics", "proteomics_reference"),
    ],
)
def test_output_guard_protects_selected_default_assets(
    tmp_path, monkeypatch, scenario, default_name
):
    asset = tmp_path / "selected_asset.tsv"
    asset.write_bytes(b"release asset")
    monkeypatch.setitem(DEFAULTS, default_name, asset)
    if default_name == "proteomics_reference":
        monkeypatch.setitem(
            DEFAULTS, "proteomics_preprocessor", tmp_path / "absent.json"
        )
    args = build_parser().parse_args(
        [
            "predict",
            scenario,
            "--input",
            str(tmp_path / "query.npz"),
            "--output",
            str(asset),
        ]
    )
    with pytest.raises(ValueError, match="overwrite"):
        _validate_prediction_output(args)
    assert asset.read_bytes() == b"release asset"


@pytest.mark.parametrize(
    "scenario,option",
    [
        ("bulk-668", "--checkpoint"),
        ("bulk-668", "--feature-order"),
        ("full-space", "--target-table"),
        ("cross-cell", "--support-input"),
        ("cross-cell", "--support-labels"),
        ("cross-cell", "--support-weights"),
        ("cross-cell", "--adapter-checkpoint"),
        ("proteomics", "--preprocessor"),
        ("proteomics", "--reference-data"),
        ("consensus", "--target-order"),
    ],
)
def test_output_guard_protects_explicit_input_roles(tmp_path, scenario, option):
    asset = tmp_path / "explicit_input.tsv"
    asset.write_bytes(b"original user input")
    args = build_parser().parse_args(
        [
            "predict",
            scenario,
            "--input",
            str(tmp_path / "query.npz"),
            option,
            str(asset),
            "--output",
            str(asset),
        ]
    )
    with pytest.raises(ValueError, match="overwrite"):
        _validate_prediction_output(args)
    assert asset.read_bytes() == b"original user input"


def test_output_guard_protects_selected_single_cell_default(tmp_path, monkeypatch):
    asset = tmp_path / "single.pth"
    asset.write_bytes(b"release weights")
    monkeypatch.setitem(DEFAULT_SINGLE_CELL, "A549", asset)
    args = build_parser().parse_args(
        [
            "predict",
            "single-cell",
            "--cell-line",
            "A549",
            "--input",
            str(tmp_path / "query.npz"),
            "--output",
            str(asset),
        ]
    )
    with pytest.raises(ValueError, match="overwrite"):
        _validate_prediction_output(args)


def test_unused_default_adapter_does_not_block_direct_output(tmp_path, monkeypatch):
    output = tmp_path / "unused_adapter.tsv"
    output.write_text("old prediction")
    monkeypatch.setitem(DEFAULTS, "adapter_checkpoint", output)
    args = build_parser().parse_args(
        [
            "predict",
            "cross-cell",
            "--mode",
            "direct",
            "--input",
            str(tmp_path / "query.npz"),
            "--output",
            str(output),
        ]
    )
    _validate_prediction_output(args)


def test_output_guard_rejects_hardlink_to_input(tmp_path):
    source = tmp_path / "input.tsv"
    source.write_bytes(b"original")
    output = tmp_path / "output.tsv"
    try:
        output.hardlink_to(source)
    except OSError:
        pytest.skip("hardlinks unavailable on this filesystem")
    args = build_parser().parse_args(
        ["predict", "consensus", "--input", str(source), "--output", str(output)]
    )
    with pytest.raises(ValueError, match="overwrite"):
        _validate_prediction_output(args)


def test_preprocessor_export_cannot_replace_reference_input(tmp_path, monkeypatch):
    monkeypatch.setitem(DEFAULTS, "proteomics_preprocessor", tmp_path / "absent.json")
    reference = tmp_path / "reference.csv"
    reference.write_bytes(b"original training reference")
    args = build_parser().parse_args(
        [
            "predict",
            "proteomics",
            "--input",
            str(tmp_path / "query.tsv"),
            "--reference-data",
            str(reference),
            "--write-preprocessor",
            str(reference),
            "--output",
            str(tmp_path / "out.tsv"),
        ]
    )
    with pytest.raises(ValueError, match="overwrite"):
        _validate_prediction_output(args)


def test_distinct_prediction_destination_can_be_reused(tmp_path):
    output = tmp_path / "prediction.tsv"
    output.write_text("previous predictions")
    args = build_parser().parse_args(
        [
            "predict",
            "bulk-668",
            "--input",
            str(tmp_path / "query.npz"),
            "--output",
            str(output),
        ]
    )
    _validate_prediction_output(args)


def test_prediction_and_preprocessor_exports_must_use_distinct_paths(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(DEFAULTS, "proteomics_preprocessor", tmp_path / "absent.json")
    output = tmp_path / "output.tsv"
    args = build_parser().parse_args(
        [
            "predict",
            "proteomics",
            "--input",
            str(tmp_path / "query.tsv"),
            "--write-preprocessor",
            str(output),
            "--output",
            str(output),
        ]
    )
    with pytest.raises(ValueError, match="outputs must use different paths"):
        _validate_prediction_output(args)
