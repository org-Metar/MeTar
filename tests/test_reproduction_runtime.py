"""Reproduction entrypoint behavior without running scientific evaluations."""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def tmp_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("reproduction_runtime")
    return Path("\\\\?\\" + str(path)) if sys.platform == "win32" else path


@pytest.fixture
def reproducer(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setenv("METAR_PROGRESS", "1")
    spec = importlib.util.spec_from_file_location(
        "metar_reproduction_entry", scripts / "reproduce_results.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_layout(package: Path) -> None:
    write = package / "compat/home/lwn/claude/manuscript/write"
    for directory in (
        "claude_round252_epiqgate_pseudoval_sweep_20260520",
        "claude_round259_crosscell_fullspace_round252query_then_sc_proteomics_20260522",
    ):
        (write / directory).mkdir(parents=True)


def test_unavailable_cuda_fails_before_preparation_or_output_creation(
    reproducer, tmp_path, monkeypatch, capsys
):
    output = tmp_path / "outputs"

    def unavailable(requested):
        assert requested == "cuda:7"
        raise RuntimeError("CUDA device cuda:7 is unavailable")

    def unexpected(*args):
        pytest.fail("device validation must precede file preparation")

    monkeypatch.setattr(reproducer, "resolve_device", unavailable)
    monkeypatch.setattr(reproducer, "prepare_args", unexpected)
    assert (
        reproducer.main(
            [
                "--package",
                str(tmp_path / "absent"),
                "--output",
                str(output),
                "--device",
                "cuda:7",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "CUDA device cuda:7 is unavailable" in captured.err
    assert "Traceback" not in captured.err and "[Done]" not in captured.err
    assert not output.exists()


def test_success_retains_stdout_json_and_resolves_auto_before_checks(
    reproducer, tmp_path, monkeypatch, capsys
):
    package, output = tmp_path / "artifacts", tmp_path / "outputs"
    prepare_layout(package)
    requests, calls = [], []

    def resolve(requested):
        requests.append(requested)
        return "cpu"

    def check(args):
        calls.append(args.device)
        return {"status": "PASS", "rows": 2}

    monkeypatch.setattr(reproducer, "resolve_device", resolve)
    monkeypatch.setattr(
        reproducer, "git_identity", lambda repo: ("test-revision", False)
    )
    for name in (
        "verify_proteomics",
        "verify_full_space",
        "verify_cross_cell",
        "verify_single_cell",
    ):
        monkeypatch.setattr(reproducer, name, check)
    assert reproducer.main(["--package", str(package), "--output", str(output)]) == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["status"] == "PASS" and report["device"] == "cpu"
    assert len(report["interfaces"]) == 4
    assert requests == ["auto"] and calls == ["cpu"] * 4
    assert json.loads((output / "all_article_exact_report.json").read_text()) == report
    assert "[Step 1]" in captured.err
    assert "[Done] Article reproduction" in captured.err
    assert "Reproduction scenarios: 4/4" in captured.err


@pytest.mark.parametrize("raises", [False, True])
def test_failed_comparison_or_exception_is_never_reported_done(
    reproducer, tmp_path, monkeypatch, capsys, raises
):
    package, output = tmp_path / "artifacts", tmp_path / "outputs"
    prepare_layout(package)

    def check(args):
        if raises:
            raise ValueError("reference rows do not match")
        return {"status": "FAIL"}

    monkeypatch.setattr(reproducer, "resolve_device", lambda requested: "cpu")
    monkeypatch.setattr(
        reproducer, "git_identity", lambda repo: ("test-revision", False)
    )
    monkeypatch.setattr(reproducer, "verify_full_space", check)
    code = reproducer.main(
        ["--mode", "full-space", "--package", str(package), "--output", str(output)]
    )
    captured = capsys.readouterr()
    assert code == (2 if raises else 1)
    assert "[Failed] Article reproduction" in captured.err
    assert "[Done]" not in captured.err and "Traceback" not in captured.err
    if raises:
        assert captured.out == ""
        assert "reference rows do not match" in captured.err
    else:
        assert json.loads(captured.out)["status"] == "FAIL"


@pytest.mark.parametrize(
    "name",
    [
        "verify_proteomics",
        "verify_full_space",
        "verify_cross_cell",
        "verify_single_cell",
    ],
)
def test_verification_helpers_reject_unavailable_cuda_before_loading_data(
    reproducer, monkeypatch, name
):
    function = getattr(reproducer, name)
    module = importlib.import_module(function.__module__)
    error = RuntimeError("CUDA unavailable before input loading")

    def unavailable(requested):
        assert requested == "cuda:0"
        raise error

    monkeypatch.setattr(module, "resolve_device", unavailable)
    with pytest.raises(RuntimeError) as captured:
        function(SimpleNamespace(device="cuda:0"))
    assert captured.value is error
