"""Device and terminal contracts without accelerator or training workloads."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from metar import cli, runtime
from metar.training import archived_training_command, training_command
from metar.training_runtime import _run_source


@pytest.fixture(autouse=True)
def enabled_reporting(monkeypatch):
    monkeypatch.setenv("METAR_PROGRESS", "1")


@pytest.fixture
def mocked_torch(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(
        torch.backends,
        "mps",
        SimpleNamespace(is_available=lambda: False),
    )
    return torch


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("auto", "cuda:0"),
        ("cuda", "cuda:2"),
        ("cuda:0", "cuda:0"),
        (" CUDA:3 ", "cuda:3"),
        ("cpu", "cpu"),
        ("cpu:0", "cpu"),
    ],
)
def test_resolve_device_preserves_requested_cuda_index(
    mocked_torch,
    requested: str,
    expected: str,
) -> None:
    assert runtime.resolve_device(requested) == expected


def test_explicit_cpu_does_not_probe_or_initialize_cuda(
    mocked_torch,
    monkeypatch,
) -> None:
    def unexpected():
        raise AssertionError("CPU must not probe CUDA")

    monkeypatch.setattr(mocked_torch.cuda, "is_available", unexpected)
    monkeypatch.setattr(mocked_torch.cuda, "current_device", unexpected)
    assert runtime.resolve_device("cpu") == "cpu"


def test_auto_uses_cpu_when_no_accelerator_is_available(
    mocked_torch,
    monkeypatch,
) -> None:
    monkeypatch.setattr(mocked_torch.cuda, "is_available", lambda: False)
    assert runtime.resolve_device() == "cpu"


def test_auto_uses_cpu_and_explicit_mps_uses_available_backend(
    mocked_torch,
    monkeypatch,
) -> None:
    monkeypatch.setattr(mocked_torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        mocked_torch.backends,
        "mps",
        SimpleNamespace(is_available=lambda: True),
    )
    assert runtime.resolve_device("auto") == "cpu"
    assert runtime.resolve_device("mps") == "mps"


@pytest.mark.parametrize(
    ("cuda_build", "reason"),
    [(None, "CPU-only Torch build"), ("12.4", "no usable CUDA driver/device")],
)
def test_explicit_unavailable_cuda_reports_actionable_error(
    mocked_torch,
    monkeypatch,
    cuda_build,
    reason: str,
) -> None:
    monkeypatch.setattr(mocked_torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(mocked_torch.version, "cuda", cuda_build)
    with pytest.raises(RuntimeError) as error:
        runtime.resolve_device("cuda:1")
    message = str(error.value)
    assert "cuda:1" in message and reason in message
    assert "--device cpu" in message and sys.executable in message


@pytest.mark.parametrize("requested", ["cuda:4", "cpu:2", "meta", "nonsense"])
def test_invalid_or_invisible_device_is_rejected(mocked_torch, requested) -> None:
    with pytest.raises(ValueError):
        runtime.resolve_device(requested)


def test_unavailable_mps_is_rejected(mocked_torch) -> None:
    with pytest.raises(RuntimeError, match="MPS.*unavailable"):
        runtime.resolve_device("mps")


def test_nonterminal_progress_throttles_and_always_shows_completion(
    monkeypatch,
) -> None:
    clock = [0.0]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock[0])
    output = StringIO()
    with runtime.RunReporter("Prediction", stream=output) as reporter:
        reporter.progress(0, 4, label="rows")
        clock[0] = 0.1
        reporter.progress(1, 4, label="rows")
        clock[0] = 15.0
        reporter.progress(2, 4, label="rows")
        clock[0] = 15.1
        reporter.progress(4, 4, label="rows")
        reporter.progress(4, 4, label="rows")
    text = output.getvalue()
    assert text.count("rows:") == 3
    assert "rows: 0/4" in text and "rows: 2/4" in text
    assert "rows: 1/4" not in text
    assert text.count("rows: 4/4 (100%)") == 1
    assert "\r" not in text and "\033" not in text
    assert "[Done] Prediction" in text


def test_terminal_progress_uses_short_interval_and_closes_bar(monkeypatch) -> None:
    class Terminal(StringIO):
        def isatty(self):
            return True

    clock = [0.0]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock[0])
    output = Terminal()
    with runtime.RunReporter("Training", stream=output) as reporter:
        reporter.progress(0, 3, label="batches")
        clock[0] = 0.49
        reporter.progress(1, 3, label="batches")
        clock[0] = 0.5
        reporter.progress(2, 3, label="batches")
        reporter.progress(3, 3, label="batches")
        reporter.step("Save checkpoint")
    text = output.getvalue()
    assert "batches: 1/3" not in text and "batches: 2/3" in text
    assert "\r" in text and "\033[K" in text
    assert "100%)\033[K\n[Step 1] Save checkpoint\n" in text


@pytest.mark.parametrize("quiet", [False, True])
def test_reporter_routes_to_stderr_and_quiet_keeps_stdout_clean(capsys, quiet):
    with runtime.RunReporter("Prediction", {"device": "cpu"}, quiet=quiet):
        runtime.report_step("Load model")
        runtime.report_progress(2, 2, label="queries")
        runtime.report_metrics({"Hit@10": 0.25, "AUROC": 0.75})
        print('{"queries": 2}')
    captured = capsys.readouterr()
    assert captured.out == '{"queries": 2}\n'
    if quiet:
        assert captured.err == ""
    else:
        assert "MeTar | Prediction" in captured.err
        assert "device: cpu" in captured.err and "Hit@10=0.25" in captured.err
        assert "[Done] Prediction" in captured.err


@pytest.mark.parametrize("value", ["0", "false", "OFF", "quiet"])
def test_environment_quiet_disables_all_reporter_messages(monkeypatch, value):
    monkeypatch.setenv("METAR_PROGRESS", value)
    output = StringIO()
    with runtime.RunReporter("Silent", stream=output) as reporter:
        reporter.step("Prepare")
        reporter.progress(1, 1, label="rows")
        reporter.metrics({"loss": 0.5})
    assert output.getvalue() == ""


def test_quiet_does_not_convert_metric_values() -> None:
    class NoFormatting:
        def __str__(self):
            raise AssertionError("quiet metrics must not convert values")

    with runtime.RunReporter("Silent", quiet=True) as reporter:
        reporter.metrics({"loss": NoFormatting()})
        reporter.progress(1, 1, label="rows", metrics={"loss": NoFormatting()})


def test_nested_reporters_restore_parent_context_after_exception() -> None:
    original = runtime._ACTIVE.get()
    outer_output = StringIO()
    inner_output = StringIO()
    with runtime.RunReporter("Outer", stream=outer_output):
        runtime.report_step("Outer before")
        with (
            pytest.raises(RuntimeError, match="training failed"),
            runtime.RunReporter("Inner", stream=inner_output),
        ):
            runtime.report_step("Inner only")
            raise RuntimeError("training failed")
        runtime.report_step("Outer after")
    assert runtime._ACTIVE.get() is original
    assert "Outer before" in outer_output.getvalue()
    assert "[Step 2] Outer after" in outer_output.getvalue()
    assert "Inner only" not in outer_output.getvalue()
    assert "[Failed] Inner | training failed" in inner_output.getvalue()
    assert "[Done] Inner" not in inner_output.getvalue()


@pytest.mark.parametrize("exit_code", [None, 0, 2])
def test_reporter_classifies_system_exit_without_swallowing_it(exit_code):
    output = StringIO()
    with pytest.raises(SystemExit), runtime.RunReporter("Source", stream=output):
        raise SystemExit(exit_code)
    text = output.getvalue()
    assert ("[Done] Source" in text) is (exit_code in (None, 0))
    assert ("[Failed] Source" in text) is (exit_code not in (None, 0))


@pytest.mark.parametrize("previous", [None, "off", "1"])
@pytest.mark.parametrize("failed", [False, True])
def test_cli_restores_progress_environment_on_success_and_failure(
    monkeypatch,
    capsys,
    previous,
    failed,
) -> None:
    if previous is None:
        monkeypatch.delenv("METAR_PROGRESS", raising=False)
    else:
        monkeypatch.setenv("METAR_PROGRESS", previous)
    observed = []

    def handler(args):
        observed.append(runtime.os.environ.get("METAR_PROGRESS"))
        runtime.report_step("Hidden by CLI quiet")
        if failed:
            raise RuntimeError("fixture failure")
        return 0

    parser = argparse.ArgumentParser()
    parser.set_defaults(command="predict", device="cpu", quiet=True, handler=handler)
    monkeypatch.setattr(cli, "build_parser", lambda: parser)
    monkeypatch.setattr(cli, "resolve_device", lambda value: value)
    monkeypatch.setattr(cli, "_validate_prediction_output", lambda args: None)
    if failed:
        with pytest.raises(SystemExit) as error:
            cli.main([])
        assert error.value.code == 2
    else:
        assert cli.main([]) == 0
    assert observed == ["0"]
    assert runtime.os.environ.get("METAR_PROGRESS") == previous
    assert "Hidden by CLI quiet" not in capsys.readouterr().err


@pytest.mark.parametrize("requested", ["cpu", "cuda:0", "cuda:3"])
def test_full_space_actual_source_receives_explicit_device_argv(
    tmp_path: Path,
    monkeypatch,
    requested: str,
) -> None:
    """Run the selected source only as far as device resolution, before data I/O."""
    torch = pytest.importorskip("torch")
    source = (
        Path(__file__).resolve().parents[1]
        / "training/modality_transfer/scripts/train_full_space.py"
    )
    script = tmp_path / "run_round259_crosscell_fullspace_round252query.py"
    shutil.copy2(source, script)
    archived = archived_training_command("full-space", tmp_path, device=requested)
    public = training_command("full-space", tmp_path, device=requested)
    assert public[public.index("--device") + 1] == requested
    assert archived[archived.index("--device") + 1] == requested
    received = []

    class Resolved(Exception):
        pass

    def resolve(value):
        received.append(value)
        raise Resolved

    monkeypatch.setattr(runtime, "resolve_device", resolve)
    # Explicit CPU must stay CPU even when an accelerator is available.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    original_argv, original_path = sys.argv, list(sys.path)
    try:
        with pytest.raises(Resolved):
            _run_source(script, archived[3:], device=requested)
    finally:
        sys.argv = original_argv
        sys.path[:] = original_path
    assert received == [requested]


def test_all_stage_training_failure_stops_chain_and_reports_failure(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    completed = []

    def run_stage(stage, *args, **kwargs):
        completed.append(stage)
        return 7 if stage == "closed-bulk" else 0

    monkeypatch.setattr(cli, "resolve_device", lambda requested: "cpu")
    monkeypatch.setattr(cli, "run_training_stage", run_stage)
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "train",
                "--stage",
                "all",
                "--workspace",
                str(tmp_path),
                "--device",
                "cpu",
            ]
        )
    assert error.value.code == 2
    assert completed == ["gene-prior", "closed-bulk"]
    captured = capsys.readouterr()
    assert "[Failed] Train" in captured.err
    assert "[Done] Train" not in captured.err
    assert "exit status 7" in captured.err
    assert "9/9" not in captured.err


@pytest.mark.parametrize(
    "ending", ["pass", "raise SystemExit(0)", "raise RuntimeError('source failure')"]
)
def test_source_execution_restores_process_arguments_and_import_path(
    tmp_path: Path,
    ending: str,
) -> None:
    script = tmp_path / "source.py"
    script.write_text(
        "import sys\n"
        "assert sys.argv[1:] == ['--device', 'cpu']\n"
        "sys.argv.append('source mutation')\n"
        "sys.path.insert(0, 'source-import-path')\n" + ending + "\n",
        encoding="utf-8",
    )
    original_argv = sys.argv
    original_path = list(sys.path)
    if "RuntimeError" in ending:
        with pytest.raises(RuntimeError, match="source failure"):
            _run_source(script, ["--device", "cpu"], device="cpu")
    else:
        _run_source(script, ["--device", "cpu"], device="cpu")
    assert sys.argv is original_argv
    assert sys.path == original_path


@pytest.fixture
def cache_builder_fixture(tmp_path: Path, monkeypatch):
    """Exercise the real cache orchestration with a tiny imported h07 builder."""
    torch = pytest.importorskip("torch")
    source = Path(__file__).resolve().parents[1] / "scripts/prepare_bulk_data.py"
    spec = importlib.util.spec_from_file_location("cache_builder_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    package = tmp_path / "package"
    public_inputs = {}
    for relative in module.SOURCE_HASHES:
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("tiny fixture: " + relative).encode())
        public_inputs[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(module, "SOURCE_HASHES", public_inputs)
    protocol = package / "data/processed/lincs/bulk_protocol_lock_v2.json"
    protocol.write_text('{"fixture": true}', encoding="utf-8")
    sources = tmp_path / "sources"
    sources.mkdir()
    wrapper = sources / "train_bulk_sharedsample_indunseenlabel_v1_20260330.py"
    wrapper.write_text(
        "from pathlib import Path\n"
        "from types import SimpleNamespace\n"
        "import json\n"
        "import os\n"
        "import torch\n"
        f"EXPECTED = {module.EXPECTED_AUDIT!r}\n"
        "def prepare_data(protocol, device):\n"
        "    root = Path(os.environ['METAR_BULK_PROJECT_ROOT'])\n"
        "    derived = root / 'derived_tables'\n"
        "    assert derived.is_dir(), 'missing derived_tables before preparation'\n"
        "    (derived / 'label_partition.json').write_text('{}')\n"
        "    assert bulk_base.DATA_DIR.is_dir()\n"
        "    assert bulk_base.MANIFEST_PATH.is_file()\n"
        "    assert Path(os.environ['METAR_BULK_PROTOCOL_PATH']).is_file()\n"
        "    return {'audit': EXPECTED, 'x': torch.tensor([1., 2.], device=device)}\n"
        "bulk_base = SimpleNamespace(\n"
        "    prepare_data=prepare_data,\n"
        "    load_protocol=lambda path: json.loads(path.read_text()),\n"
        ")\n",
        encoding="utf-8",
    )
    base = sources / "train_bulk_dualbranch_v2_20260328.py"
    base.write_text("# Required sibling source fixture\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "resolve_source",
        lambda relative, *args: sources / Path(relative).name,
    )
    monkeypatch.setenv("METAR_BULK_PROJECT_ROOT", "original-root")
    monkeypatch.delenv("METAR_BULK_PROTOCOL_PATH", raising=False)
    return module, package, torch


@pytest.mark.parametrize("existing_graph", [False, True])
def test_cache_builder_creates_required_directory_preserves_inputs_and_environment(
    cache_builder_fixture,
    existing_graph: bool,
) -> None:
    module, package, torch = cache_builder_fixture
    ppi_source = package / "data/processed/fullspace/pertkge_ppi_graph.csv"
    compatibility = package / "data/processed/lincs/pertkge_ppi_graph.csv"
    if existing_graph:
        # The same compatibility input must survive regardless of the platform's
        # symlink privileges; use the real link when available.
        try:
            compatibility.symlink_to(ppi_source)
        except OSError:
            shutil.copyfile(ppi_source, compatibility)
    was_symlink = compatibility.is_symlink()
    before = ppi_source.read_bytes()
    report = module.prepare_cache(SimpleNamespace(package=package, device="cpu"))
    output = Path(report["output"])
    assert report["status"] == "PASS"
    assert report["audit"] == module.EXPECTED_AUDIT
    assert report["output_sha256"] == module.sha256_file(output)
    assert json.loads(output.with_suffix(".audit.json").read_text()) == report
    cached = torch.load(output, map_location="cpu", weights_only=False)
    assert cached["protocol"] == {"fixture": True}
    assert cached["data"]["x"].tolist() == [1.0, 2.0]
    assert ppi_source.read_bytes() == before
    assert compatibility.exists() is existing_graph
    assert compatibility.is_symlink() is was_symlink
    assert module.os.environ["METAR_BULK_PROJECT_ROOT"] == "original-root"
    assert "METAR_BULK_PROTOCOL_PATH" not in module.os.environ
    assert list(output.parent.glob(".h07_*.pt")) == []


def test_cache_save_failure_leaves_no_partial_output_or_temporary_graph(
    cache_builder_fixture,
    monkeypatch,
) -> None:
    module, package, torch = cache_builder_fixture
    output = package / "data/processed/fullspace/h07_prepared_data.pt"
    graph = package / "data/processed/lincs/pertkge_ppi_graph.csv"

    def fail_save(payload, path):
        Path(path).write_bytes(b"partial cache")
        raise OSError("fixture disk write failed")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="fixture disk write failed"):
        module.prepare_cache(SimpleNamespace(package=package, device="cpu"))
    assert not output.exists()
    assert not output.with_suffix(".audit.json").exists()
    assert list(output.parent.glob(".h07_*.pt")) == []
    assert not graph.exists() and not graph.is_symlink()
    assert module.os.environ["METAR_BULK_PROJECT_ROOT"] == "original-root"
    assert "METAR_BULK_PROTOCOL_PATH" not in module.os.environ


@pytest.mark.parametrize(
    "error",
    [RuntimeError("CUDA is unavailable; use cpu"), ImportError("install .[training]")],
)
def test_cache_main_reports_preflight_failure_without_writing(
    cache_builder_fixture, monkeypatch, capsys, error
):
    module, package, _torch = cache_builder_fixture
    monkeypatch.setattr(
        module, "parse_args", lambda: SimpleNamespace(package=package, device="cpu")
    )

    def failed(_stage):
        raise error

    monkeypatch.setattr(module, "check_training_dependencies", failed)
    assert module.main() == 2
    captured = capsys.readouterr()
    assert str(error) in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""
    assert not (package / "data/processed/fullspace/h07_prepared_data.pt").exists()
