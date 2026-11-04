"""Small regressions for bulk device propagation and real evaluation progress."""

from __future__ import annotations

import ast
import csv
import random
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from metar import runtime, training_runtime
from metar.training_workspace import ROUND_NAMES, WRITE_ROOT

ROOT = Path(__file__).resolve().parents[1]


def load_function(relative: str, name: str, namespace: dict):
    """Load a pure entrypoint/helper without loading historical data modules."""
    path = ROOT / "training" / relative
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    # Execute only isolated functions parsed from trusted first-party sources.
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
    return namespace[name]


@pytest.mark.parametrize(
    ("relative", "phase", "runner"),
    [
        ("closed_bulk/train_closed_bulk.py", "joint", "run_training"),
        ("support_initializer/train_support_initializer.py", "train", "run_eval"),
    ],
)
@pytest.mark.parametrize("requested", ["cpu", "cuda:0", "cuda:3"])
def test_bulk_entrypoints_preserve_explicit_device(
    relative: str,
    phase: str,
    runner: str,
    requested: str,
) -> None:
    resolved = []
    executed = []
    args = SimpleNamespace(device=requested, phase=phase)

    def resolve(value):
        resolved.append(value)
        return value

    namespace = {
        "ensure_dirs": lambda: None,
        "parse_args": lambda: args,
        "set_seed": lambda seed: None,
        "SEED": 1,
        "require_torch": lambda: None,
        "torch": SimpleNamespace(device=lambda value: value),
        "resolve_device": resolve,
        runner: lambda received_args, device: executed.append(device),
    }
    load_function(relative, "main", namespace)()
    assert resolved == [requested]
    assert executed == [requested]


def test_explicit_unavailable_device_fails_before_training() -> None:
    def unavailable(value):
        raise RuntimeError("CUDA was requested but is unavailable")

    executed = []
    namespace = {
        "ensure_dirs": lambda: None,
        "parse_args": lambda: SimpleNamespace(device="cuda:0", phase="joint"),
        "set_seed": lambda seed: None,
        "SEED": 1,
        "torch": SimpleNamespace(device=lambda value: value),
        "resolve_device": unavailable,
        "run_training": lambda *args: executed.append(args),
    }
    main = load_function("closed_bulk/train_closed_bulk.py", "main", namespace)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        main()
    assert not executed


@pytest.mark.parametrize(
    "relative",
    ["gene_prior/pretrain_gene_prior.py", "bulk_readout/bulk_readout.py"],
)
def test_readout_progress_preserves_scores_and_batch_budget(relative: str) -> None:
    torch = pytest.importorskip("torch")
    calls = []
    progress = []

    class Model(torch.nn.Module):
        def forward(self, x):
            calls.append(len(x))
            return x * 2 + 1

    predict = load_function(
        relative,
        "predict",
        {
            "torch": torch,
            "np": np,
            "report_progress": lambda current, total, **kwargs: progress.append(
                (current, total)
            ),
        },
    )
    x = np.arange(14, dtype=np.float32).reshape(7, 2)
    scores = predict(Model(), x, torch.device("cpu"), batch_size=3)
    np.testing.assert_array_equal(scores, x * 2 + 1)
    assert calls == [3, 3, 1]
    assert progress == [(0, 7), (3, 7), (6, 7), (7, 7)]


def test_epiqgate_evaluation_counts_completed_blocks_and_keeps_metrics() -> None:
    calls = []
    progress = []

    def evaluate(*args):
        # Positional support size and selector are part of the locked evaluator.
        calls.append((args[9], args[10], args[7]))
        return (
            pd.DataFrame([{"Hit@10": 0.25}]),
            pd.DataFrame([{"cell_line": "C1"}]),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    evaluate_all = load_function(
        "cross_cell_adapter/train_cross_cell_adapter.py",
        "evaluate_epiqgate_all",
        {
            "pd": pd,
            "evaluate_epiqgate": evaluate,
            "EPIQGATE_DEFAULT": {"shift_scale": 0.15},
            "report_step": lambda message: None,
            "report_progress": lambda current, total, **kwargs: progress.append(
                (current, total)
            ),
        },
    )
    metrics, cells, per, audit = evaluate_all(
        {},
        None,
        None,
        "cpu",
        {"variant": "test"},
        support_sizes=[0, 8, 32, -1],
        selectors=["top_count", "diverse_label"],
        keep_per_unit=True,
    )
    # Five actual splits per positive support size and selector; skipped sizes
    # retain their historical behavior and do not inflate the progress total.
    assert len(calls) == 20
    assert progress == [(value, 20) for value in range(21)]
    assert len(metrics) == len(cells) == 20
    assert metrics["Hit@10"].tolist() == [0.25] * 20
    assert per.empty and audit.empty


@pytest.mark.parametrize(
    ("stage", "round_key", "selected", "other_rows", "expected"),
    [
        (
            "closed-bulk",
            "r246",
            {
                "source": "Round246_joint_official_unseen",
                "variant": "r246_joint_lm0p5_sc0p15_mn2p5_official_unseen_no_adapt",
                "split": "official_test",
                "evaluation_unit": "drug-mean",
                "evaluation_family": "seen_668_no_adapt",
                "Hit@10": "0.3873962308754852",
                "MeanAP": "0.2240392648987411",
                "AUROC": "0.7997184822660499",
                "n_cells": "14",
                "n_groups": "1452",
                "n_profiles": "7754",
            },
            [
                {"evaluation_family": "seen_668_support_adapt", "Hit@10": "0.9"},
                {
                    "split": "official_test_profile",
                    "evaluation_unit": "true-profile-specific",
                },
                {"split": "pseudo_val"},
                {"split": "seen_test"},
            ],
            {"Hit@10": 0.3873962308754852, "n_cells": 14, "n_groups": 1452},
        ),
        (
            "support-initializer",
            "r247",
            {
                "source": "Round247_query_gated",
                "variant": "r247_query_gated_h128_sc0p15_mn2p5_official_test_diverse_label_s64",
                "split": "official_test",
                "evaluation_unit": "drug-mean",
                "evaluation_family": "seen_668_support_adapt",
                "selector": "diverse_label",
                "requested_support_size": "64.0",
                "Hit@10": "0.3982097258686542",
                "MeanAP": "0.22737784699459374",
                "AUROC": "0.8023274364912021",
                "n_cells": "14",
                "n_groups": "1452",
                "n_profiles": "7754",
            },
            [
                {"source": "Round246_cell_level_adapter"},
                {"selector": "top_count"},
                {"requested_support_size": "128.0", "Hit@10": "0.9"},
                {
                    "split": "official_test_profile",
                    "evaluation_unit": "true-profile-specific",
                },
            ],
            {"Hit@10": 0.3982097258686542, "n_cells": 14, "n_groups": 1452},
        ),
        (
            "cross-cell-adapter",
            "r252",
            {
                "source": "Round252_EpiQGate",
                "variant": "r252_epiqgate_seed20260524_official_unseen_diverse_label_s64",
                "split": "official_test",
                "evaluation_unit": "drug-mean",
                "evaluation_family": "seen_668_support_adapt",
                "selector": "diverse_label",
                "requested_support_size": "",
                "adapt_support_size": "64",
                "used_support_rows": "64.0",
                "Hit@10": "0.40051380071574744",
                "MeanAP": "0.22866396192602306",
                "AUROC": "0.800411789209039",
                "n_cells": "14",
                "n_groups": "1452",
                "n_profiles": "7754",
            },
            [
                {
                    "source": "Round251_QGateMAMLHybrid",
                    "evaluation_family": "seen_668_maml_hybrid",
                    "Hit@10": "0.4053554835656078",
                },
                {"source": "Round247_query_gated_official_unseen"},
                {"selector": "random"},
                {"adapt_support_size": "32"},
                {
                    "split": "official_test_profile",
                    "evaluation_unit": "true-profile-specific",
                },
                {"split": "pseudo_val"},
            ],
            {"Hit@10": 0.40051380071574744, "n_cells": 14, "n_groups": 1452},
        ),
    ],
)
def test_final_bulk_metrics_select_actual_official_drug_mean_schema(
    tmp_path,
    monkeypatch,
    stage,
    round_key,
    selected,
    other_rows,
    expected,
) -> None:
    """Rows/denominators mirror the completed 2026-11-04 scratch run tables."""
    rows = [{**selected, **changes} for changes in other_rows] + [selected]
    path = _write_metric_table(tmp_path, round_key, rows)
    original = path.read_bytes()
    captured = []
    messages = []
    monkeypatch.setattr(
        training_runtime,
        "report_metrics",
        lambda metrics, *, label: captured.append((metrics, label)),
    )
    monkeypatch.setattr(training_runtime, "report_step", messages.append)
    training_runtime.report_saved_metrics(tmp_path, stage)
    assert len(captured) == 1
    metrics, label = captured[0]
    assert {key: metrics[key] for key in expected} == pytest.approx(expected)
    assert metrics["n_profiles"] == 7754
    assert "official_test" in label and "cell-macro, drug-mean" in label
    assert messages == [f"Full metric table: {path}"]
    assert path.read_bytes() == original


def _write_metric_table(root, round_key, rows):
    path = root / WRITE_ROOT / ROUND_NAMES[round_key] / "source_data/metric_summary.tsv"
    path.parent.mkdir(parents=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_final_full_space_summary_excludes_higher_remove_seen_metric(
    tmp_path,
    monkeypatch,
) -> None:
    # Exact primary and conditional values/denominators from the earlier scratch
    # R259 result: removing the 668 trained labels gives a different, higher rate.
    seen = {
        "variant": "e13_bce001",
        "split": "test",
        "cell_split_role": "official_unseen_final",
        "evaluation_unit": "profile",
        "evaluation_family": "seen_668",
        "Hit@10": "0.2603817384575703",
        "MeanAP": "0.10007330638281858",
        "AUROC": "0.8014017247060777",
        "n_profiles": "7754",
        "best_epoch": "1",
    }
    primary = {
        "variant": "e13_bce001",
        "split": "holdout_test",
        "cell_split_role": "official_unseen_final",
        "evaluation_unit": "profile",
        "evaluation_family": "holdout_fullspace_top5",
        "n_profiles": "4463",
        "best_epoch": "1",
        "top5pct_success": "0.3269101501232355",
        "top10pct_success": "0.6139368138023751",
        "exclude_seen_labels": "False",
    }
    conditional = {
        **primary,
        "evaluation_family": "holdout_fullspace_top5_remove_seen",
        "top5pct_success": "0.5361864216894465",
        "top10pct_success": "",
        "exclude_seen_labels": "True",
    }
    path = _write_metric_table(
        tmp_path,
        "r259_261",
        [
            conditional,
            {**primary, "variant": "other_variant", "top5pct_success": "0.99"},
            {**primary, "split": "holdout_val"},
            {**primary, "split": "pc3_holdout_test"},
            seen,
            primary,
        ],
    )
    original = path.read_bytes()
    captured = []
    monkeypatch.setattr(
        training_runtime,
        "report_metrics",
        lambda metrics, *, label: captured.append((metrics, label)),
    )
    monkeypatch.setattr(training_runtime, "report_step", lambda message: None)
    training_runtime.report_saved_metrics(tmp_path, "full-space")
    assert len(captured) == 2
    assert captured[0][0]["Hit@10"] == pytest.approx(0.2603817384575703)
    metrics, label = captured[1]
    assert metrics == pytest.approx(
        {
            "top5pct_success": 0.3269101501232355,
            "top10pct_success": 0.6139368138023751,
            "n_profiles": 4463,
            "best_epoch": 1,
        }
    )
    assert "holdout_test | holdout_fullspace_top5 | profile" in label
    assert all("remove_seen" not in label for _, label in captured)
    assert path.read_bytes() == original


@pytest.mark.parametrize("quiet", [False, True])
def test_progress_and_metric_reporting_preserve_all_cpu_random_states(
    monkeypatch,
    quiet,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("METAR_PROGRESS", "1")
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            random.seed(321)
            np.random.seed(321)
            torch.default_generator.manual_seed(321)
            before_python = random.getstate()
            before_numpy = np.random.get_state()
            before_torch = torch.get_rng_state().clone()
            with runtime.RunReporter(
                "Training", {"seed": 321}, quiet=quiet, stream=StringIO()
            ):
                runtime.report_step("Existing evaluation")
                for completed in range(5):
                    runtime.report_progress(completed, 4, label="Episodes")
                runtime.report_metrics({"loss": 0.5, "Hit@10": 0.4}, label="Epoch 1/1")
            assert random.getstate() == before_python
            after_numpy = np.random.get_state()
            assert after_numpy[0] == before_numpy[0]
            np.testing.assert_array_equal(after_numpy[1], before_numpy[1])
            assert after_numpy[2:] == before_numpy[2:]
            assert torch.equal(torch.get_rng_state(), before_torch)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@pytest.mark.parametrize(
    "relative",
    [
        "closed_bulk/train_closed_bulk.py",
        "support_initializer/train_support_initializer.py",
        "cross_cell_adapter/train_cross_cell_adapter.py",
        "leakage_safe_support/leakage_safe_support.py",
        "episodic_meta_learning/episodic_meta_learning.py",
        "support_query_shift/support_query_shift.py",
        "gene_prior/pretrain_gene_prior.py",
        "bulk_readout/bulk_readout.py",
        "bulk_data/bulk_dual_branch.py",
        "bulk_data/prepare_inductive_bulk.py",
    ],
)
def test_core_epoch_reports_are_compact_and_preserve_full_history(relative):
    """Execute the actual display expressions without loading historical models."""
    source = ROOT / "training" / relative
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "report_metrics"
        and not any(
            keyword.arg == "label"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "Training settings"
            for keyword in node.keywords
        )
    ]
    assert calls
    # Include all numeric fields used by these scripts, plus distracting metadata.
    # A complete record remains available to checkpoint selection and TSV writers.
    record = {
        key: 0.25 for key in (
            "loss", "train_loss", "kg_loss", "kg_val_loss", "kg_val_pair_acc",
            "Hit@10", "MeanAP", "val_Hit@10", "val_MeanAP",
            "pseudo_val_Hit@10", "pseudo_val_MeanAP", "meta_seen_loss",
            "meta_unseen_loss", "meta_loss", "val_seen_hit10", "val_seen_map",
            "val_holdout_top5pct", "val_hit10", "val_meanap", "AUROC",
            "NDCG@10", "delta_norm", "gate_mean",
        )
    }
    record.update({"epoch": 2, "variant": "full_scientific_record"})
    original = record.copy()
    history = [record.copy() for _ in range(3)]
    original_history = [row.copy() for row in history]
    for call in calls:
        captured = []
        namespace = {
            "rec": record, "best": record, "best_val": record,
            "best_epoch": 2, "history_rows": history, "history": history,
            "cfg": {"max_epochs": 60}, "flow": {"kg_epochs": 30},
            "epoch": 3, "max_epochs": 60,
            "args": SimpleNamespace(n_epochs=60, pooled_epochs=60, meta_epochs=60),
            "report_metrics": lambda metrics, *, label, output=captured: output.append((metrics, label)),
        }
        expression = ast.Expression(call)
        ast.fix_missing_locations(expression)
        eval(compile(expression, str(source), "eval"), namespace)
        metrics, label = captured[0]
        assert 1 <= len(metrics) <= 5
        assert not any(
            forbidden in key
            for key in metrics
            for forbidden in ("variant", "AUROC", "NDCG", "delta", "gate")
        )
        if label.startswith("Selected"):
            # Early stopping after 3/60 is represented honestly; the best epoch
            # can precede the last completed epoch without forcing a 100% bar.
            assert metrics["epochs_run"] == 3
            assert metrics["best_epoch"] == 2
        else:
            assert any("loss" in key for key in metrics)
        assert record == original
        assert history == original_history
