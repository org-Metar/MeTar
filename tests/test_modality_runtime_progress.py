from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
MODALITY = REPO / "training" / "modality_transfer"
SC_DIRECTORIES = ("scripts", "train_A549", "train_SW480", "train_NCI_H460")
FULLSPACE = MODALITY / "scripts" / "train_full_space.py"
ARCHITECTURE = (
    REPO / "training" / "full_space_architecture" / "full_space_architecture.py"
)


def functions_from(path: Path, names: tuple[str, ...], **context):
    """Exercise isolated runners without requiring the archived data or optional readers."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(definitions) == len(names)
    namespace = {
        "torch": torch,
        "np": np,
        "pd": pd,
        "Path": Path,
        "math": math,
        "json": json,
        "argparse": argparse,
        "importlib": importlib,
        **context,
    }
    isolated = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *definitions,
        ],
        type_ignores=[],
    )
    # Execute only function definitions extracted from trusted repository sources.
    exec(compile(ast.fix_missing_locations(isolated), str(path), "exec"), namespace)  # noqa: S102
    return namespace


@pytest.mark.parametrize("directory", SC_DIRECTORIES)
def test_single_cell_evaluation_counts_skips_without_changing_denominator(directory):
    progress = []
    namespace = functions_from(
        MODALITY / directory / "single_cell_model.py",
        ("eval_seen_hitk", "eval_holdout_all_percentiles", "rank_diagnostics"),
        KS=[1, 5, 10],
        HOLDOUT_PERCENTILES=[0.05, 0.10],
        report_progress=lambda current, total, **kwargs: progress.append(
            (current, total, kwargs)
        ),
        fetch_x_batch=lambda *_: torch.ones(2, 2),
    )

    class Model:
        def __init__(self):
            self.calls = 0

        def eval(self):
            pass

        def forward_logits_batch(self, x, targets):
            self.calls += 1
            return torch.arange(20.0, 0.0, -1.0).reshape(1, 20)

    model = Model()
    adata = SimpleNamespace(n_vars=2)
    cells = {"included": np.array([0, 1]), "unlabeled": np.array([2, 3])}
    metrics, rows = namespace["eval_seen_hitk"](
        model,
        adata,
        ["included", "missing_cells", "unlabeled"],
        cells,
        {"included": ["G0"], "missing_cells": ["G0"]},
        {"G0": 0},
        {},
        None,
        "val",
        "cpu",
    )
    assert model.calls == 1
    assert metrics == {
        "val_n_total": 3,
        "val_n_eval": 1,
        "val_n_skipped": 2,
        "val_Hit@1": 1.0,
        "val_Hit@5": 1.0,
        "val_Hit@10": 1.0,
    }
    assert [row["status"] for row in rows] == ["evaluated", "skipped", "skipped"]
    assert [(current, total) for current, total, _ in progress] == [
        (1, 3),
        (2, 3),
        (3, 3),
    ]

    progress.clear()
    metrics, rows = namespace["eval_holdout_all_percentiles"](
        model,
        adata,
        ["included", "missing_cells"],
        ["unlabeled"],
        cells,
        {"included": [0], "missing_cells": [0]},
        None,
        "cpu",
    )
    assert model.calls == 2
    assert metrics["holdout_all_n_eval"] == 1
    assert metrics["holdout_all_top5pct"] == 1.0
    assert len(rows) == 1
    assert [(current, total) for current, total, _ in progress] == [
        (1, 3),
        (2, 3),
        (3, 3),
    ]


@pytest.mark.parametrize("directory", SC_DIRECTORIES)
def test_single_cell_wrapper_finds_maintained_sibling_model(directory, tmp_path):
    model_path = tmp_path / "single_cell_model.py"
    model_path.write_text("SENTINEL = 'maintained model'\n", encoding="utf-8")
    namespace = functions_from(
        MODALITY / directory / "train_single_cell.py",
        ("load_base_module",),
        SCRIPT_DIR=tmp_path,
        REMOTE_BASE_SCRIPT=tmp_path / "absent_archived_name.py",
    )
    module, selected = namespace["load_base_module"]()
    assert selected == model_path
    assert module.SENTINEL == "maintained model"


@pytest.mark.parametrize("path", [FULLSPACE, ARCHITECTURE])
def test_full_space_accepts_explicit_cuda_index_and_auto_default(path, monkeypatch):
    namespace = functions_from(path, ("parse_args",), VARIANTS={"existing": {}})
    monkeypatch.setattr(sys, "argv", [str(path)])
    assert namespace["parse_args"]().device == "auto"
    monkeypatch.setattr(sys, "argv", [str(path), "--device", "cuda:3"])
    assert namespace["parse_args"]().device == "cuda:3"


@pytest.mark.parametrize(
    "path",
    [
        FULLSPACE,
        ARCHITECTURE,
        *(
            MODALITY / directory / filename
            for directory in SC_DIRECTORIES
            for filename in ("train_single_cell.py", "single_cell_model.py")
        ),
    ],
)
def test_entrypoints_validate_explicit_device_before_loading_data(path):
    requests = []

    def unavailable(requested):
        requests.append(requested)
        raise RuntimeError("explicit CUDA unavailable")

    namespace = functions_from(
        path,
        ("main",),
        parse_args=lambda: SimpleNamespace(device="cuda:3"),
        resolve_device=unavailable,
    )
    with pytest.raises(RuntimeError, match="explicit CUDA unavailable"):
        namespace["main"]()
    assert requests == ["cuda:3"]


@pytest.mark.parametrize(
    "path,function_name",
    [
        (FULLSPACE, "evaluate_holdout_percent"),
        (ARCHITECTURE, "evaluate_holdout_top5"),
    ],
)
def test_full_space_evaluation_progress_counts_partial_last_batch(path, function_name):
    progress = []
    namespace = functions_from(
        path,
        (function_name,),
        report_progress=lambda current, total, **kwargs: progress.append(
            (current, total)
        ),
    )

    class Model:
        target_full_anchor = torch.zeros(20, 2)

        def eval(self):
            pass

        def logits_full(self, x):
            return x

    bundle = {
        "x": np.tile(np.arange(20.0, 0.0, -1.0), (3, 1)),
        "true_indices": [[0], [1], [0]],
    }
    kwargs = {"percent": 0.05} if function_name == "evaluate_holdout_percent" else {}
    metrics, rows = namespace[function_name](
        Model(), bundle, torch.device("cpu"), batch_size=2, **kwargs
    )
    assert progress == [(2, 3), (3, 3)]
    assert metrics["top5pct_success"] == pytest.approx(2 / 3)
    assert metrics["n_profiles"] == 3
    assert rows is None


def test_single_cell_training_reports_completed_updates_and_keeps_selected_outputs(
    tmp_path,
):
    progress = []
    reported_metrics = []

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 20)

        def forward_logits_batch(self, x, targets):
            return self.linear(x.mean(dim=1))

    def batch(adata, drugs, cells, labels, rng, device):
        x = torch.tensor(rng.normal(size=(2, 2, 2)), dtype=torch.float32)
        y = torch.zeros(2, 20)
        y[:, 0] = 1.0
        return x, y, None

    def evaluate(*args):
        split = args[-2]
        return {
            f"{split}_Hit@1": 0.5,
            f"{split}_Hit@5": 0.5,
            f"{split}_Hit@10": 0.5,
            f"{split}_n_total": 1,
            f"{split}_n_eval": 1,
            f"{split}_n_skipped": 0,
        }, []

    namespace = functions_from(
        MODALITY / "scripts" / "single_cell_model.py",
        ("train_one",),
        time=time,
        set_seed=torch.manual_seed,
        build_model=lambda *args, **kwargs: TinyModel(),
        LR=3e-4,
        WEIGHT_DECAY=1e-5,
        MAX_EPOCHS=1,
        STEPS_PER_EPOCH=2,
        BATCH_SIZE=2,
        MIN_EPOCHS_BEFORE_STOP=4,
        EARLY_STOP_PATIENCE=4,
        KS=[1, 5, 10],
        TARGET_EMBEDDING_PATH=tmp_path / "targets.csv",
        PROTOCOL_JSON=tmp_path / "protocol.json",
        sample_batch=batch,
        compute_adapter_loss=lambda ctx, logits, labels, *args: (
            torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        ),
        eval_seen_hitk=evaluate,
        eval_holdout_all_percentiles=lambda *args: ({"holdout_all_top5pct": 0.5}, []),
        report_progress=lambda current, total, **kwargs: progress.append(
            (current, total, kwargs)
        ),
        report_step=lambda *args: None,
        report_metrics=lambda metrics, **kwargs: reported_metrics.append(
            (metrics, kwargs["label"])
        ),
    )
    ctx = SimpleNamespace(
        device=torch.device("cpu"),
        adata=SimpleNamespace(n_vars=2),
        target_df=pd.DataFrame(np.zeros((20, 2))),
        cell_line="A549",
        available_fit_drugs=["drug"],
        cell_pool_by_drug={},
        y_seen_by_drug={},
        w_seen=None,
        val_drugs=["drug"],
        test_drugs=["drug"],
        eval_cells_by_drug={},
        drug2targets={},
        seen_name2idx={},
        gene_to_global={},
        exclusive_drugs=[],
        mixed_drugs=[],
        holdout_idx_by_drug={},
        w_all=None,
        adata_path=tmp_path / "input.h5ad",
        seen_targets=[f"G{i}" for i in range(20)],
        qa={},
    )
    run_dir = tmp_path / "run"
    result = namespace["train_one"](ctx, "current_bce_baseline", 2041, run_dir)
    assert [(current, total) for current, total, _ in progress] == [
        (1, 2),
        (2, 2),
        (1, 1),
    ]
    assert all(isinstance(event[2]["metrics"]["loss"], float) for event in progress[:2])
    assert result["best_epoch"] == 1
    assert [label.rsplit(" selected ", 1)[1] for _, label in reported_metrics] == [
        "validation metrics",
        "test metrics",
        "holdout metrics",
    ]
    assert reported_metrics[0][0]["val_Hit@10"] == result["val_metrics"]["val_Hit@10"]
    assert (
        reported_metrics[1][0]["test_Hit@10"] == result["test_metrics"]["test_Hit@10"]
    )
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))[
        "history"
    ]
    assert len(history) == 1
    assert history[0]["train_loss"] >= 0
    assert (run_dir / "checkpoint.pth").is_file()


def test_proteomics_cpu_smoke_writes_all_output_parents_and_progress(tmp_path):
    """A tiny real optimization/evaluation run checks the standalone output contract."""
    rng = np.random.default_rng(32)
    genes = [f"G{i}" for i in range(72)]
    rows = [
        {
            "drug": f"drug{i}",
            "target": repr([genes[i], genes[(i + 1) % 72], genes[(i + 9) % 72]]),
            **{f"feature{j}": float(rng.uniform(0.1, 2.0)) for j in range(4)},
        }
        for i in range(72)
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "proteomicsdb_data.csv")
    target_path = tmp_path / "targets.csv"
    pd.DataFrame(rng.normal(size=(72, 4)), index=genes).to_csv(target_path)
    outputs = {
        "OUT_PATH": "result.json",
        "SAVE_CKPT_PATH": "checkpoints/selected.pth",
        "PER_DRUG_PATH": "per_drug/results.tsv",
        "HISTORY_TSV_PATH": "history/training.tsv",
        "METHOD_LOCK_PATH": "method/lock.json",
        "METRIC_SUMMARY_PATH": "metrics/summary.tsv",
        "LEAKAGE_AUDIT_PATH": "audits/leakage.md",
        "SELECTED_RESULT_PATH": "selection/result.json",
    }
    env = {
        **os.environ,
        **outputs,
        "PYTHONPATH": str(REPO / "src"),
        "PT_DATA_DIR": str(tmp_path),
        "LABEL_REPR_PATH": str(target_path),
        "DEVICE": "cpu",
        "EPOCHS": "1",
        "BATCH_SIZE": "16",
        "PROJ_DIM": "8",
        "N_RES": "0",
        "DROPOUT": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "METAR_PROGRESS": "1",
    }
    completed = subprocess.run(
        [sys.executable, str(MODALITY / "scripts" / "train_proteomics.py")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert all((tmp_path / path).is_file() for path in outputs.values())
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["best_epoch"] == 1
    assert len(result["history"]) == 1
    assert result["config"]["epochs"] == 1
    output = completed.stdout + completed.stderr
    assert "proteomics epoch 1 batches" in output
    assert "proteomics epochs" in output
    assert "proteomics selected test metrics" in output
    checkpoint = torch.load(
        tmp_path / outputs["SAVE_CKPT_PATH"], map_location="cpu", weights_only=True
    )
    assert all(
        tensor.device.type == "cpu"
        for tensor in checkpoint["model_state_dict"].values()
    )
