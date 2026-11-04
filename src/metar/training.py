"""Safe orchestration of the selected article training stages."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .runtime import report_step, resolve_device
from .training_workspace import MARKER, ROUND_NAMES, assert_private_output


@dataclass(frozen=True)
class TrainingStage:
    name: str
    description: str
    target: str
    arguments: tuple[str, ...]


ROUND_ROOT = "compat/home/lwn/claude/manuscript/write"
ROUND259 = f"{ROUND_ROOT}/claude_round259_crosscell_fullspace_round252query_then_sc_proteomics_20260522"
STAGES: dict[str, TrainingStage] = {
    "gene-prior": TrainingStage(
        "gene-prior",
        "Round210 gene-relation-only prior pretraining",
        f"{ROUND_ROOT}/claude_round210_dim256_checkpoint_sweep_20260515/scripts/run_round210_dim256_checkpoint_sweep.py",
        (
            "--graph-ids",
            "process_only",
            "--flow-ids",
            "r210_a02_reladd2_d256_f16_t020_edge050",
            "--readout-ids",
            "rbase_hit10",
        ),
    ),
    "closed-bulk": TrainingStage(
        "closed-bulk",
        "Round246 joint 668-target bulk model",
        f"{ROUND_ROOT}/claude_round246_joint_seen_target_train_pc3_a375_mcf7_20260519/scripts/run_round246_joint_seen_target_train_pc3_a375_mcf7.py",
        (
            "--phase",
            "joint",
            "--lambda-meta",
            "0.5",
            "--shift-scale",
            "0.15",
            "--max-delta-norm",
            "2.5",
        ),
    ),
    "support-initializer": TrainingStage(
        "support-initializer",
        "Round247 query-gated adapter initialization used by Round252",
        f"{ROUND_ROOT}/claude_round247_support_curve_query_gated_adapter_20260520/scripts/"
        "run_round247_support_curve_query_gated_adapter.py",
        ("--phase", "train"),
    ),
    "cross-cell-adapter": TrainingStage(
        "cross-cell-adapter",
        "Round252 selected EpiQGate support adapter",
        f"{ROUND_ROOT}/claude_round252_epiqgate_pseudoval_sweep_20260520/scripts/run_round252_epiqgate_pseudoval_sweep.py",
        (
            "--phase",
            "train",
            "--init-adapter",
            "round247",
            "--adapter-hidden-dim",
            "128",
            "--adapter-dropout",
            "0.1",
            "--max-delta-norm",
            "2.5",
            "--shift-scale",
            "0.18",
            "--base-anchor-weight",
            "0.1",
            "--delta-l2",
            "0.001",
            "--gate-l2",
            "0.001",
            "--train-support-size",
            "32",
            "--train-query-size",
            "32",
            "--eval-support-size",
            "64",
            "--tasks-per-epoch",
            "240",
            "--max-epochs",
            "32",
            "--min-epochs",
            "8",
            "--patience",
            "7",
            "--adapter-lr",
            "0.00015",
            "--adapter-weight-decay",
            "0.00001",
            "--lambda-episode",
            "1.0",
            "--run-seed",
            "20260524",
        ),
    ),
    "full-space": TrainingStage(
        "full-space",
        "Round259 selected 16,583-gene branch",
        f"{ROUND259}/scripts/run_round259_crosscell_fullspace_round252query.py",
        ("--runs", "e13_bce001"),
    ),
    "proteomics": TrainingStage(
        "proteomics",
        "Round261 selected proteomics fitting",
        "analysis/applications/proteomics/fit_proteomics.py",
        (),
    ),
}

SINGLE_CELL_SELECTIONS = {
    "A549": ("drug_normalized_bce", 2043, "train_A549"),
    "SW480": ("label_bag_retrieval", 2041, "train_SW480"),
    "NCI-H460": ("drug_normalized_bce", 2042, "train_NCI_H460"),
}

LOCKED_OUTPUTS = {
    "gene-prior": (
        f"{ROUND_ROOT}/claude_round210_dim256_checkpoint_sweep_20260515/checkpoints/process_only/"
        "r210_a02_reladd2_d256_f16_t020_edge050/"
        "r210_a02_reladd2_d256_f16_t020_edge050_kg_checkpoint_epoch030.pth"
    ),
    "closed-bulk": (
        f"{ROUND_ROOT}/claude_round246_joint_seen_target_train_pc3_a375_mcf7_20260519/checkpoints/"
        "r246_joint_lm0p5_sc0p15_mn2p5/whole_model_checkpoint.pth"
    ),
    "support-initializer": (
        f"{ROUND_ROOT}/claude_round247_support_curve_query_gated_adapter_20260520/checkpoints/"
        "r247_query_gated_h128_sc0p15_mn2p5/adapter_checkpoint.pth"
    ),
    "cross-cell-adapter": (
        f"{ROUND_ROOT}/claude_round252_epiqgate_pseudoval_sweep_20260520/checkpoints/"
        "r252_epiqgate_lm1_lr0p00015_sc0p18_mn2p5_ba0p1_s32_q32_diverse_label_seed20260524/"
        "adapter_checkpoint.pth"
    ),
    "full-space": f"{ROUND259}/checkpoints/e13_bce001/selected_checkpoint.pth",
}


def archived_training_command(
    stage_name: str,
    workspace: str | Path,
    *,
    device: str,
    output_dir: str | Path | None = None,
    cell_line: str | None = None,
) -> list[str]:
    root = Path(workspace).resolve()
    runner = root / "scripts/run_archived_script.py"
    if stage_name == "single-cell":
        if cell_line not in SINGLE_CELL_SELECTIONS:
            raise ValueError(
                f"single-cell stage requires one of {sorted(SINGLE_CELL_SELECTIONS)}"
            )
        loss_mode, seed, directory = SINGLE_CELL_SELECTIONS[cell_line]
        target = f"{ROUND259}/{directory}/run_round260_sc_round259_embedding.py"
        output = (
            Path(output_dir).resolve()
            if output_dir
            else root / "training_runs" / f"single_cell_{cell_line}"
        )
        arguments = (
            "--cell-line",
            cell_line,
            "--out-dir",
            str(output),
            "--device",
            device,
            "--loss-mode",
            loss_mode,
            "--seed",
            str(seed),
        )
    else:
        if stage_name not in STAGES:
            raise ValueError(f"unknown training stage: {stage_name}")
        stage = STAGES[stage_name]
        target = stage.target
        arguments = stage.arguments
        if stage_name in {
            "gene-prior",
            "closed-bulk",
            "support-initializer",
            "cross-cell-adapter",
            "full-space",
        }:
            arguments = (*arguments, "--device", device)
        elif stage_name == "proteomics":
            output = (
                Path(output_dir).resolve()
                if output_dir
                else root / "training_runs" / "proteomics"
            )
            return [
                sys.executable,
                str(root / target),
                "--package-root",
                str(root),
                "--output-dir",
                str(output),
                "--device",
                device,
            ]
    return [sys.executable, str(runner), target, *arguments]


def validate_training_workspace(
    workspace: str | Path,
    target: list[str],
    stage_name: str,
) -> None:
    root = Path(workspace).resolve()
    if not (root / MARKER).is_file():
        raise RuntimeError(
            "training requires an isolated workspace created by "
            "scripts/prepare_training.py; a pretrained compatibility tree is not a training workspace"
        )
    missing = [
        relative
        for relative in stage_inputs(stage_name, target)
        if not (root / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "missing training stage dependencies:\n" + "\n".join(missing)
        )
    # Inspect directories before descending, including Windows junctions whose
    # is_symlink() can be false. Never traverse a redirected external tree.
    mutable = [root / ROUND_ROOT / name for name in ROUND_NAMES.values()]
    mutable.extend(
        (
            root / ROUND_ROOT / "bulk_sharedsample_indunseenlabel_20260330_h07",
            root / "checkpoints/selected",
        )
    )
    for directory in mutable:
        pending = [directory]
        while pending:
            path = pending.pop()
            assert_private_output(root, path)
            if path.is_dir():
                pending.extend(path.iterdir())
    # These directories contain external read-only input FILE links. Check the
    # directories and generated outputs, without rejecting deliberate inputs.
    for relative in (
        "data/processed/fullspace",
        "data/processed/proteomics",
        "data/processed/fullspace/round259_target_embedding.csv",
        "data/processed/fullspace/round259_target_embedding_aligned_to_proteomics_gene_order.csv",
        "data/processed/proteomics/round261_preprocessor.json",
    ):
        assert_private_output(root, root / relative)
    locked_output = LOCKED_OUTPUTS.get(stage_name)
    if locked_output and (root / locked_output).exists():
        raise RuntimeError(
            "the selected checkpoint output already exists in this workspace. "
            "Use scripts/prepare_training.py to create a fresh workspace for this stage. "
            f"Existing output for {stage_name!r}: {locked_output}"
        )


def training_command(
    stage_name: str,
    workspace: str | Path,
    *,
    device: str,
    output_dir: str | Path | None = None,
    cell_line: str | None = None,
) -> list[str]:
    # Validate the stage/cell arguments even for a command-only dry run.
    archived_training_command(
        stage_name, workspace, device=device, output_dir=output_dir, cell_line=cell_line
    )
    command = [
        sys.executable,
        "-m",
        "metar.training_runtime",
        "--stage",
        stage_name,
        "--workspace",
        str(Path(workspace).resolve()),
        "--device",
        device,
    ]
    if output_dir:
        command.extend(("--output-dir", str(output_dir)))
    if cell_line:
        command.extend(("--cell-line", cell_line))
    return command


def stage_inputs(stage_name: str, command: list[str]) -> tuple[str, ...]:
    """File dependencies, including outputs generated by preceding stages."""
    bulk = (
        "data/processed/lincs/large_data.parquet",
        "data/processed/lincs/compoundinfo_beta.txt",
        "data/processed/fullspace/pertkge_ppi_graph.csv",
        "data/processed/lincs/bulk_protocol_lock_v2.json",
        "data/processed/fullspace/splits_v3.json",
    )
    if stage_name == "gene-prior":
        return (
            "data/public_raw/gene_prior/process.txt",
            "data/processed/closed/r210/pc3_landmark_response_978.tsv.gz",
            "data/processed/closed/r210/pc3_metar_drugcell_scores_668.tsv.gz",
            STAGES[stage_name].target,
        )
    if stage_name in {"closed-bulk", "support-initializer", "cross-cell-adapter"}:
        prior = (LOCKED_OUTPUTS["gene-prior"],)
        upstream = (
            () if stage_name == "closed-bulk" else (LOCKED_OUTPUTS["closed-bulk"],)
        )
        if stage_name == "cross-cell-adapter":
            upstream += (LOCKED_OUTPUTS["support-initializer"],)
        if stage_name == "support-initializer":
            upstream += (
                f"{ROUND_ROOT}/{ROUND_NAMES['r246']}/source_data/selected_r246_joint_lm0p5_sc0p15_mn2p5.json",
            )
        return (*bulk, *prior, *upstream, STAGES[stage_name].target)
    if stage_name == "full-space":
        return (
            "data/processed/fullspace/h07_prepared_data.pt",
            LOCKED_OUTPUTS["closed-bulk"],
            f"{ROUND_ROOT}/{ROUND_NAMES['r252']}/source_data/cell_splits.json",
            STAGES[stage_name].target,
        )
    if stage_name == "single-cell":
        cell = command[command.index("--cell-line") + 1]
        name = {"A549": "a549", "SW480": "sw480", "NCI-H460": "nci_h460"}[cell]
        directory = SINGLE_CELL_SELECTIONS[cell][2]
        return (
            "data/processed/fullspace/round259_target_embedding.csv",
            "data/processed/tahoe/sc_protocol_lock_v3_20260326.json",
            "data/processed/tahoe/tahoe_dti.csv",
            f"data/processed/tahoe/tahoe_{name}.h5ad",
            f"{ROUND259}/{directory}/run_round260_sc_round259_embedding.py",
            f"{ROUND259}/{directory}/run_sc_label_bag_calibration_multicell_base.py",
            f"{ROUND259}/source_data/round259_target_embedding.csv",
        )
    return (
        "data/processed/proteomics/proteomicsdb_data.csv",
        "data/processed/proteomics/gene_order.tsv",
        "data/processed/fullspace/round259_target_embedding.csv",
        "code/original_runs/r259_261/proteomics_results/round261_proteomics_round259_embedding_method_lock.json",
        f"{ROUND259}/scripts/run_round261_proteomics_round259_embedding.py",
    )


def check_training_dependencies(stage_name: str) -> None:
    """Check optional packages before allocating a model or starting a child."""
    packages = {
        "gene-prior": ("scipy", "sklearn"),
        "closed-bulk": ("scipy", "sklearn", "pyarrow"),
        "support-initializer": ("scipy", "sklearn", "pyarrow"),
        "cross-cell-adapter": ("scipy", "sklearn", "pyarrow"),
        "full-space": ("scipy", "sklearn"),
        "single-cell": ("scipy", "sklearn", "anndata"),
        "proteomics": ("scipy", "sklearn"),
    }
    missing = [
        name for name in packages[stage_name] if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise ImportError(
            "Missing training dependencies: "
            + ", ".join(missing)
            + f'. In this environment run: {sys.executable} -m pip install -e ".[training]"'
        )


def run_training_stage(
    stage_name: str,
    workspace: str | Path,
    *,
    device: str,
    output_dir: str | Path | None = None,
    cell_line: str | None = None,
    dry_run: bool = False,
    check_only: bool = False,
) -> int:
    command = training_command(
        stage_name,
        workspace,
        device=device,
        output_dir=output_dir,
        cell_line=cell_line,
    )
    if dry_run:
        print(
            json.dumps(
                {
                    "stage": stage_name,
                    "command": command,
                    "archived_command": archived_training_command(
                        stage_name,
                        workspace,
                        device=device,
                        output_dir=output_dir,
                        cell_line=cell_line,
                    ),
                    "validation": "command preview only",
                },
                indent=2,
            )
        )
        return 0
    if check_only:
        root = Path(workspace).resolve()
        missing = [
            path
            for path in stage_inputs(stage_name, command)
            if not (root / path).is_file()
        ]
        prepared = (root / MARKER).is_file()
        workspace_error = None
        if prepared and not missing:
            try:
                validate_training_workspace(root, command, stage_name)
            except (FileNotFoundError, RuntimeError) as exc:
                workspace_error = str(exc)
        print(
            json.dumps(
                {
                    "stage": stage_name,
                    "prepared_workspace": prepared,
                    "missing_inputs": missing,
                    "workspace_error": workspace_error,
                    "validation": "file presence only; no model loading or training",
                },
                indent=2,
            )
        )
        return 0 if prepared and not missing and workspace_error is None else 2
    report_step(f"Check {stage_name} workspace, dependencies and device")
    validate_training_workspace(workspace, command, stage_name)
    check_training_dependencies(stage_name)
    resolved = resolve_device(device)
    command[command.index("--device") + 1] = resolved
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    report_step(f"Run {stage_name} on {resolved}")
    completed = subprocess.run(
        command, cwd=Path(workspace).resolve(), check=False, env=environment
    )
    return int(completed.returncode)
