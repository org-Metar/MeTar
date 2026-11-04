"""Execute the selected historical stages in a prepared, isolated workspace."""

from __future__ import annotations

import argparse
import csv
import functools
import json
import os
import shutil
import sys
from pathlib import Path

from .runtime import RunReporter, report_metrics, report_step, resolve_device
from .training import (
    LOCKED_OUTPUTS,
    ROUND259,
    SINGLE_CELL_SELECTIONS,
    archived_training_command,
    check_training_dependencies,
    training_command,
    validate_training_workspace,
)
from .training_workspace import WRITE_ROOT, assert_private_output

SELECTED_NAMES = {
    "gene-prior": "round210_process_only_epoch030.pth",
    "closed-bulk": "round246_whole_model_checkpoint.pth",
    "support-initializer": "round247_adapter_init.pth",
    "cross-cell-adapter": "round252_adapter_seed20260524.pth",
    "full-space": "round259_e13_bce001.pth",
    "proteomics": "round261_proteomics_epoch070.pth",
}


def _private_output(root: Path, output_dir: str | None, default: str) -> Path:
    output = Path(output_dir) if output_dir else Path("training_runs") / default
    if not output.is_absolute():
        output = root / output
    if not output.resolve().is_relative_to(root):
        raise ValueError("training output must stay inside the prepared workspace")
    assert_private_output(root, output)
    return output.resolve()


def _copy_generated(source: Path, destination: Path, *, root: Path) -> None:
    assert_private_output(root, destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace generated output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def report_saved_metrics(root: Path, stage: str) -> None:
    """Print selected existing result rows; never run an additional evaluation."""
    directories = {
        "closed-bulk": "r246",
        "support-initializer": "r247",
        "cross-cell-adapter": "r252",
        "full-space": "r259_261",
    }
    if stage not in directories:
        return
    from .training_workspace import ROUND_NAMES

    path = (
        root
        / WRITE_ROOT
        / ROUND_NAMES[directories[stage]]
        / "source_data/metric_summary.tsv"
    )
    if not path.is_file():
        return
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if stage == "full-space":
                if row.get("variant") != "e13_bce001" or row.get("split") not in {
                    "test",
                    "holdout_test",
                }:
                    continue
                if row.get("evaluation_family", "").endswith("_remove_seen"):
                    continue
            else:
                if (
                    row.get("split") != "official_test"
                    or row.get("evaluation_unit") != "drug-mean"
                ):
                    continue
                if (
                    stage == "closed-bulk"
                    and row.get("evaluation_family") != "seen_668_no_adapt"
                ):
                    continue
                if (
                    stage == "support-initializer"
                    and row.get("source") != "Round247_query_gated"
                ):
                    continue
                if (
                    stage == "cross-cell-adapter"
                    and row.get("source") != "Round252_EpiQGate"
                ):
                    continue
                if stage != "closed-bulk" and (
                    row.get("selector") != "diverse_label"
                    or float(
                        row.get("requested_support_size")
                        or row.get("adapt_support_size")
                        or -1
                    )
                    != 64
                ):
                    continue
            metrics = {
                name: row[name]
                for name in (
                    "Hit@10",
                    "MeanAP",
                    "AUROC",
                    "top5pct_success",
                    "top10pct_success",
                    "n_cells",
                    "n_groups",
                    "n_profiles",
                    "best_epoch",
                )
                if row.get(name) not in (None, "")
            }
            metrics = {name: float(value) for name, value in metrics.items()}
            label = f"Final {row['split']} | {row.get('evaluation_family', '')} | "
            label += "profile" if stage == "full-space" else "cell-macro, drug-mean"
            report_metrics(metrics, label=label)
    report_step(f"Full metric table: {path}")


def _run_source(script: Path, arguments: list[str], *, device: str) -> None:
    source = script.read_text(encoding="utf-8-sig")
    original_argv = sys.argv
    original_path = list(sys.path)
    sys.argv = [str(script), *arguments]
    sys.path.insert(0, str(script.parent))
    namespace = {"__file__": str(script), "__name__": "__main__", "__package__": None}
    # Sources are the explicit, copied first-party archive entrypoints.
    try:
        exec(compile(source, str(script), "exec"), namespace)  # noqa: S102
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    finally:
        sys.argv = original_argv
        sys.path[:] = original_path


def align_proteomics_targets(root: Path) -> Path:
    """Apply the historical aliases and locked gene order to the new R259 export."""
    import pandas as pd

    # Read only gene names from the release schema, never its trained vectors.
    order = pd.read_csv(root / "data/processed/proteomics/gene_order.tsv", sep="\t")[
        "gene"
    ].astype(str)
    table = pd.read_csv(
        root / "data/processed/fullspace/round259_target_embedding.csv", index_col=0
    )
    aliases = {
        "15-Sep": "SEPT15",
        "3-Sep": "SEPT3",
        "4-Sep": "SEPT4",
        "5-Sep": "SEPT5",
        "6-Sep": "SEPT6",
        "7-Sep": "SEPT7",
        "9-Sep": "SEPT9",
        "8-Mar": "MARCHF8",
    }
    table = table.rename(index=aliases)
    if table.index.duplicated().any() or not set(order).issubset(table.index):
        raise ValueError(
            "new Round259 target export does not match the proteomics gene order"
        )
    output = (
        root
        / "data/processed/fullspace/round259_target_embedding_aligned_to_proteomics_gene_order.csv"
    )
    assert_private_output(root, output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    table.loc[order].to_csv(output)
    return output


def _run_proteomics(root: Path, output: Path, device: str) -> Path:
    from .preprocessing import fit_round261_preprocessor

    report_step("Align generated target embeddings to the locked proteomics gene order")
    label_table = align_proteomics_targets(root)
    lock = json.loads(
        (
            root
            / "code/original_runs/r259_261/proteomics_results/round261_proteomics_round259_embedding_method_lock.json"
        ).read_text()
    )
    config = lock["config"]
    output.mkdir(parents=True, exist_ok=False)
    stem = output / "round261_proteomics_round259_embedding"
    env = {
        "PT_DATA_DIR": str(root / "data/processed/proteomics"),
        "LABEL_REPR_PATH": str(label_table),
        "OUT_PATH": str(stem.with_suffix(".json")),
        "SAVE_CKPT_PATH": str(stem.with_suffix(".pth")),
        "PER_DRUG_PATH": str(output / f"{stem.name}_per_drug.tsv"),
        "HISTORY_TSV_PATH": str(output / f"{stem.name}_training_history.tsv"),
        "METHOD_LOCK_PATH": str(output / f"{stem.name}_method_lock.json"),
        "METRIC_SUMMARY_PATH": str(output / f"{stem.name}_metric_summary.tsv"),
        "LEAKAGE_AUDIT_PATH": str(
            output / f"{stem.name}_leakage_and_denominator_audit.md"
        ),
        "SELECTED_RESULT_PATH": str(
            output / f"{stem.name}_selected_primary_result.json"
        ),
        "DEVICE": device,
        "INIT_SCALE": "10.0",
    }
    key_map = {
        "epochs": "EPOCHS",
        "eval_every": "EVAL_EVERY",
        "lr": "LR",
        "weight_decay": "WEIGHT_DECAY",
        "val_frac": "VAL_FRAC",
        "holdout_k_percent": "HOLDOUT_K_PERCENT",
        "seed": "SEED",
        "proj_dim": "PROJ_DIM",
        "n_res": "N_RES",
        "dropout": "DROPOUT",
        "batch_size": "BATCH_SIZE",
        "topk_feat": "TOPK_FEAT",
        "label_adapter_mode": "LABEL_ADAPTER_MODE",
        "label_bottleneck": "LABEL_BOTTLENECK",
        "adapter_l2": "ADAPTER_L2",
        "ortho_lambda": "ORTHO_LAMBDA",
    }
    env.update({name: str(config[key]) for key, name in key_map.items()})
    os.environ.update(env)
    _run_source(
        root / ROUND259 / "scripts/run_round261_proteomics_round259_embedding.py",
        [],
        device=device,
    )
    import pandas as pd

    targets = tuple(pd.read_csv(label_table, index_col=0).index.astype(str))
    report_step("Export training-derived proteomics normalization")
    preprocessor = fit_round261_preprocessor(
        root / "data/processed/proteomics/proteomicsdb_data.csv",
        targets,
        seed=int(config["seed"]),
        validation_fraction=float(config["val_frac"]),
        top_k_features=int(config["topk_feat"]),
    )
    preprocessor.write(root / "data/processed/proteomics/round261_preprocessor.json")
    return stem.with_suffix(".pth")


def _execute(args: argparse.Namespace) -> Path:
    """Execute one stage; callers own presentation and error reporting."""
    root = args.workspace.resolve()
    report_step("Check stage inputs and protected output paths")
    command = training_command(
        args.stage,
        root,
        device=args.device,
        output_dir=args.output_dir,
        cell_line=args.cell_line,
    )
    validate_training_workspace(root, command, args.stage)
    check_training_dependencies(args.stage)
    output = _private_output(
        root,
        args.output_dir,
        f"{args.stage}_{args.cell_line}" if args.cell_line else args.stage,
    )
    if args.stage in {"single-cell", "proteomics"} and output.exists():
        raise FileExistsError(f"refusing to reuse training output directory: {output}")
    import torch

    original_load = torch.load
    original_environment = dict(os.environ)
    # These first-party checkpoints contain NumPy metadata. This opt-in is
    # limited to the trusted training workspace and restored after execution.
    torch.load = functools.partial(original_load, weights_only=False)
    bulk = root / WRITE_ROOT / "bulk_sharedsample_indunseenlabel_20260330_h07"
    os.environ["METAR_BULK_PROJECT_ROOT"] = str(bulk)
    os.environ["METAR_BULK_PROTOCOL_PATH"] = str(
        root / "data/processed/lincs/bulk_protocol_lock_v2.json"
    )
    try:
        if args.stage == "proteomics":
            generated = _run_proteomics(root, output, args.device)
        else:
            archived = archived_training_command(
                args.stage,
                root,
                device=args.device,
                output_dir=output,
                cell_line=args.cell_line,
            )
            script = root / archived[2]
            report_step("Prepare data, train and evaluate the selected model")
            _run_source(script, archived[3:], device=args.device)
            if args.stage == "single-cell":
                loss, seed, _directory = SINGLE_CELL_SELECTIONS[args.cell_line]
                generated = (
                    output
                    / f"train_{args.cell_line.replace('-', '_')}"
                    / f"{loss}__seed{seed}/checkpoint.pth"
                )
            else:
                generated = root / LOCKED_OUTPUTS[args.stage]
            if args.stage == "full-space":
                _copy_generated(
                    root / ROUND259 / "source_data/round259_target_embedding.csv",
                    root / "data/processed/fullspace/round259_target_embedding.csv",
                    root=root,
                )
        selected_name = SELECTED_NAMES.get(args.stage)
        if args.stage == "single-cell":
            _loss, seed, _directory = SINGLE_CELL_SELECTIONS[args.cell_line]
            selected_name = (
                f"round260_{args.cell_line.replace('-', '_')}_seed{seed}.pth"
            )
        report_step("Export selected checkpoint")
        destination = root / "checkpoints/selected" / selected_name
        _copy_generated(generated, destination, root=root)
        report_saved_metrics(root, args.stage)
        return destination
    finally:
        torch.load = original_load
        os.environ.clear()
        os.environ.update(original_environment)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--cell-line")
    args = parser.parse_args()
    try:
        requested = args.device
        args.device = resolve_device(requested)
        with RunReporter(
            "Training",
            {
                "setting": args.stage,
                "cell_line": args.cell_line,
                "device": args.device,
                "requested_device": requested,
                "workspace": args.workspace.resolve(),
            },
        ) as report:
            checkpoint = _execute(args)
            report.finish(checkpoint=checkpoint)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
