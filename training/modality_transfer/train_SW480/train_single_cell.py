#!/usr/bin/env python3
"""Train a single-cell response encoder against the fixed genome-wide target table."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from metar.runtime import report_metrics, report_progress, report_step, resolve_device


SCRIPT_DIR = Path(__file__).resolve().parent
ROUND_ROOT = SCRIPT_DIR.parent
ROUND259_TARGET_CSV = ROUND_ROOT / "source_data" / "round259_target_embedding.csv"
REMOTE_BASE_SCRIPT = SCRIPT_DIR / "run_sc_label_bag_calibration_multicell_base.py"
DEFAULT_OUT_DIR = ROUND_ROOT
DEFAULT_DEVICE = "auto"
CELL_LINE_TO_ADATA = {
    "A549": Path("/home/lwn/claude/sc/data/tahoe_a549.h5ad"),
    "SW480": Path("/NFS2_home/NFS2_home_1/lwn/claude/sc/data/tahoe_sw480.h5ad"),
    "NCI-H460": Path("/NFS2_home/NFS2_home_1/lwn/claude/sc/data/tahoe_nci_h460.h5ad"),
}


def load_base_module():
    base_script = REMOTE_BASE_SCRIPT
    if not base_script.exists():
        base_script = SCRIPT_DIR / "single_cell_model.py"
    if not base_script.exists():
        raise RuntimeError(f"Cannot find base single-cell runner: {base_script}")
    spec = importlib.util.spec_from_file_location("round178_sc_base", base_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base single-cell runner from {base_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, base_script


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata-path", default=None)
    parser.add_argument("--cell-line", required=True)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--loss-mode", default="all")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def copy_self_and_base(out_dir: Path, base_script: Path) -> None:
    dst_self = out_dir / "run_round260_sc_round259_embedding.py"
    if Path(__file__).resolve() != dst_self.resolve():
        shutil.copy2(Path(__file__).resolve(), dst_self)
    dst_base = out_dir / "run_sc_label_bag_calibration_multicell_base.py"
    if base_script.resolve() != dst_base.resolve():
        shutil.copy2(base_script, dst_base)


def write_round260_report(out_dir: Path, cell_line: str, selected_result: dict[str, Any]) -> None:
    lines = [
        f"# {cell_line} single-cell target prediction",
        "",
        "## Selection rule",
        "",
        "Primary selection used validation only: `val_Hit@10`, then `val_Hit@5`, then `val_Hit@20`, then smaller parameter count.",
        "",
        "## Selected primary",
        "",
        f"- Loss mode: `{selected_result['loss_mode']}`",
        f"- Seed: `{selected_result['seed']}`",
        f"- Val Hit@10: {selected_result['val_metrics']['val_Hit@10']:.4f}",
        f"- Test Hit@5: {selected_result['test_metrics']['test_Hit@5']:.4f}",
        f"- Test Hit@10: {selected_result['test_metrics']['test_Hit@10']:.4f}",
        f"- Test Hit@20: {selected_result['test_metrics']['test_Hit@20']:.4f}",
        f"- Holdout top-5%: {selected_result['holdout_metrics']['holdout_all_top5pct']:.4f}",
        f"- Holdout top-10%: {selected_result['holdout_metrics']['holdout_all_top10pct']:.4f}",
        "",
        "## Locked inputs",
        "",
        f"- Round259 target embedding: `{ROUND259_TARGET_CSV}`",
        f"- Adata: `{selected_result['adata_path']}`",
        f"- Target universe size: `{selected_result['gene_universe_size']}`",
        "",
        "## Interpretation",
        "",
        "The single-cell model uses the exported Round259 target-embedding table with the locked pipeline, head, split and evaluation protocol.",
    ]
    (out_dir / f"round260_{cell_line.lower().replace('-', '_')}_sc_label_bag_calibration_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_leakage_audit(out_dir: Path, cell_line: str, ctx, selected_result: dict[str, Any]) -> None:
    lines = [
        "# Leakage and denominator audit",
        "",
        f"- Round: `Round260`",
        f"- Cell line: `{cell_line}`",
        f"- Adata path: `{selected_result['adata_path']}`",
        f"- Adata shape: `{ctx.qa['adata_shape'][0]} x {ctx.qa['adata_shape'][1]}`",
        f"- Target embedding path: `{ROUND259_TARGET_CSV}`",
        f"- Target embedding shape: `{ctx.qa['target_universe_size']} x {ctx.qa['target_embedding_dim']}`",
        f"- Gene universe size: `{selected_result['gene_universe_size']}`",
        f"- Seen target count: `{selected_result['n_seen_targets']}`",
        "",
        "## Leakage checks",
        "",
        "- The target table is the exported Round259 embedding table only.",
        "- The single-cell head, split, loss and evaluation code are inherited from the locked Round178 pipeline.",
        "- Validation is used for selection; holdout metrics are not used for model choice.",
    ]
    (out_dir / "leakage_and_denominator_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_per_drug_metrics(out_dir: Path, run_dir: Path) -> None:
    import pandas as pd

    frames = []
    seen_path = run_dir / "seen_per_drug.tsv"
    holdout_path = run_dir / "holdout_per_drug.tsv"
    if seen_path.exists():
        seen = pd.read_csv(seen_path, sep="\t")
        seen.insert(0, "metric_family", "seen")
        frames.append(seen)
    if holdout_path.exists():
        holdout = pd.read_csv(holdout_path, sep="\t")
        holdout.insert(0, "metric_family", "holdout")
        frames.append(holdout)
    if frames:
        pd.concat(frames, ignore_index=True, sort=False).to_csv(out_dir / "per_drug_metrics.tsv", sep="\t", index=False)


def write_training_history(out_dir: Path, run_dir: Path, loss_mode: str, seed: int) -> None:
    history_path = run_dir / "history.json"
    if not history_path.exists():
        return
    import pandas as pd

    history = json.loads(history_path.read_text(encoding="utf-8")).get("history", [])
    rows = []
    for row in history:
        item = {"loss_mode": loss_mode, "seed": seed}
        item.update(row)
        rows.append(item)
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "training_history.tsv", sep="\t", index=False)


def main() -> None:
    args = parse_args()
    device = torch.device(resolve_device(args.device))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_step(f"Loading single-cell runner for {args.cell_line}; device={device}")
    base, base_script = load_base_module()

    if not ROUND259_TARGET_CSV.exists():
        raise RuntimeError(f"Missing Round259 target embedding: {ROUND259_TARGET_CSV}")

    target_embedding_path = ROUND259_TARGET_CSV
    base.TARGET_EMBEDDING_PATH = target_embedding_path
    if hasattr(base, "DEFAULT_OUT_DIR"):
        base.DEFAULT_OUT_DIR = out_dir

    cell_out_dir = out_dir / f"train_{args.cell_line.replace('-', '_')}"
    cell_out_dir.mkdir(parents=True, exist_ok=True)
    copy_self_and_base(cell_out_dir, base_script)

    print(f"device={device}", flush=True)
    print(f"out_dir={out_dir}", flush=True)
    print(f"cell_out_dir={cell_out_dir}", flush=True)
    adata_path = Path(args.adata_path) if args.adata_path else CELL_LINE_TO_ADATA.get(args.cell_line)
    if adata_path is None:
        raise RuntimeError(f"No default adata mapping for cell line {args.cell_line}; pass --adata-path explicitly")
    print(f"adata_path={adata_path}", flush=True)
    print(f"target_embedding_path={target_embedding_path}", flush=True)

    report_step("Preparing single-cell target embeddings, locked protocol, cells and labels")
    ctx = base.SweepContext(device=device, out_dir=cell_out_dir, adata_path=adata_path, cell_line=args.cell_line)
    ctx.write_global_lock()
    print("target/protocol lock ok", json.dumps(ctx.qa, sort_keys=True), flush=True)

    loss_modes = base.LOSS_MODES if args.loss_mode == "all" else [args.loss_mode]
    seeds = base.SEEDS if args.seed < 0 else [args.seed]
    rows: list[dict[str, Any]] = []

    for loss_mode in loss_modes:
        for seed in seeds:
            run_dir = cell_out_dir / f"{loss_mode}__seed{seed}"
            result_path = run_dir / "result.json"
            if result_path.exists() and not args.force:
                print(f"SKIP existing {run_dir}", flush=True)
                result = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                print(f"START loss_mode={loss_mode} seed={seed}", flush=True)
                result = base.train_one(ctx, loss_mode, seed, run_dir)
                print("RESULT", json.dumps(base.flatten_result(result), sort_keys=True), flush=True)
            rows.append(base.flatten_result(result))
            report_progress(len(rows), len(loss_modes) * len(seeds), label=f"single-cell {args.cell_line} runs")

    report_step("Writing single-cell selection, training history and metric summaries")
    base.write_summaries(cell_out_dir, rows, ctx)
    selected = json.loads((cell_out_dir / "selected_primary_result.json").read_text(encoding="utf-8"))
    for split, key in (("validation", "val_metrics"), ("test", "test_metrics"), ("holdout", "holdout_metrics")):
        report_metrics(
            {name: value for name, value in selected[key].items()
             if name.endswith(("Hit@5", "Hit@10", "Hit@20", "top5pct", "top10pct"))},
            label=f"single-cell {args.cell_line} selected {split} metrics",
        )
    write_round260_report(cell_out_dir, args.cell_line, selected)
    write_leakage_audit(cell_out_dir, args.cell_line, ctx, selected)

    metric_summary = cell_out_dir / "metric_summary.tsv"
    sweep_summary = cell_out_dir / "sweep_summary.tsv"
    if sweep_summary.exists():
        shutil.copy2(sweep_summary, metric_summary)

    source_data_dir = cell_out_dir / "source_data"
    source_data_dir.mkdir(exist_ok=True)
    for filename in ["sweep_summary.tsv", "metric_summary.tsv", "loss_mode_mean_std.tsv", "selected_primary_result.json", "method_lock.json", "qa_target_space_check.json", "target_burden_stratified_recovery.tsv"]:
        src = cell_out_dir / filename
        if src.exists():
            shutil.copy2(src, source_data_dir / filename)
    export_audit = out_dir / "source_data" / "round259_target_embedding_audit.json"
    if export_audit.exists():
        shutil.copy2(export_audit, source_data_dir / "target_embedding_audit.json")
    selected_run_dir = cell_out_dir / f"{selected['loss_mode']}__seed{selected['seed']}"
    write_training_history(source_data_dir, selected_run_dir, selected["loss_mode"], int(selected["seed"]))
    write_per_drug_metrics(source_data_dir, selected_run_dir)

    print(f"saved sweep summary to {sweep_summary}", flush=True)
    print(f"saved selected primary to {cell_out_dir / 'selected_primary_result.json'}", flush=True)


if __name__ == "__main__":
    main()
