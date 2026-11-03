#!/usr/bin/env python
"""Summarize L3/L4/L5 MeTar score tables under the current closed-set evaluator."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from common import add_code_release_to_path, code_release_root_from_config, read_table, read_yaml, write_table


def run_eval(code_root: Path, label_order: Path, score_args: list[str], output_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(code_root / "scripts" / "evaluate_closed_set.py"),
        "--label-order",
        str(label_order),
        "--group-cols",
        "drug",
        "cell_line",
        "--strict-key-cols",
        "drug",
        "cell_line",
        "--include-random",
        "--output-dir",
        str(output_dir),
    ]
    for item in score_args:
        cmd.extend(["--score-table", item])
    subprocess.run(cmd, check=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Summarize Round429 layer sensitivity results")
    parser.add_argument("--config", required=True)
    parser.add_argument("--label-order", required=True)
    parser.add_argument("--score-table", action="append", required=True, help="METHOD=PATH")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cfg = read_yaml(Path(args.config).resolve())
    code_root = add_code_release_to_path(code_release_root_from_config(cfg))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    eval_out = out / "closed_set_eval"
    run_eval(code_root, Path(args.label_order), args.score_table, eval_out)

    per = read_table(eval_out / "closed_set_per_unit_metrics.tsv")
    seed_summary = read_table(eval_out / "closed_set_metric_summary.tsv")
    method_re = re.compile(r"^MeTar_(L[345])_(direct|adapted)_seed(\d+)$")

    def parse_method(value: object) -> tuple[str, str, int | None]:
        text = str(value)
        match = method_re.match(text)
        if match:
            return match.group(1), match.group(2), int(match.group(3))
        return text, "fixed_reference", None

    parsed = seed_summary["method"].map(parse_method)
    seed_summary["layer"] = [x[0] for x in parsed]
    seed_summary["mode"] = [x[1] for x in parsed]
    seed_summary["seed"] = [x[2] for x in parsed]
    metric_cols = [c for c in ["Hit@1", "Hit@5", "Hit@10", "Hit@20", "Hit@30", "Hit@50", "AUROC", "NDCG@10", "MeanAP"] if c in seed_summary.columns]
    seed_methods = seed_summary[seed_summary["seed"].notna()].copy()
    fixed_methods = seed_summary[seed_summary["seed"].isna()].copy()
    summary_parts = []
    if len(seed_methods):
        agg = seed_methods.groupby(["layer", "mode"], dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
        agg.columns = ["_".join([str(x) for x in col if str(x)]) for col in agg.columns]
        agg = agg.rename(columns={"layer_": "layer", "mode_": "mode"})
        agg["method"] = "MeTar_" + agg["layer"].astype(str) + "_" + agg["mode"].astype(str)
        agg["n_seeds"] = seed_methods.groupby(["layer", "mode"], dropna=False)["seed"].nunique().to_numpy()
        summary_parts.append(agg)
    if len(fixed_methods):
        fixed = fixed_methods.copy()
        for col in metric_cols:
            fixed[f"{col}_mean"] = fixed[col]
            fixed[f"{col}_std"] = float("nan")
        fixed["n_seeds"] = 0
        summary_parts.append(fixed[["method", "layer", "mode", "n_seeds", *[f"{m}_mean" for m in metric_cols], *[f"{m}_std" for m in metric_cols]]])
    summary = pd.concat(summary_parts, ignore_index=True, sort=False) if summary_parts else seed_summary

    per_parsed = per["method"].map(parse_method)
    per["layer"] = [x[0] for x in per_parsed]
    per["mode"] = [x[1] for x in per_parsed]
    per["seed"] = [x[2] for x in per_parsed]
    per_cell_seed = per.groupby(["method", "layer", "mode", "seed", "cell_line"], dropna=False)[
        ["Hit@10", "NDCG@10", "MeanAP", "AUROC", "Hit@50"]
    ].mean().reset_index()
    per_cell = per_cell_seed.groupby(["layer", "mode", "cell_line"], dropna=False)[
        ["Hit@10", "NDCG@10", "MeanAP", "AUROC", "Hit@50"]
    ].agg(["mean", "std"]).reset_index()
    per_cell.columns = ["_".join([str(x) for x in col if str(x)]) for col in per_cell.columns]
    per_cell = per_cell.rename(columns={"layer_": "layer", "mode_": "mode", "cell_line_": "cell_line"})

    write_table(seed_summary, out / "layer_sensitivity_seed_summary.tsv")
    write_table(summary, out / "layer_sensitivity_summary.tsv")
    write_table(per_cell, out / "layer_sensitivity_per_cell.tsv")

    delta_rows = []
    for seed in sorted(x for x in per["seed"].dropna().unique()):
        seed_per = per[per["seed"].eq(seed)]
        for layer in ["L3", "L4", "L5"]:
            direct = seed_per[(seed_per["layer"].eq(layer)) & (seed_per["mode"].eq("direct"))]
            adapted = seed_per[(seed_per["layer"].eq(layer)) & (seed_per["mode"].eq("adapted"))]
            if direct.empty or adapted.empty:
                continue
            d = direct[["drug", "cell_line", "Hit@10", "NDCG@10", "MeanAP", "AUROC", "Hit@50"]].rename(
                columns={m: f"{m}_direct" for m in ["Hit@10", "NDCG@10", "MeanAP", "AUROC", "Hit@50"]}
            )
            a = adapted[["drug", "cell_line", "Hit@10", "NDCG@10", "MeanAP", "AUROC", "Hit@50"]].rename(
                columns={m: f"{m}_adapted" for m in ["Hit@10", "NDCG@10", "MeanAP", "AUROC", "Hit@50"]}
            )
            joined = d.merge(a, on=["drug", "cell_line"], how="inner")
            joined["layer"] = layer
            joined["seed"] = int(seed)
            for metric in ["Hit@10", "NDCG@10", "MeanAP", "AUROC", "Hit@50"]:
                joined[f"{metric}_adapted_minus_direct"] = joined[f"{metric}_adapted"] - joined[f"{metric}_direct"]
            delta_rows.append(joined)
    if delta_rows:
        write_table(pd.concat(delta_rows, ignore_index=True), out / "adaptation_delta_by_unit.tsv")

    print(out)


if __name__ == "__main__":
    main()
