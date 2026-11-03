#!/usr/bin/env python
"""Aggregate same-drug target score vectors across held-out cell lines."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from common import add_code_release_to_path, first_existing_path, read_table, read_yaml, write_json, write_table


META_COLS = ["drug", "cell_line", "n_profiles", "true_targets"]
AGG_METHODS = ["raw_score_mean", "cell_zscore_mean", "rank_borda", "rrf", "profile_weighted_zmean"]
METRIC_COLS = ["Hit@1", "Hit@5", "Hit@10", "Hit@20", "Hit@30", "Hit@50", "AUROC", "NDCG@10", "MeanAP"]


def resolve_existing(value: str | Path) -> Path:
    text = os.path.expandvars(str(value))
    return Path(text).resolve()


def code_root_from_config(cfg: dict) -> Path:
    return first_existing_path([cfg["code_release"].get("local_root", ""), cfg["code_release"].get("server_root", "")])


def round429_root_from_config(cfg: dict) -> Path:
    return first_existing_path([cfg["round429"].get("local_root", ""), cfg["round429"].get("server_root", "")])


def label_order_from_config(cfg: dict) -> Path:
    return first_existing_path([cfg["label_order"].get("local_path", ""), cfg["label_order"].get("server_path", "")])


def score_path(cfg: dict, root429: Path, mode: str, seed: int) -> Path:
    layer = str(cfg["round429"]["layer"])
    filename = cfg["modes"][mode]
    return root429 / "outputs" / "lincs_layer_sensitivity" / "scores" / layer / f"seed_{seed}" / filename


def canonical_target_text(value: object) -> str:
    from metar_legacy.data_io import split_targets

    return "|".join(sorted(split_targets(value)))


def validate_input(df: pd.DataFrame, labels: list[str], path: Path) -> None:
    missing_meta = [c for c in META_COLS if c not in df.columns]
    if missing_meta:
        raise ValueError(f"{path}: missing metadata columns {missing_meta}")
    missing_labels = [label for label in labels if label not in df.columns]
    if missing_labels:
        raise ValueError(f"{path}: missing {len(missing_labels)} label columns, e.g. {missing_labels[:5]}")


def collapse_drug_cell(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    target_check = df.assign(_target=df["true_targets"].map(canonical_target_text))
    inconsistent = target_check.groupby(["drug", "cell_line"])["_target"].nunique()
    bad = inconsistent[inconsistent.gt(1)]
    if len(bad):
        raise ValueError(f"Target labels differ within drug-cell units: {bad.index[:5].tolist()}")

    score = df.groupby(["drug", "cell_line"], dropna=False)[labels].mean().reset_index()
    meta = (
        target_check.groupby(["drug", "cell_line"], dropna=False)
        .agg(n_profiles=("n_profiles", "sum"), true_targets=("_target", "first"), input_rows=("drug", "size"))
        .reset_index()
    )
    return score.merge(meta, on=["drug", "cell_line"], how="left", validate="one_to_one")


def check_drug_targets(unit: pd.DataFrame) -> None:
    target_n = unit.groupby("drug")["true_targets"].nunique()
    bad = target_n[target_n.gt(1)]
    if len(bad):
        raise ValueError(f"Target labels differ across cell lines for drugs: {bad.index[:10].tolist()}")


def rank_descending(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.float64)
    return ranks


def zscore_row(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    sd = float(arr.std())
    if not math.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - float(arr.mean())) / sd


def aggregate_scores(grp: pd.DataFrame, labels: list[str], method: str, rrf_k: float) -> np.ndarray:
    scores = grp[labels].to_numpy(dtype=np.float64)
    if method == "raw_score_mean":
        return scores.mean(axis=0)
    if method == "cell_zscore_mean":
        return np.vstack([zscore_row(row) for row in scores]).mean(axis=0)
    if method == "rank_borda":
        ranks = np.vstack([rank_descending(row) for row in scores])
        return -ranks.mean(axis=0)
    if method == "rrf":
        ranks = np.vstack([rank_descending(row) for row in scores])
        return (1.0 / (float(rrf_k) + ranks)).sum(axis=0)
    if method == "profile_weighted_zmean":
        z = np.vstack([zscore_row(row) for row in scores])
        w = np.log1p(grp["n_profiles"].to_numpy(dtype=np.float64)).reshape(-1, 1)
        if not np.isfinite(w).all() or float(w.sum()) <= 0:
            return z.mean(axis=0)
        return (z * w).sum(axis=0) / float(w.sum())
    raise ValueError(f"Unknown aggregation method: {method}")


def make_drug_score_table(unit: pd.DataFrame, labels: list[str], method: str, min_cell_lines: int, rrf_k: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for drug, grp in unit.groupby("drug", sort=True, dropna=False):
        n_cells = int(grp["cell_line"].nunique())
        if n_cells < min_cell_lines:
            continue
        rec: dict[str, object] = {
            "drug": drug,
            "n_cell_lines": n_cells,
            "cell_lines": "|".join(sorted(map(str, grp["cell_line"].unique()))),
            "n_profiles_total": int(grp["n_profiles"].sum()),
            "n_drug_cell_units": int(len(grp)),
            "true_targets": str(grp["true_targets"].iloc[0]),
        }
        agg = aggregate_scores(grp, labels, method, rrf_k)
        rec.update({label: float(value) for label, value in zip(labels, agg)})
        rows.append(rec)
    return pd.DataFrame(rows)


def evaluate_score_table(df: pd.DataFrame, labels: list[str], *, method: str, mode: str, seed: int) -> pd.DataFrame:
    from metar_legacy.data_io import labels_to_matrix
    from metar_legacy.metrics import UnitMetricConfig, compute_unit_metrics

    meta_cols = [c for c in df.columns if c not in set(labels)]
    y = labels_to_matrix(df, labels, target_col="true_targets")
    per = compute_unit_metrics(
        df[labels].to_numpy(dtype=np.float64),
        y,
        df[meta_cols].reset_index(drop=True),
        config=UnitMetricConfig(ks=(1, 5, 10, 20, 30, 50), best_percentile_higher_is_better=True),
    )
    per.insert(0, "method", method)
    per.insert(1, "mode", mode)
    per.insert(2, "seed", seed)
    return per


def single_cell_per_drug(unit: pd.DataFrame, labels: list[str], min_cell_lines: int, mode: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_cell = evaluate_score_table(unit, labels, method="single_cell_unit", mode=mode, seed=seed)
    keep = unit.groupby("drug")["cell_line"].nunique()
    keep_drugs = set(keep[keep.ge(min_cell_lines)].index.astype(str))
    per_cell = per_cell[per_cell["drug"].astype(str).isin(keep_drugs)].copy()

    mean_rows = []
    best_rows = []
    for drug, grp in per_cell.groupby("drug", sort=True, dropna=False):
        first = grp.iloc[0]
        base = {
            "drug": drug,
            "mode": mode,
            "seed": seed,
            "n_cell_lines": int(grp["cell_line"].nunique()),
            "cell_lines": "|".join(sorted(map(str, grp["cell_line"].unique()))),
            "true_targets": first["true_targets"],
        }
        mean_rec = dict(base, method="single_cell_mean")
        best_rec = dict(base, method="single_cell_best")
        for metric in METRIC_COLS:
            mean_rec[metric] = float(grp[metric].mean())
            best_rec[metric] = float(grp[metric].max())
        mean_rows.append(mean_rec)
        best_rows.append(best_rec)
    return pd.DataFrame(mean_rows), pd.DataFrame(best_rows)


def topk_jaccard(score_rows: pd.DataFrame, labels: list[str], k: int = 10) -> pd.DataFrame:
    out = []
    for drug, grp in score_rows.groupby("drug", sort=True, dropna=False):
        if grp["cell_line"].nunique() < 2:
            continue
        top_sets = []
        for _, row in grp.iterrows():
            arr = row[labels].to_numpy(dtype=np.float64)
            order = np.argsort(-arr, kind="mergesort")[:k]
            top_sets.append(set(order.tolist()))
        vals = []
        for i in range(len(top_sets)):
            for j in range(i + 1, len(top_sets)):
                denom = len(top_sets[i] | top_sets[j])
                vals.append(len(top_sets[i] & top_sets[j]) / denom if denom else float("nan"))
        out.append({"drug": drug, "top10_jaccard_mean": float(np.nanmean(vals)) if vals else float("nan")})
    return pd.DataFrame(out)


def true_rank_variance(score_rows: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    from metar_legacy.data_io import split_targets

    pos = {label: i for i, label in enumerate(labels)}
    out = []
    for drug, grp in score_rows.groupby("drug", sort=True, dropna=False):
        targets = [pos[t] for t in split_targets(grp["true_targets"].iloc[0]) if t in pos]
        if not targets or grp["cell_line"].nunique() < 2:
            continue
        best_ranks = []
        for _, row in grp.iterrows():
            arr = row[labels].to_numpy(dtype=np.float64)
            ranks = rank_descending(arr)
            best_ranks.append(float(np.min(ranks[targets])))
        out.append(
            {
                "drug": drug,
                "best_true_rank_mean_across_cells": float(np.mean(best_ranks)),
                "best_true_rank_std_across_cells": float(np.std(best_ranks)),
            }
        )
    return pd.DataFrame(out)


def summarize(per: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, grp in per.groupby(["mode", "method"], dropna=False):
        mode, method = keys
        rec: dict[str, object] = {"mode": mode, "method": method, "n_seeds": int(grp["seed"].nunique()), "n_drugs": int(grp["drug"].nunique())}
        seed_means = grp.groupby("seed")[METRIC_COLS].mean()
        for metric in METRIC_COLS:
            rec[f"{metric}_mean"] = float(seed_means[metric].mean())
            rec[f"{metric}_std"] = float(seed_means[metric].std(ddof=1)) if len(seed_means) > 1 else float("nan")
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["mode", "method"]).reset_index(drop=True)


def summarize_by_n_cells(per: pd.DataFrame) -> pd.DataFrame:
    bins = [
        ("n_cells_eq2", lambda s: s.eq(2)),
        ("n_cells_ge3", lambda s: s.ge(3)),
        ("n_cells_ge5", lambda s: s.ge(5)),
        ("all_multicell", lambda s: s.ge(2)),
    ]
    rows = []
    for group_name, mask_fn in bins:
        sub = per[mask_fn(per["n_cell_lines"].astype(int))].copy()
        if sub.empty:
            continue
        for keys, grp in sub.groupby(["mode", "method"], dropna=False):
            mode, method = keys
            rec: dict[str, object] = {
                "n_cell_group": group_name,
                "mode": mode,
                "method": method,
                "n_seeds": int(grp["seed"].nunique()),
                "n_drugs": int(grp["drug"].nunique()),
            }
            seed_means = grp.groupby("seed")[METRIC_COLS].mean()
            for metric in METRIC_COLS:
                rec[f"{metric}_mean"] = float(seed_means[metric].mean())
                rec[f"{metric}_std"] = float(seed_means[metric].std(ddof=1)) if len(seed_means) > 1 else float("nan")
            rows.append(rec)
    return pd.DataFrame(rows).sort_values(["n_cell_group", "mode", "method"]).reset_index(drop=True)


def add_deltas(per: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = per[per["method"].eq("single_cell_mean")][["mode", "seed", "drug", *METRIC_COLS]].copy()
    for method, grp in per.groupby("method", dropna=False):
        joined = grp.merge(base, on=["mode", "seed", "drug"], how="left", suffixes=("", "_single_cell_mean"))
        for metric in METRIC_COLS:
            joined[f"{metric}_minus_single_cell_mean"] = joined[metric] - joined[f"{metric}_single_cell_mean"]
        rows.append(joined[[c for c in joined.columns if not c.endswith("_single_cell_mean")]])
    return pd.concat(rows, ignore_index=True)


def random_row(n_labels: int, n_drugs: int) -> dict[str, object]:
    rec: dict[str, object] = {"mode": "reference", "method": "Random", "n_seeds": 0, "n_drugs": n_drugs}
    for metric in METRIC_COLS:
        if metric.startswith("Hit@"):
            k = int(metric.split("@", 1)[1])
            rec[f"{metric}_mean"] = min(float(k) / float(n_labels), 1.0)
        elif metric == "AUROC":
            rec[f"{metric}_mean"] = 0.5
        else:
            rec[f"{metric}_mean"] = float("nan")
        rec[f"{metric}_std"] = float("nan")
    return rec


def write_source_lock(cfg: dict, out_root: Path, code_root: Path, root429: Path, label_path: Path) -> None:
    data = {
        "round": cfg.get("round"),
        "task": cfg.get("task"),
        "code_release_root": str(code_root),
        "round429_root": str(root429),
        "label_order": str(label_path),
        "layer": cfg["round429"]["layer"],
        "seeds": cfg["seeds"],
        "modes": cfg["modes"],
        "aggregation": cfg["aggregation"],
    }
    write_json(data, out_root / "source_lock.json")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run Round430 multi-cell consensus aggregation")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = read_yaml(cfg_path)
    overlay = cfg_path.parents[1]
    out_root = overlay / cfg["outputs"]["root"]
    score_out_root = out_root / cfg["outputs"]["scores_dir"]
    metrics_dir = out_root / cfg["outputs"]["metrics_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)

    code_root = add_code_release_to_path(code_root_from_config(cfg))
    from metar_legacy.data_io import read_label_order

    root429 = round429_root_from_config(cfg)
    label_path = label_order_from_config(cfg)
    labels = read_label_order(label_path)
    if len(labels) != 668:
        raise ValueError(f"Expected 668 labels, found {len(labels)}")

    min_cell_lines = int(cfg["aggregation"]["min_cell_lines"])
    rrf_k = float(cfg["aggregation"].get("rrf_k", 60))
    methods = list(cfg["aggregation"].get("methods", AGG_METHODS))
    write_source_lock(cfg, out_root, code_root, root429, label_path)

    all_per_rows = []
    audit_rows = []
    diagnostics = []
    denominators: dict[tuple[str, int], set[str]] = {}

    for mode in cfg["modes"]:
        for seed in cfg["seeds"]:
            path = score_path(cfg, root429, mode, int(seed))
            df = read_table(path)
            validate_input(df, labels, path)
            unit = collapse_drug_cell(df, labels)
            check_drug_targets(unit)

            cell_counts = unit.groupby("drug")["cell_line"].nunique()
            multi_drugs = set(cell_counts[cell_counts.ge(min_cell_lines)].index.astype(str))
            denominators[(mode, int(seed))] = multi_drugs
            unit_multi = unit[unit["drug"].astype(str).isin(multi_drugs)].copy()

            jacc = topk_jaccard(unit_multi, labels)
            rank_var = true_rank_variance(unit_multi, labels)
            diag = jacc.merge(rank_var, on="drug", how="outer")
            diag["mode"] = mode
            diag["seed"] = int(seed)
            diagnostics.append(diag)

            mean_base, best_base = single_cell_per_drug(unit, labels, min_cell_lines, mode, int(seed))
            all_per_rows.extend([mean_base, best_base])

            audit_rows.append(
                {
                    "mode": mode,
                    "seed": int(seed),
                    "source_path": str(path),
                    "input_rows": int(len(df)),
                    "drug_cell_units": int(len(unit)),
                    "n_drugs": int(cell_counts.shape[0]),
                    "multi_cell_drugs": int(len(multi_drugs)),
                    "n_cells_min": int(cell_counts.min()),
                    "n_cells_max": int(cell_counts.max()),
                    "target_consistency_issues": 0,
                }
            )

            for method in methods:
                drug_scores = make_drug_score_table(unit, labels, method, min_cell_lines, rrf_k)
                score_path_out = score_out_root / mode / f"seed_{seed}" / f"{method}_drug_scores.tsv.gz"
                write_table(drug_scores, score_path_out)
                per = evaluate_score_table(drug_scores, labels, method=method, mode=mode, seed=int(seed))
                all_per_rows.append(per)

    common_denominator = set.intersection(*denominators.values()) if denominators else set()
    per_all = pd.concat(all_per_rows, ignore_index=True)
    per_all = per_all[per_all["drug"].astype(str).isin(common_denominator)].copy()
    per_all = add_deltas(per_all)
    if diagnostics:
        diag_all = pd.concat(diagnostics, ignore_index=True)
        diag_all = diag_all[diag_all["drug"].astype(str).isin(common_denominator)].copy()
        per_all = per_all.merge(diag_all, on=["mode", "seed", "drug"], how="left")

    summary = summarize(per_all)
    summary = pd.concat([summary, pd.DataFrame([random_row(len(labels), len(common_denominator))])], ignore_index=True, sort=False)
    by_cells = summarize_by_n_cells(per_all)
    audit = pd.DataFrame(audit_rows)
    audit["strict_common_multicell_drugs"] = len(common_denominator)
    audit["candidate_space"] = len(labels)
    audit["expected_input_rows"] = 7754
    audit["expected_drug_cell_units"] = 1452
    audit["expected_multi_cell_drugs"] = 415

    write_table(per_all, metrics_dir / "multicell_consensus_per_drug.tsv")
    write_table(summary, metrics_dir / "multicell_consensus_summary.tsv")
    write_table(by_cells, metrics_dir / "multicell_consensus_by_n_cells.tsv")
    write_table(audit, metrics_dir / "multicell_consensus_audit.tsv")

    print(json.dumps({"output_root": str(out_root), "strict_common_multicell_drugs": len(common_denominator)}, indent=2))


if __name__ == "__main__":
    main()
