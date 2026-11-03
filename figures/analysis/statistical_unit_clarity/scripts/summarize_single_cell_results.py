"""Compute manuscript result tables without historical figure layouts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

BOOTSTRAP_SEED=20260713
BOOTSTRAP_REPLICATES=20_000
ALPHA=0.05
CELLS=["A549","SW480","NCI-H460"]


def parse_value(value: str | None):
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return ""
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value

def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{k: parse_value(v) for k, v in row.items()} for row in csv.DictReader(handle, delimiter="\t")]

def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

def stable_seed(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
    return BOOTSTRAP_SEED + int(digest[:8], 16)

def bootstrap_mean_ci(values: np.ndarray, label: str) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or len(arr) == 0 or np.isnan(arr).any():
        raise ValueError(f"Invalid bootstrap input for {label}")
    rng = np.random.default_rng(stable_seed(label))
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    chunk = 2_000
    for start in range(0, BOOTSTRAP_REPLICATES, chunk):
        stop = min(start + chunk, BOOTSTRAP_REPLICATES)
        indices = rng.integers(0, len(arr), size=(stop - start, len(arr)))
        draws[start:stop] = arr[indices].mean(axis=1)
    return tuple(np.quantile(draws, [ALPHA / 2, 1 - ALPHA / 2]).tolist())

def exact_signflip_integer(differences: np.ndarray, scale: int) -> float:
    scaled = np.rint(np.asarray(differences, dtype=float) * scale).astype(int)
    if not np.allclose(scaled / scale, differences, atol=1e-9):
        raise ValueError("Differences are not representable at the requested scale")
    nonzero = scaled[scaled != 0]
    observed = abs(int(nonzero.sum()))
    counts = {0: 1}
    for value in nonzero:
        next_counts: dict[int, int] = {}
        for total, count in counts.items():
            next_counts[total + int(value)] = next_counts.get(total + int(value), 0) + count
            next_counts[total - int(value)] = next_counts.get(total - int(value), 0) + count
        counts = next_counts
    denominator = 2 ** len(nonzero)
    numerator = sum(count for total, count in counts.items() if abs(total) >= observed)
    return numerator / denominator if denominator else 1.0

def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()

def pct(v: float) -> float:
    return 100.0 * float(v)

def add_drug_cluster_intervals(overall_curve: list[dict], perdrug_rows: list[dict]) -> list[dict]:
    holdout = [
        row
        for row in perdrug_rows
        if row.get("metric_family") == "holdout"
        and row.get("cell_line") in CELLS
        and row.get("drug") not in {None, ""}
    ]
    by_drug: dict[str, list[dict]] = {}
    for row in holdout:
        by_drug.setdefault(str(row["drug"]), []).append(row)
    if len(by_drug) != 30 or any({row["cell_line"] for row in rows} != set(CELLS) for rows in by_drug.values()):
        raise AssertionError("Fig. 5 clustered interval requires 30 drugs with all three fixed cell lines")

    corrected = []
    for row in overall_curve:
        threshold = int(round(pct(row["threshold_percent"])))
        hit_column = f"hit_top{threshold}pct"
        values = np.array(
            [np.mean([float(item[hit_column]) for item in rows]) for rows in by_drug.values()],
            dtype=float,
        )
        low, high = bootstrap_mean_ci(values, f"fig5_recovery_top{threshold}")
        updated = dict(row)
        updated.update(
            {
                "n_independent_drugs": len(values),
                "n_nested_drug_cell_units": int(row["n_units"]),
                "inferential_unit": "drug",
                "cluster95_low": low,
                "cluster95_high": high,
                "interval_type": "whole_drug_cluster_percentile_bootstrap_95",
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED,
            }
        )
        corrected.append(updated)
    return corrected

def derive_pseudobulk_paired_summary(single_cell_rows: list[dict], pseudo_bulk_rows: list[dict]) -> list[dict]:
    single = {
        (str(row["drug"]), str(row["cell_line"])): row
        for row in single_cell_rows
        if row.get("metric_family") == "holdout" and row.get("cell_line") in CELLS
    }
    pseudo = {
        (str(row["drug"]), str(row["cell_line"])): row
        for row in pseudo_bulk_rows
        if row.get("selection_policy") == "matched_singlecell_objective" and row.get("cell_line") in CELLS
    }
    if len(single) != 90 or set(single) != set(pseudo):
        raise AssertionError("Fig. 5 pseudo-bulk comparison requires the same 30 drugs x 3 cells")

    summaries = []
    raw_p_values = []
    drugs = sorted({drug for drug, _ in single})
    for threshold in [5, 10]:
        column = f"hit_top{threshold}pct"
        single_drug = []
        pseudo_drug = []
        for drug in drugs:
            single_drug.append(np.mean([float(single[(drug, cell)][column]) for cell in CELLS]))
            pseudo_drug.append(np.mean([float(pseudo[(drug, cell)][column]) for cell in CELLS]))
        single_values = np.asarray(single_drug, dtype=float)
        pseudo_values = np.asarray(pseudo_drug, dtype=float)
        differences = single_values - pseudo_values
        low, high = bootstrap_mean_ci(differences, f"fig5_pseudobulk_diff_top{threshold}")
        p_value = exact_signflip_integer(differences, scale=3)
        raw_p_values.append(p_value)
        summaries.append(
            {
                "threshold_percent": threshold,
                "endpoint_role": "secondary_control",
                "n_independent_drugs": len(drugs),
                "n_nested_drug_cell_units": len(single),
                "single_cell_mean": float(single_values.mean()),
                "pseudo_bulk_mean": float(pseudo_values.mean()),
                "paired_difference": float(differences.mean()),
                "paired_drug_bootstrap95_low": low,
                "paired_drug_bootstrap95_high": high,
                "exact_signflip_p_two_sided": p_value,
                "test_unit": "drug_after_averaging_three_fixed_cell_lines",
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED,
            }
        )
    for row, adjusted in zip(summaries, holm_adjust(raw_p_values), strict=True):
        row["holm_adjusted_p_across_top5_top10"] = adjusted
    return summaries

def derive_tables(curve_rows: list[dict], summary_rows: list[dict], perdrug_rows: list[dict]) -> dict[str, list[dict]]:
    overall_curve = [
        r
        for r in curve_rows
        if r.get("cell_line", "") == "" and r["evaluation_unit"] == "drug" and r["n_units"] == 90 and r["threshold_percent"] in [0.01, 0.05, 0.1, 0.2]
    ]
    overall_curve = sorted(overall_curve, key=lambda r: r["threshold_percent"])

    cell_curve = [
        r
        for r in curve_rows
        if r.get("cell_line", "") in ["A549", "SW480", "NCI-H460"]
        and r["evaluation_unit"] == "drug"
        and r["n_units"] == 30
        and r["threshold_percent"] in [0.01, 0.05, 0.1, 0.2]
    ]
    cell_curve = sorted(cell_curve, key=lambda r: (r["threshold_percent"], ["A549", "SW480", "NCI-H460"].index(r["cell_line"])))

    per_cell_summary = []
    for cell in ["A549", "SW480", "NCI-H460"]:
        sub = [
            r
            for r in curve_rows
            if r.get("cell_line") == cell and r["evaluation_unit"] == "drug" and r["n_units"] == 30 and r["threshold_percent"] in [0.05, 0.1]
        ]
        sub = sorted(sub, key=lambda r: r["threshold_percent"])
        seen = next(r for r in summary_rows if r["cell_line"] == cell)
        for row in sub:
            per_cell_summary.append(
                {
                    "cell_line": cell,
                    "threshold_label": "top5%" if row["threshold_percent"] == 0.05 else "top10%",
                    "threshold_percent": row["threshold_percent"],
                    "n_units": row["n_units"],
                    "n_success": row["n_success"],
                    "recovery": row["recovery"],
                    "wilson95_low": row["wilson95_low"],
                    "wilson95_high": row["wilson95_high"],
                    "matched_random_anyhit_mean": row["matched_random_anyhit_mean"],
                    "seen_test_Hit@10": seen["seen_test_Hit@10"],
                    "seen_test_Hit@50": seen["seen_test_Hit@50"],
                }
            )

    map_rows = []
    grouped: dict[str, list[dict]] = {}
    for row in perdrug_rows:
        if row.get("metric_family") != "holdout":
            continue
        grouped.setdefault(str(row["drug"]), []).append(row)
    ordered = []
    for drug, rows in grouped.items():
        state_values = []
        avg_pct = float(np.mean([float(r["best_percentile"]) for r in rows]))
        for cell in ["A549", "SW480", "NCI-H460"]:
            row = next(r for r in rows if r["cell_line"] == cell)
            top5 = int(row["hit_top5pct"])
            top10 = int(row["hit_top10pct"])
            if top5 == 1:
                state = "top5"
                score = 2
            elif top10 == 1:
                state = "top10"
                score = 1
            else:
                state = "missed"
                score = 0
            map_rows.append(
                {
                    "drug": drug,
                    "cell_line": cell,
                    "state": state,
                    "best_percentile": row["best_percentile"],
                    "hit_top5pct": top5,
                    "hit_top10pct": top10,
                }
            )
            state_values.append(score)
        ordered.append(
            {
                "drug": drug,
                "pattern": tuple(state_values),
                "top5_count": sum(v == 2 for v in state_values),
                "top10_count": sum(v >= 1 for v in state_values),
                "avg_percentile": avg_pct,
            }
        )
    ordered = sorted(
        ordered,
        key=lambda x: (-x["top5_count"], -x["top10_count"], x["avg_percentile"], x["drug"]),
    )
    order_lookup = {row["drug"]: i for i, row in enumerate(ordered)}
    map_rows = sorted(map_rows, key=lambda r: (order_lookup[r["drug"]], ["A549", "SW480", "NCI-H460"].index(r["cell_line"])))

    seen_diag = []
    for row in summary_rows:
        seen_diag.append(
            {
                "cell_line": row["cell_line"],
                "seen_test_Hit@5": row["seen_test_Hit@5"],
                "seen_test_Hit@10": row["seen_test_Hit@10"],
                "seen_test_Hit@20": row["seen_test_Hit@20"],
                "seen_test_Hit@50": row["seen_test_Hit@50"],
            }
        )

    audit = [
        {"field": "overall_n_units", "value": 90},
        {"field": "per_cell_n_units", "value": 30},
        {"field": "shared_holdout_drugs", "value": 30},
        {"field": "candidate_space", "value": 16583},
        {"field": "overall_top5_recovery", "value": next(r["recovery"] for r in overall_curve if r["threshold_percent"] == 0.05)},
        {"field": "overall_top10_recovery", "value": next(r["recovery"] for r in overall_curve if r["threshold_percent"] == 0.1)},
        {"field": "overall_random_top5", "value": next(r["matched_random_anyhit_mean"] for r in overall_curve if r["threshold_percent"] == 0.05)},
        {"field": "overall_random_top10", "value": next(r["matched_random_anyhit_mean"] for r in overall_curve if r["threshold_percent"] == 0.1)},
        {"field": "A549_top1_top5_top10_top20_success", "value": "15/30,21/30,23/30,23/30"},
        {"field": "SW480_top1_top5_top10_top20_success", "value": "10/30,19/30,21/30,21/30"},
        {"field": "NCI-H460_top1_top5_top10_top20_success", "value": "10/30,18/30,21/30,23/30"},
    ]

    return {
        "overall_curve": overall_curve,
        "cell_curve": cell_curve,
        "per_cell_summary": per_cell_summary,
        "endpoint_map": map_rows,
        "seen_diag": seen_diag,
        "audit": audit,
        "drug_order": [{"drug": row["drug"], "order": i + 1, "top5_count": row["top5_count"], "top10_count": row["top10_count"], "avg_percentile": row["avg_percentile"]} for i, row in enumerate(ordered)],
    }

def derive_pseudobulk_control(control_rows: list[dict], overall_curve: list[dict]) -> list[dict]:
    random_by_threshold = {float(row["threshold_percent"]): row["matched_random_anyhit_mean"] for row in overall_curve}
    rows = [
        row
        for row in control_rows
        if row.get("cell_line") == "pooled"
        and row.get("threshold_percent") in [0.01, 0.05, 0.1, 0.2]
        and row.get("method") in {"Single-cell adapter", "Pseudo-bulk adapter", "X128 pseudo-bulk adapter"}
    ]
    rows = sorted(rows, key=lambda row: (row["method"], float(row["threshold_percent"])))
    out = []
    for row in rows:
        threshold = float(row["threshold_percent"])
        out.append(
            {
                "method": "Pseudo-bulk adapter" if row["method"] == "X128 pseudo-bulk adapter" else row["method"],
                "cell_line": row["cell_line"],
                "threshold_percent": threshold,
                "n_units": int(row["n_units"]),
                "n_success": int(row["n_success"]),
                "recovery": float(row["recovery"]),
                "wilson95_low": float(row["wilson95_low"]),
                "wilson95_high": float(row["wilson95_high"]),
                "candidate_space": int(row["candidate_space"]),
                "matched_random_anyhit_mean": float(random_by_threshold[threshold]),
            }
        )
    return out

def aggregate_upset_combinations(rows: list[dict]) -> list[dict]:
    by_drug: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_drug.setdefault(str(row["drug"]), {})[str(row["cell_line"])] = row

    nested: dict[str, dict[str, int]] = {}
    for drug, by_cell in by_drug.items():
        states = [str(by_cell[cell]["state"]) for cell in CELLS]
        code = "".join("1" if state in {"top5", "top10"} else "0" for state in states)
        recovered_states = [state for state in states if state in {"top5", "top10"}]
        if not recovered_states:
            quality = "unrecovered_drugs"
        elif all(state == "top5" for state in recovered_states):
            quality = "top5_only_drugs"
        else:
            quality = "top10_present_drugs"
        nested.setdefault(
            code,
            {"top5_only_drugs": 0, "top10_present_drugs": 0, "unrecovered_drugs": 0},
        )[quality] += 1

    combos = []
    for code, counts in nested.items():
        total = sum(counts.values())
        combos.append(
            {
                "combo_code": code,
                "A549_recovered_top10": int(code[0] == "1"),
                "SW480_recovered_top10": int(code[1] == "1"),
                "NCI-H460_recovered_top10": int(code[2] == "1"),
                "top5_only_drugs": counts["top5_only_drugs"],
                "top10_present_drugs": counts["top10_present_drugs"],
                "unrecovered_drugs": counts["unrecovered_drugs"],
                "total_drugs": total,
            }
        )
    return sorted(combos, key=lambda r: (-r["total_drugs"], -str(r["combo_code"]).count("1"), str(r["combo_code"])))



def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-curve",type=Path,required=True)
    parser.add_argument("--metric-summary",type=Path,required=True)
    parser.add_argument("--per-drug",type=Path,required=True)
    parser.add_argument("--pseudobulk-control",type=Path)
    parser.add_argument("--pseudobulk-matched",type=Path)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    perdrug=read_rows(args.per_drug)
    tables=derive_tables(read_rows(args.recovery_curve),read_rows(args.metric_summary),perdrug)
    tables["overall_curve"]=add_drug_cluster_intervals(tables["overall_curve"],perdrug)
    if args.pseudobulk_control:
        tables["pseudobulk_control"]=derive_pseudobulk_control(read_rows(args.pseudobulk_control),tables["overall_curve"])
    if args.pseudobulk_matched:
        tables["pseudobulk_paired_statistics"]=derive_pseudobulk_paired_summary(perdrug,read_rows(args.pseudobulk_matched))
    for name,rows in tables.items():
        if name!="audit" and rows:
            write_tsv(args.output_dir / f"single_cell_{name}.tsv",rows,list(rows[0]))


if __name__ == "__main__":
    main()
