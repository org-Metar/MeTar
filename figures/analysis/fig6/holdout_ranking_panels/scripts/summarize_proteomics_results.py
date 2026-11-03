"""Compute manuscript result tables without historical figure layouts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

L_SEEN=102
L_FULL=16583
K_SEEN=[1,5,10]
HOLDOUT_THRESHOLDS=[0.01,0.05,0.10]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))

def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def fval(value: str | object | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)

def ival(value: str | object | None) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))

def pct(value: float) -> float:
    return 100.0 * value

def mean(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")

def anyhit_random(n_positive: int, k: int, universe: int) -> float:
    if n_positive <= 0:
        return float("nan")
    if k <= 0:
        return 0.0
    if k >= universe:
        return 1.0
    miss = 1.0
    for i in range(k):
        miss *= (universe - n_positive - i) / (universe - i)
    return 1.0 - miss

def recovery_band(row: dict[str, object]) -> str:
    if float(row["holdout_top1pct"]) >= 0.5:
        return "top 1%"
    if float(row["holdout_top5pct"]) >= 0.5:
        return "top 5%"
    if float(row["holdout_top10pct"]) >= 0.5:
        return "top 10%"
    return "> top 10%"

def derive_tables(per_drug: list[dict[str, str]], holdout_numeric: list[dict[str, str]]) -> dict[str, list[dict[str, object]]]:
    seen_eval = [r for r in per_drug if ival(r.get("n_seen_pos")) > 0 and not math.isnan(fval(r.get("Hit@1")))]
    holdout_eval = [r for r in per_drug if ival(r.get("n_holdout_pos")) > 0]
    seen_drugs = {r["drug"] for r in seen_eval}
    holdout_drugs = {r["drug"] for r in holdout_eval}
    overlap = seen_drugs & holdout_drugs

    seen_hitk: list[dict[str, object]] = []
    for k in K_SEEN:
        values = [fval(r[f"Hit@{k}"]) for r in seen_eval]
        random_values = [anyhit_random(ival(r["n_seen_pos"]), k, L_SEEN) for r in seen_eval]
        n_recovered = sum(1 for v in values if v >= 0.5)
        seen_hitk.append(
            {
                "metric": f"Hit@{k}",
                "k": k,
                "universe": L_SEEN,
                "n_evaluable_drugs": len(seen_eval),
                "n_recovered": n_recovered,
                "MeTar": mean(values),
                "matched_random": mean(random_values),
            }
        )

    ranking_quality = [
        {
            "metric": "NDCG@10",
            "n_evaluable_drugs": len(seen_eval),
            "MeTar": mean([fval(r["NDCG@10"]) for r in seen_eval]),
        },
        {
            "metric": "MAP@10",
            "n_evaluable_drugs": len(seen_eval),
            "MeTar": mean([fval(r["MAP@10"]) for r in seen_eval]),
        },
    ]

    holdout: list[dict[str, object]] = []
    for threshold in HOLDOUT_THRESHOLDS:
        k = int(round(threshold * L_FULL))
        col = f"holdout_top{int(threshold * 100)}pct"
        values = [fval(r[col]) for r in holdout_eval]
        random_values = [anyhit_random(ival(r["n_holdout_pos"]), k, L_FULL) for r in holdout_eval]
        n_recovered = sum(1 for v in values if v >= 0.5)
        holdout.append(
            {
                "threshold": f"top{int(threshold * 100)}%",
                "threshold_percent": 100 * threshold,
                "k": k,
                "universe": L_FULL,
                "n_evaluable_drugs": len(holdout_eval),
                "n_recovered": n_recovered,
                "MeTar": mean(values),
                "matched_random": mean(random_values),
            }
        )

    numeric_by_drug = {row["drug"]: row for row in holdout_numeric}
    holdout_detail: list[dict[str, object]] = []
    for r in holdout_eval:
        if r["drug"] not in numeric_by_drug:
            raise AssertionError(f"Missing numeric holdout percentile for {r['drug']}")
        numeric_row = numeric_by_drug[r["drug"]]
        for col in ["holdout_top1pct", "holdout_top5pct", "holdout_top10pct"]:
            if abs(fval(r.get(col)) - fval(numeric_row.get(col))) > 1e-12:
                raise AssertionError(f"Numeric holdout source disagrees with Round261 {col} for {r['drug']}")
        row = {
            "drug": r["drug"],
            "n_holdout_pos": ival(r.get("n_holdout_pos")),
            "seen_evaluable": int(r["drug"] in seen_drugs),
            "best_rank": ival(numeric_row.get("best_rank")),
            "best_percentile": fval(numeric_row.get("best_percentile")),
            "best_percentile_pct": fval(numeric_row.get("best_percentile_pct")),
            "all_positive_ranks": numeric_row.get("all_positive_ranks", ""),
            "threshold_interval": numeric_row.get("threshold_interval", ""),
            "holdout_top1pct": fval(r.get("holdout_top1pct")),
            "holdout_top5pct": fval(r.get("holdout_top5pct")),
            "holdout_top10pct": fval(r.get("holdout_top10pct")),
        }
        row["n_recovered_thresholds"] = sum(
            1
            for key in ["holdout_top1pct", "holdout_top5pct", "holdout_top10pct"]
            if float(row[key]) >= 0.5
        )
        row["best_recovery_band"] = recovery_band(row)
        holdout_detail.append(row)
    holdout_detail = sorted(
        holdout_detail,
        key=lambda r: (float(r["best_percentile_pct"]), str(r["drug"])),
    )

    denominator = [
        {
            "analysis_set": "scored proteomic test drugs",
            "n_drugs": len(per_drug),
            "candidate_space": L_FULL,
            "evaluable_rule": "score vector available after fixed-space inference",
            "n_not_evaluable_from_scored": 0,
            "role": "starting set",
        },
        {
            "analysis_set": "seen-target diagnostic",
            "n_drugs": len(seen_eval),
            "candidate_space": L_SEEN,
            "evaluable_rule": "at least one seen positive target in the proteomic 102-label space",
            "n_not_evaluable_from_scored": len(per_drug) - len(seen_eval),
            "role": "secondary diagnostic",
        },
        {
            "analysis_set": "holdout-target primary",
            "n_drugs": len(holdout_eval),
            "candidate_space": L_FULL,
            "evaluable_rule": "at least one held-out positive target in the 16,583-gene space",
            "n_not_evaluable_from_scored": len(per_drug) - len(holdout_eval),
            "role": "primary transfer evidence",
        },
        {
            "analysis_set": "seen-holdout overlap",
            "n_drugs": len(overlap),
            "candidate_space": "",
            "evaluable_rule": "drugs evaluable for both endpoint families",
            "n_not_evaluable_from_scored": "",
            "role": "denominator audit",
        },
    ]

    audit = [
        {"field": "scored_test_drugs", "value": len(per_drug)},
        {"field": "seen_target_evaluable_drugs", "value": len(seen_eval)},
        {"field": "holdout_evaluable_drugs", "value": len(holdout_eval)},
        {"field": "seen_holdout_overlap_drugs", "value": len(overlap)},
        {"field": "seen_candidate_universe", "value": L_SEEN},
        {"field": "full_gene_universe", "value": L_FULL},
        {"field": "seen_Hit@1", "value": seen_hitk[0]["MeTar"]},
        {"field": "seen_Hit@5", "value": seen_hitk[1]["MeTar"]},
        {"field": "seen_Hit@10", "value": seen_hitk[2]["MeTar"]},
        {"field": "seen_NDCG@10", "value": ranking_quality[0]["MeTar"]},
        {"field": "seen_MAP@10", "value": ranking_quality[1]["MeTar"]},
        {"field": "holdout_top1pct", "value": holdout[0]["MeTar"]},
        {"field": "holdout_top5pct", "value": holdout[1]["MeTar"]},
        {"field": "holdout_top10pct", "value": holdout[2]["MeTar"]},
        {"field": "holdout_top5pct_random", "value": holdout[1]["matched_random"]},
    ]

    return {
        "seen_hitk": seen_hitk,
        "ranking_quality": ranking_quality,
        "holdout": holdout,
        "holdout_detail": holdout_detail,
        "denominator": denominator,
        "audit": audit,
    }



def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-drug",type=Path,required=True)
    parser.add_argument("--holdout-percentiles",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    tables=derive_tables(read_tsv(args.per_drug),read_tsv(args.holdout_percentiles))
    for name,rows in tables.items():
        if name!="audit" and rows:
            write_tsv(args.output_dir / f"proteomics_{name}.tsv",rows,list(rows[0]))


if __name__ == "__main__":
    main()
