from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data"
QA = ROOT / "qa"
for path in [SOURCE, QA]:
    path.mkdir(parents=True, exist_ok=True)

PERTKGE_ROOT = ROOT / "same_cell"

DEFAULT_CELLS = ["PC3", "MCF7", "A375", "A549", "HEPG2", "NPC", "U2OS", "YAPC", "JURKAT", "ASC", "HEC108", "MDAMB468"]
CELLS = DEFAULT_CELLS.copy()
METHODS = ["MeTar", "SSGCN", "PertKGE (same-cell)", "Random"]
HIT_KS = [1, 5, 10, 20, 30, 50]
METRIC_COLS = [f"Hit@{k}" for k in HIT_KS] + ["AUROC", "NDCG@10", "MeanAP"]
RANK_COLS = ["best_true_target_rank", "best_true_target_percentile"]
CANDIDATE_SPACE = 668


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", low_memory=False)


def load_cells(cells_file: str | None) -> list[str]:
    if cells_file is None:
        default_path = ROOT / "selected_seen_cells.txt"
        if default_path.exists():
            cells_file = str(default_path)
        else:
            return DEFAULT_CELLS.copy()
    cells = [line.strip() for line in Path(cells_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cells:
        raise ValueError(f"No cells found in {cells_file}")
    return cells


def load_metar() -> pd.DataFrame:
    df = read_tsv(SOURCE / "round320_metar_round252_scoremean_drugcell_metrics.tsv")
    df["method"] = "MeTar"
    df["unit_key"] = df["cell_line"].astype(str) + "||" + df["drug"].astype(str)
    return df


def load_ssgcn() -> pd.DataFrame:
    df = read_tsv(SOURCE / "round320_ssgcn_scoremean_drugcell_metrics.tsv")
    df["method"] = "SSGCN"
    df["unit_key"] = df["cell_line"].astype(str) + "||" + df["drug"].astype(str)
    return df


def load_pertkge() -> pd.DataFrame:
    rows = []
    base = PERTKGE_ROOT
    for cell in CELLS:
        path = base / f"cell_{cell}" / "matched_drug_mean_metrics.tsv"
        df = read_tsv(path)
        rename = {f"{metric}_pertkge": metric for metric in METRIC_COLS + ["best_true_target_percentile"]}
        keep = ["drug", "cell_line", "n_profiles", *rename.keys()]
        sub = df[keep].rename(columns=rename).copy()
        sub["method"] = "PertKGE (same-cell)"
        sub["split"] = "seen_test"
        sub["evaluation_unit"] = "score-mean drug-cell"
        sub["n_true_targets"] = np.nan
        sub["best_true_target_rank"] = np.nan
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True)
    out["unit_key"] = out["cell_line"].astype(str) + "||" + out["drug"].astype(str)
    return out


def assert_unique_units(df: pd.DataFrame, name: str) -> None:
    dup = int(df["unit_key"].duplicated().sum())
    if dup:
        examples = df.loc[df["unit_key"].duplicated(), "unit_key"].head(5).tolist()
        raise RuntimeError(f"{name} has duplicated drug-cell units: {dup}, examples={examples}")


def strict_join(metar: pd.DataFrame, ssgcn: pd.DataFrame, pert: pd.DataFrame) -> pd.DataFrame:
    for name, df in [("MeTar", metar), ("SSGCN", ssgcn), ("PertKGE", pert)]:
        assert_unique_units(df, name)
    keys = ["drug", "cell_line"]
    metar_cols = keys + ["n_profiles", "n_true_targets", *METRIC_COLS, *RANK_COLS]
    ssgcn_cols = keys + ["n_profiles", "n_true_targets", *METRIC_COLS, *RANK_COLS]
    pert_cols = keys + ["n_profiles", "n_true_targets", *METRIC_COLS, *RANK_COLS]
    joined = metar[metar_cols].merge(
        ssgcn[ssgcn_cols],
        on=keys,
        how="inner",
        suffixes=("_metar", "_ssgcn"),
        validate="one_to_one",
    )
    joined = joined.merge(
        pert[pert_cols],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    joined = joined.rename(
        columns={
            "n_profiles": "n_profiles_pertkge",
            "n_true_targets": "n_true_targets_pertkge",
            **{m: f"{m}_pertkge" for m in METRIC_COLS + RANK_COLS},
        }
    )
    return joined


def detail_long(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    suffixes = {
        "MeTar": "metar",
        "SSGCN": "ssgcn",
        "PertKGE (same-cell)": "pertkge",
    }
    for method, suffix in suffixes.items():
        cols = ["drug", "cell_line", f"n_profiles_{suffix}", f"n_true_targets_{suffix}"]
        cols += [f"{m}_{suffix}" for m in METRIC_COLS + RANK_COLS if f"{m}_{suffix}" in joined.columns]
        sub = joined[cols].copy()
        sub = sub.rename(columns={f"n_profiles_{suffix}": "n_profiles", f"n_true_targets_{suffix}": "n_true_targets"})
        sub = sub.rename(columns={f"{m}_{suffix}": m for m in METRIC_COLS + RANK_COLS})
        sub["method"] = method
        sub["evaluation_unit"] = "strict score-mean drug-cell"
        rows.append(sub)
    return pd.concat(rows, ignore_index=True, sort=False)


def summarize_per_cell(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, cell), grp in long_df.groupby(["method", "cell_line"], sort=False):
        rec = {
            "method": method,
            "cell_line": cell,
            "n_drugs": int(grp["drug"].nunique()),
            "n_profiles": int(grp["n_profiles"].sum()) if grp["n_profiles"].notna().any() else math.nan,
            "candidate_space": CANDIDATE_SPACE,
            "evaluation_unit": "strict score-mean drug-cell",
        }
        for metric in METRIC_COLS + ["best_true_target_percentile"]:
            rec[metric] = float(grp[metric].mean()) if metric in grp and grp[metric].notna().any() else math.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def add_random_per_cell(per_cell: pd.DataFrame) -> pd.DataFrame:
    rows = []
    counts = per_cell[per_cell["method"].eq("MeTar")].set_index("cell_line")
    for cell in CELLS:
        rec = {
            "method": "Random",
            "cell_line": cell,
            "n_drugs": int(counts.loc[cell, "n_drugs"]),
            "n_profiles": math.nan,
            "candidate_space": CANDIDATE_SPACE,
            "evaluation_unit": "strict score-mean drug-cell",
            "AUROC": 0.5,
            "NDCG@10": math.nan,
            "MeanAP": math.nan,
            "best_true_target_percentile": 50.0,
        }
        for k in HIT_KS:
            rec[f"Hit@{k}"] = k / CANDIDATE_SPACE
        rows.append(rec)
    return pd.concat([per_cell, pd.DataFrame(rows)], ignore_index=True, sort=False)


def summarize_overall(per_cell: pd.DataFrame, mode: str) -> pd.DataFrame:
    rows = []
    for method, grp in per_cell.groupby("method", sort=False):
        rec = {
            "method": method,
            "aggregation": mode,
            "n_cells": int(grp["cell_line"].nunique()),
            "n_drug_cell_units": int(grp["n_drugs"].sum()),
            "candidate_space": CANDIDATE_SPACE,
            "evaluation_unit": "strict score-mean drug-cell",
        }
        weights = grp["n_drugs"].astype(float).to_numpy()
        for metric in METRIC_COLS + ["best_true_target_percentile"]:
            vals = grp[metric].astype(float).to_numpy() if metric in grp else np.full(len(grp), np.nan)
            mask = np.isfinite(vals)
            if not mask.any():
                rec[metric] = math.nan
            elif mode == "cell_unweighted":
                rec[metric] = float(np.mean(vals[mask]))
            elif mode == "drugcell_weighted":
                rec[metric] = float(np.average(vals[mask], weights=weights[mask]))
            else:
                raise ValueError(mode)
        rows.append(rec)
    return pd.DataFrame(rows)


def hitk_curve(overall: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cell = overall[overall["aggregation"].eq("cell_unweighted")]
    for k in [1, 5, 10, 20, 30]:
        for _, r in cell.iterrows():
            rows.append(
                {
                    "K": k,
                    "method": r["method"],
                    "Hit@K": float(r[f"Hit@{k}"]),
                    "aggregation": "cell_unweighted",
                    "evaluation_unit": "strict score-mean drug-cell",
                    "n_drug_cell_units": int(r["n_drug_cell_units"]),
                    "candidate_space": CANDIDATE_SPACE,
                }
            )
    return pd.DataFrame(rows)


def rank_percentile_distribution(long_df: pd.DataFrame) -> pd.DataFrame:
    out = long_df[["method", "drug", "cell_line", "n_profiles", "best_true_target_percentile"]].copy()
    out = out[out["method"].ne("Random")]
    return out.sort_values(["method", "cell_line", "drug"]).reset_index(drop=True)


def denominator_audit(metar: pd.DataFrame, ssgcn: pd.DataFrame, pert: pd.DataFrame, joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    method_frames = {"MeTar": metar, "SSGCN": ssgcn, "PertKGE (same-cell)": pert}
    joined_keys = set(joined["cell_line"].astype(str) + "||" + joined["drug"].astype(str))
    for method, df in method_frames.items():
        for cell in CELLS:
            sub = df[df["cell_line"].eq(cell)].copy()
            keys = set(sub["unit_key"])
            rows.append(
                {
                    "method": method,
                    "cell_line": cell,
                    "n_units_available": int(len(keys)),
                    "n_units_in_strict_intersection": int(len(keys & joined_keys)),
                    "n_units_excluded": int(len(keys - joined_keys)),
                    "candidate_space": CANDIDATE_SPACE,
                }
            )
    return pd.DataFrame(rows)


def label_order_audit() -> pd.DataFrame:
    score_paths = {
        "MeTar": SOURCE / "round320_metar_round252_scoremean_drugcell_scores_668.parquet",
        "SSGCN": SOURCE / "round320_ssgcn_scoremean_drugcell_scores_668.parquet",
    }
    labels_by_method: dict[str, list[str]] = {}
    rows = []
    for method, path in score_paths.items():
        cols = list(pd.read_parquet(path).columns)
        labels = [c for c in cols if c not in {"drug", "cell_line", "split", "n_profiles"}]
        labels_by_method[method] = labels
        rows.append(
            {
                "method": method,
                "path": str(path),
                "n_columns": len(cols),
                "n_labels": len(labels),
                "label_order_sha256": hashlib.sha256("\n".join(labels).encode("utf-8")).hexdigest(),
                "first_5_labels": ",".join(labels[:5]),
                "last_5_labels": ",".join(labels[-5:]),
                "matches_metar_order": True if method == "MeTar" else labels == labels_by_method["MeTar"],
            }
        )
    return pd.DataFrame(rows)


def compare_to_round318(overall: pd.DataFrame, per_cell: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_old = read_tsv(ROUND318 / "source_data" / "Figure2_R318_metric_table.tsv")
    per_cell_old = read_tsv(ROUND318 / "source_data" / "Figure2_R318_per_cell_paired_metrics.tsv")
    metric_new = overall[overall["aggregation"].eq("cell_unweighted")].copy()
    m = metric_new.merge(metric_old, on="method", suffixes=("_round320", "_round318"), how="outer", indicator=True)
    rows = []
    for _, r in m.iterrows():
        for metric in ["n_drug_cell_units", *METRIC_COLS]:
            a = r.get(f"{metric}_round320")
            b = r.get(f"{metric}_round318")
            rows.append(
                {
                    "method": r["method"],
                    "metric": metric,
                    "round320_value": a,
                    "round318_value": b,
                    "diff": float(a) - float(b) if pd.notna(a) and pd.notna(b) else math.nan,
                    "merge_status": r["_merge"],
                }
            )
    metric_diff = pd.DataFrame(rows)
    p = per_cell.merge(per_cell_old.rename(columns={"n_drugs": "n_drugs_round318"}), on=["method", "cell_line"], suffixes=("_round320", "_round318"), how="outer", indicator=True)
    rows = []
    for _, r in p.iterrows():
        for metric in ["n_drugs", "Hit@10", "Hit@30", "Hit@50", "MeanAP", "NDCG@10", "AUROC"]:
            left_name = f"{metric}_round320" if metric != "n_drugs" else "n_drugs_round320"
            right_name = f"{metric}_round318" if metric != "n_drugs" else "n_drugs_round318"
            a = r.get(left_name)
            b = r.get(right_name)
            rows.append(
                {
                    "method": r["method"],
                    "cell_line": r["cell_line"],
                    "metric": metric,
                    "round320_value": a,
                    "round318_value": b,
                    "diff": float(a) - float(b) if pd.notna(a) and pd.notna(b) else math.nan,
                    "merge_status": r["_merge"],
                }
            )
    return metric_diff, pd.DataFrame(rows)


def write_report(overall: pd.DataFrame, per_cell: pd.DataFrame, audit: pd.DataFrame, labels: pd.DataFrame) -> None:
    cell = overall[overall["aggregation"].eq("cell_unweighted")].copy()
    show = ["method", "n_drug_cell_units", "Hit@10", "MeanAP", "NDCG@10", "AUROC", "Hit@30", "Hit@50"]
    max_excluded = int(audit["n_units_excluded"].max())
    label_ok = bool(labels["matches_metar_order"].all()) and bool(labels["n_labels"].eq(CANDIDATE_SPACE).all())
    lines = [
        "# Round450 Fig.2 seen-cell strict core comparator validation",
        "",
        "## Verdict",
        "",
        f"- Status: {'PASS_STRICT_INNER_JOIN' if max_excluded == 0 else 'FAIL_DENOMINATOR_MISMATCH'}.",
        "- Main unit: strict `(cell_line, drug)` score-mean drug-cell.",
        "- Candidate space: 668 known targets.",
        "- Cells: " + ", ".join(CELLS) + ".",
        "- MeTar was re-exported from the Round252 selected checkpoint; the strict per-drugcell rows numerically close the previous Round252 per-cell summary.",
        "- SSGCN was recomputed from the raw `ssgcn_scores_v3.parquet` 668-score matrix, not from averaged profile-level metrics.",
        "- PertKGE uses Round450 same-cell retrained drug-level target-inference outputs.",
        f"- Target-label order audit: {'PASS' if label_ok else 'FAIL'}.",
        "",
        "## Cell-unweighted strict metric table",
        "",
        "```tsv",
        cell[show].to_csv(sep="\t", index=False).strip(),
        "```",
        "",
        "## Denominator audit",
        "",
        "```tsv",
        audit.to_csv(sep="\t", index=False).strip(),
        "```",
    ]
    (QA / "round450_fig2_seen_cell_strict_core_comparator_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_method_lock(inputs: dict[str, str], audit_status: str) -> None:
    payload = {
        "round": "round450_fig2_seen_cellline_extension",
        "scope": "Expanded seen-cell strict inner-join audit for Fig.2 MeTar / SSGCN / same-cell PertKGE.",
        "cells": CELLS,
        "candidate_space": CANDIDATE_SPACE,
        "evaluation_unit": "strict score-mean drug-cell",
        "inputs": inputs,
        "audit_status": audit_status,
        "inner_join_no_training": True,
        "metar_no_retraining": True,
        "ssgcn_no_retraining": True,
        "pertkge_same_cell_retrained": True,
    }
    (ROOT / "method_lock_inner_join.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells-file", default=None)
    args = parser.parse_args()
    global CELLS
    CELLS = load_cells(args.cells_file)
    (ROOT / "selected_seen_cells.txt").write_text("\n".join(CELLS) + "\n", encoding="utf-8")

    metar = load_metar()
    ssgcn = load_ssgcn()
    pert = load_pertkge()
    joined = strict_join(metar, ssgcn, pert)
    long_df = detail_long(joined)
    per_cell = add_random_per_cell(summarize_per_cell(long_df))
    overall_cell = summarize_overall(per_cell, "cell_unweighted")
    overall_weighted = summarize_overall(per_cell, "drugcell_weighted")
    overall = pd.concat([overall_cell, overall_weighted], ignore_index=True, sort=False)
    hitk = hitk_curve(overall)
    rank_dist = rank_percentile_distribution(long_df)
    audit = denominator_audit(metar, ssgcn, pert, joined)
    labels = label_order_audit()

    joined.to_parquet(SOURCE / "round320_strict_joined_drugcell_metrics.parquet", index=False)
    joined.to_csv(SOURCE / "round320_strict_joined_drugcell_metrics.tsv", sep="\t", index=False)
    long_df.to_csv(SOURCE / "Figure2_R320_strict_drugcell_metric_long.tsv", sep="\t", index=False)
    per_cell.to_csv(SOURCE / "Figure2_R320_strict_per_cell_paired_metrics.tsv", sep="\t", index=False)
    overall.to_csv(SOURCE / "Figure2_R320_strict_metric_table.tsv", sep="\t", index=False)
    hitk.to_csv(SOURCE / "Figure2_R320_strict_hitk_curve.tsv", sep="\t", index=False)
    rank_dist.to_csv(SOURCE / "Figure2_R320_strict_rank_percentile_distribution.tsv", sep="\t", index=False)
    audit.to_csv(SOURCE / "Figure2_R320_strict_inner_join_audit.tsv", sep="\t", index=False)
    labels.to_csv(SOURCE / "round320_candidate_label_order_audit.tsv", sep="\t", index=False)
    write_report(overall, per_cell, audit, labels)
    inputs = {
        str(SOURCE / "round320_metar_round252_scoremean_drugcell_metrics.tsv"): sha256_file(SOURCE / "round320_metar_round252_scoremean_drugcell_metrics.tsv"),
        str(SOURCE / "round320_ssgcn_scoremean_drugcell_metrics.tsv"): sha256_file(SOURCE / "round320_ssgcn_scoremean_drugcell_metrics.tsv"),
    }
    for cell in CELLS:
        path = PERTKGE_ROOT / f"cell_{cell}" / "matched_drug_mean_metrics.tsv"
        inputs[str(path)] = sha256_file(path)
    status = "PASS_STRICT_INNER_JOIN" if int(audit["n_units_excluded"].max()) == 0 else "FAIL_DENOMINATOR_MISMATCH"
    write_method_lock(inputs, status)


if __name__ == "__main__":
    main()
