#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score


ROUND = Path("/home/lwn/claude/manuscript/write/claude_round375_fig3_ssgcn_crosscell_replacement_20260609")
SOURCE = ROUND / "source_data"
QA = ROUND / "qa"
LOGS = ROUND / "logs"

METAR_ROOT = Path("/home/lwn/claude/manuscript/write")
ROUND252 = METAR_ROOT / "claude_round252_epiqgate_pseudoval_sweep_20260520"
ROUND309 = METAR_ROOT / "claude_round309_fig4_round307_metar_druglevel_20260601"
ROUND87 = METAR_ROOT / "unifiedlabel_figures_20260331" / "Figure2_r1"
SSGCN_SCORE_PATH = ROUND87 / "ssgcn_scores_v3.parquet"

METAR_RUN = METAR_ROOT / "bulk_sharedsample_indunseenlabel_20260330_h07_w060_u8e4_med"
LOCK_PATH = METAR_RUN / "manifests" / "bulk_protocol_lock_v2.json"
DATA_DIR = Path("/home/lwn/claude/bulk/data")
COMPOUNDINFO = DATA_DIR / "compoundinfo_beta.txt"

OFFICIAL_UNSEEN_CELLS = [
    "HCC515",
    "HELA",
    "HT115",
    "HT29",
    "HUES3",
    "LOVO",
    "NCIH1694",
    "NCIH841",
    "RMUGS",
    "SKLU1",
    "SNGM",
    "VCAP",
    "XC.P909",
    "XC.P910",
]
HIT_KS = [1, 5, 10, 20, 30, 50]
METRIC_COLS = [f"Hit@{k}" for k in HIT_KS] + ["AUROC", "NDCG@10", "MeanAP"]
CANDIDATE_SPACE = 668


def ensure_dirs() -> None:
    for path in [ROUND, SOURCE, QA, LOGS]:
        path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def split_targets(value: str) -> Iterable[str]:
    for gene in str(value).split("|"):
        gene = gene.strip()
        if gene and gene not in {"0", "-666", "nan", "None"}:
            yield gene


def load_target_context() -> tuple[list[str], dict[str, list[str]], dict[str, Any]]:
    protocol = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    inner_train = set(protocol["drug_split"]["inner_train_drugs"])
    cinfo = pd.read_csv(COMPOUNDINFO, sep="\t")
    cinfo = cinfo[cinfo["target"].notna() & ~cinfo["target"].isin(["0", "", "-666"])]
    labels = sorted({g for t in cinfo[cinfo["pert_id"].isin(inner_train)]["target"].dropna() for g in split_targets(t)})
    if len(labels) != CANDIDATE_SPACE or len(set(labels)) != CANDIDATE_SPACE:
        raise RuntimeError(f"Expected {CANDIDATE_SPACE} labels, observed {len(labels)} / {len(set(labels))}")
    target_map: dict[str, list[str]] = {}
    for drug, grp in cinfo.groupby("pert_id"):
        targets = sorted({g for t in grp["target"].astype(str) for g in split_targets(t)})
        if targets:
            target_map[str(drug)] = targets
    return labels, target_map, protocol


def binary_auroc(y: np.ndarray, score: np.ndarray) -> float:
    n_pos = int(y.sum())
    if n_pos <= 0 or n_pos >= len(y):
        return 0.5
    return float(roc_auc_score(y, score))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    if int(y.sum()) <= 0:
        return 0.0
    return float(average_precision_score(y, score))


def ndcg10(y: np.ndarray, score: np.ndarray) -> float:
    if int(y.sum()) <= 0:
        return 0.0
    return float(ndcg_score([y], [score], k=10))


def metric_records_from_score_df(scores_df: pd.DataFrame, labels: list[str], target_map: dict[str, list[str]], method: str) -> pd.DataFrame:
    label_set = set(labels)
    rows = []
    for _, row in scores_df.iterrows():
        drug = str(row["drug"])
        true_targets = set(target_map.get(drug, [])) & label_set
        if not true_targets:
            continue
        scores = row[labels].to_numpy(dtype=np.float64)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        order = np.argsort(scores)[::-1]
        y = np.array([1 if labels[i] in true_targets else 0 for i in range(len(labels))], dtype=np.int8)
        pos_idx = set(np.where(y > 0)[0].tolist())
        best_rank = min(int(np.where(order == idx)[0][0]) + 1 for idx in pos_idx)
        rec = {
            "method": method,
            "split": "official_test",
            "evaluation_unit": "score-mean drug-cell",
            "drug": drug,
            "cell_line": str(row["cell_line"]),
            "n_profiles": int(row.get("n_profiles", 1)),
            "n_true_targets": int(y.sum()),
            "best_true_target_rank": int(best_rank),
            "best_true_target_percentile": 100.0 * (1.0 - float(best_rank) / float(len(labels))),
        }
        for k in HIT_KS:
            rec[f"Hit@{k}"] = int(bool(set(order[:k]) & pos_idx))
        rec["AUROC"] = binary_auroc(y, scores)
        rec["NDCG@10"] = ndcg10(y, scores)
        rec["MeanAP"] = average_precision(y, scores)
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_per_cell(per: pd.DataFrame, method: str) -> pd.DataFrame:
    rows = []
    for cell, grp in per.groupby("cell_line", sort=False):
        rec = {
            "method": method,
            "method_label": method,
            "cell_line": cell,
            "split": "official_test",
            "evaluation_unit": "score-mean drug-cell",
            "n_groups": int(len(grp)),
            "n_drugs": int(grp["drug"].nunique()),
            "n_profiles": int(grp["n_profiles"].sum()),
            "candidate_space": CANDIDATE_SPACE,
        }
        for metric in METRIC_COLS + ["best_true_target_percentile"]:
            rec[metric] = float(grp[metric].mean()) if metric in grp else float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_cell_unweighted(cells: pd.DataFrame, method: str) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "method": method,
        "variant": "ssgcn_scores_v3",
        "split": "official_test",
        "evaluation_unit": "score-mean drug-cell",
        "aggregation_scope": "cell_unweighted_mean",
        "denominator_scope": "full_round252_official",
        "candidate_space": CANDIDATE_SPACE,
        "n_cells": int(len(cells)),
        "n_groups": int(cells["n_groups"].sum()),
        "n_drugs": int(cells["n_drugs"].sum()),
        "n_profiles": int(cells["n_profiles"].sum()),
        "comparison_mode": "score_mean",
        "method_label": method,
    }
    for metric in METRIC_COLS + ["best_true_target_percentile"]:
        rec[metric] = float(cells[metric].mean())
    return rec


def score_matrix_by_drugcell(profile_scores: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    sub = profile_scores[profile_scores["cell_line"].isin(OFFICIAL_UNSEEN_CELLS)].copy()
    grouped = sub.groupby(["drug", "cell_line"], sort=False)
    score_mean = grouped[labels].mean().reset_index()
    counts = grouped["profile_id"].count().reset_index(name="n_profiles")
    out = counts.merge(score_mean, on=["drug", "cell_line"], how="inner", validate="one_to_one")
    out["split"] = "official_test"
    return out[["drug", "cell_line", "split", "n_profiles", *labels]]


def export_ssgcn(labels: list[str], target_map: dict[str, list[str]], write_scores: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    use_cols = ["profile_id", "drug", "cell_line", *labels]
    print("[round375] loading raw SSGCN score cache", flush=True)
    raw = pd.read_parquet(
        SSGCN_SCORE_PATH,
        columns=use_cols,
        filters=[("cell_line", "in", OFFICIAL_UNSEEN_CELLS)],
    )
    missing = [c for c in labels if c not in raw.columns]
    if missing:
        raise RuntimeError(f"SSGCN score cache missing labels: {missing[:5]}")
    matrix = score_matrix_by_drugcell(raw, labels)
    per = metric_records_from_score_df(matrix, labels, target_map, "SSGCN")
    cells = summarize_per_cell(per, "SSGCN")
    if write_scores:
        matrix.to_parquet(SOURCE / "round375_ssgcn_official_scoremean_drugcell_scores_668.parquet", index=False)
    per.to_csv(SOURCE / "round375_ssgcn_official_scoremean_drugcell_metrics.tsv", sep="\t", index=False)
    cells.to_csv(SOURCE / "round375_ssgcn_official_scoremean_per_cell_metrics.tsv", sep="\t", index=False)
    pd.DataFrame([summarize_cell_unweighted(cells, "SSGCN")]).to_csv(
        SOURCE / "round375_ssgcn_official_scoremean_summary.tsv", sep="\t", index=False
    )
    audit = {
        "raw_profile_rows_official_unseen": int(len(raw)),
        "scoremean_drugcell_units": int(len(per)),
        "official_unseen_cells": sorted(cells["cell_line"].tolist()),
        "per_cell_counts": cells[["cell_line", "n_groups", "n_profiles"]].to_dict(orient="records"),
    }
    return per, cells, audit


def write_method_lock(extra: dict[str, Any]) -> None:
    payload = {
        "round": "round375_fig3_ssgcn_crosscell_replacement",
        "scope": "Export SSGCN score-mean drug-cell official-unseen metrics for Fig.3 unknown-cell known-target comparison.",
        "official_unseen_cells": OFFICIAL_UNSEEN_CELLS,
        "candidate_space": CANDIDATE_SPACE,
        "evaluation_unit": "score-mean drug-cell",
        "inputs": {
            "ssgcn_score_cache": {"path": SSGCN_SCORE_PATH, "sha256": sha256_file(SSGCN_SCORE_PATH)},
            "protocol_lock": {"path": LOCK_PATH, "sha256": sha256_file(LOCK_PATH)},
            "compoundinfo": {"path": COMPOUNDINFO, "sha256": sha256_file(COMPOUNDINFO)},
            "round309_source": {"path": ROUND309 / "source_data"},
            "round252_source": {"path": ROUND252 / "source_data"},
        },
        "extra": extra,
    }
    (ROUND / "method_lock.json").write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")
    (SOURCE / "method_lock.json").write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-scores", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    labels, target_map, protocol = load_target_context()
    if len(labels) != CANDIDATE_SPACE:
        raise RuntimeError("candidate-space audit failed")
    per, cells, audit = export_ssgcn(labels, target_map, bool(args.write_scores))
    extra = {
        "python": str(Path().absolute()),
        "pandas": pd.__version__,
        "n_labels": len(labels),
        "n_target_map_drugs": len(target_map),
        "protocol_train_cells": len(protocol["cell_split"]["train_cells"]),
        "ssgcn_units": int(len(per)),
        "audit": audit,
    }
    write_method_lock(extra)
    print("[round375] export complete", flush=True)


if __name__ == "__main__":
    main()
