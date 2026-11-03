#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score


ROUND_ROOT = Path("/home/lwn/claude/manuscript/write/claude_round450_fig2_seen_cellline_extension_20260702")
SOURCE = ROUND_ROOT / "source_data"
QA = ROUND_ROOT / "qa"
LOGS = ROUND_ROOT / "logs"

METAR_ROOT = Path("/home/lwn/claude/manuscript/write")
ROUND252 = METAR_ROOT / "claude_round252_epiqgate_pseudoval_sweep_20260520"
ROUND252_SCRIPT = ROUND252 / "scripts" / "run_round252_epiqgate_pseudoval_sweep.py"
ROUND252_ADAPTER_CKPT = ROUND252 / "checkpoints" / "r252_epiqgate_lm1_lr0p00015_sc0p18_mn2p5_ba0p1_s32_q32_diverse_label_seed20260524" / "adapter_checkpoint.pth"

ROUND87 = METAR_ROOT / "unifiedlabel_figures_20260331" / "Figure2_r1"
SSGCN_SCORE_PATH = ROUND87 / "ssgcn_scores_v3.parquet"
METAR_OLD_SCORE_PATH = ROUND87 / "metar_scores_v4.parquet"

METAR_RUN = METAR_ROOT / "bulk_sharedsample_indunseenlabel_20260330_h07_w060_u8e4_med"
LOCK_PATH = METAR_RUN / "manifests" / "bulk_protocol_lock_v2.json"
DATA_DIR = Path("/home/lwn/claude/bulk/data")
COMPOUNDINFO = DATA_DIR / "compoundinfo_beta.txt"

DEFAULT_CELLS = ["PC3", "MCF7", "A375", "A549", "HEPG2", "NPC", "U2OS", "YAPC", "JURKAT", "ASC", "HEC108", "MDAMB468"]
CELLS = DEFAULT_CELLS.copy()
HIT_KS = [1, 5, 10, 20, 30, 50]
METRIC_COLS = [f"Hit@{k}" for k in HIT_KS] + ["AUROC", "NDCG@10", "MeanAP"]
CANDIDATE_SPACE = 668
SELECTED_VARIANT = "r252_epiqgate_lm1_lr0p00015_sc0p18_mn2p5_ba0p1_s32_q32_diverse_label_seed20260524_seen_diagnostic_diverse_label_s64"
SUPPORT_SIZE = 64
SUPPORT_SELECTOR = "diverse_label"


def ensure_dirs() -> None:
    for path in [ROUND_ROOT, SOURCE, QA, LOGS]:
        path.mkdir(parents=True, exist_ok=True)


def load_cells(cells_file: str | None) -> list[str]:
    if cells_file is None:
        return DEFAULT_CELLS.copy()
    cells = [line.strip() for line in Path(cells_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cells:
        raise ValueError(f"No cells found in {cells_file}")
    return cells


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def import_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def metric_records_from_score_df(scores_df: pd.DataFrame, labels: list[str], target_map: dict[str, list[str]], method: str, split: str) -> pd.DataFrame:
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
            "split": split,
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
            "cell_line": cell,
            "split": str(grp["split"].iloc[0]),
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


def score_matrix_by_drugcell(profile_scores: pd.DataFrame, labels: list[str], cells: list[str]) -> pd.DataFrame:
    sub = profile_scores[profile_scores["cell_line"].isin(cells)].copy()
    grouped = sub.groupby(["drug", "cell_line"], sort=False)
    score_mean = grouped[labels].mean().reset_index()
    counts = grouped["profile_id"].count().reset_index(name="n_profiles")
    out = counts.merge(score_mean, on=["drug", "cell_line"], how="inner", validate="one_to_one")
    out["split"] = "seen_test"
    return out[["drug", "cell_line", "split", "n_profiles", *labels]]


def export_ssgcn(labels: list[str], target_map: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("[round450] loading raw SSGCN score cache", flush=True)
    use_cols = ["profile_id", "drug", "cell_line", *labels]
    raw = pd.read_parquet(SSGCN_SCORE_PATH, columns=use_cols)
    missing = [c for c in labels if c not in raw.columns]
    if missing:
        raise RuntimeError(f"SSGCN score cache missing labels: {missing[:5]}")
    matrix = score_matrix_by_drugcell(raw, labels, CELLS)
    per = metric_records_from_score_df(matrix, labels, target_map, "SSGCN", "seen_test")
    cells = summarize_per_cell(per, "SSGCN")
    matrix.to_parquet(SOURCE / "round320_ssgcn_scoremean_drugcell_scores_668.parquet", index=False)
    per.to_csv(SOURCE / "round320_ssgcn_scoremean_drugcell_metrics.tsv", sep="\t", index=False)
    cells.to_csv(SOURCE / "round320_ssgcn_scoremean_drugcell_per_cell.tsv", sep="\t", index=False)
    return per, cells


def build_round252_args(r252) -> argparse.Namespace:
    old_argv = sys.argv[:]
    try:
        sys.argv = ["round252_export_only"]
        args = r252.parse_args()
    finally:
        sys.argv = old_argv
    args.device = "cuda"
    args.phase = "finalize_existing"
    args.support_checkpoint = None
    args.round247_adapter_checkpoint = None
    args.report_support_sizes = [SUPPORT_SIZE]
    args.report_selectors = [SUPPORT_SELECTOR]
    args.seen_diagnostic_cells = ",".join(CELLS)
    return args


def export_metar_round252(labels: list[str], target_map: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("[round450] loading Round252 selected model", flush=True)
    r252 = import_from_path("round450_round252", ROUND252_SCRIPT)
    r252.ROUND_ROOT = ROUND_ROOT
    r252.SOURCE = SOURCE
    r252.QA = QA
    r252.CHECKPOINTS = ROUND_ROOT / "checkpoints"
    r252.LOGS = LOGS

    r250 = r252.load_round250_module()
    r247 = r252.load_round247_module()
    args = build_round252_args(r252)
    r252.set_seed(int(args.run_seed))
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"[round450] Round252 export device={device}", flush=True)
    context = r252.load_context(r250, args, device)
    support_cells = sorted(str(c) for c in context["tables"]["seen_support"].get("by_cell", {}).keys())
    query_cells = sorted(str(c) for c in context["tables"]["seen_test"].get("by_cell", {}).keys())
    missing_context = sorted(set(CELLS) - (set(support_cells) & set(query_cells)))
    if missing_context:
        raise RuntimeError(f"Round252 expanded seen diagnostic context missing cells: {missing_context}")
    payload = torch.load(ROUND252_ADAPTER_CKPT, map_location="cpu", weights_only=False)
    selected = {"cfg": payload["cfg"], "checkpoint": str(ROUND252_ADAPTER_CKPT)}
    if selected["cfg"].get("eval_support_size") != SUPPORT_SIZE:
        raise RuntimeError("Selected Round252 eval_support_size no longer matches strict Fig.2 support size")
    if selected["cfg"].get("eval_selector") != SUPPORT_SELECTOR:
        raise RuntimeError("Selected Round252 eval_selector no longer matches strict Fig.2 selector")
    adapter = r252.new_query_adapter(r247, context["support_model"], selected["cfg"], device)
    adapter.load_state_dict(payload["adapter_state_dict"], strict=True)
    adapter.eval()

    support_item = context["tables"]["seen_support"]
    query_item = context["tables"]["seen_test"]
    model = context["support_model"]
    score_parts = []
    audit_rows = []
    with torch.no_grad():
        for cell in CELLS:
            qidx = np.asarray(query_item["by_cell"].get(cell, np.asarray([], dtype=np.int64)), dtype=np.int64)
            support_all = np.asarray(support_item["by_cell"].get(cell, np.asarray([], dtype=np.int64)), dtype=np.int64)
            if len(qidx) == 0 or len(support_all) == 0:
                raise RuntimeError(f"Missing Round252 seen support/query rows for {cell}")
            support_rows = support_item["rows"].iloc[support_all].reset_index(drop=True)
            support_y_np = support_item["y"][support_all]
            local = r252.select_support_indices(
                r247,
                support_rows,
                support_y_np,
                SUPPORT_SIZE,
                SUPPORT_SELECTOR,
                r252.stable_seed(r252.SEED, "r251", cell, SUPPORT_SELECTOR, SUPPORT_SIZE),
            )
            support_idx = support_all[local]
            qrows = query_item["rows"].iloc[qidx].reset_index(drop=True)
            overlap = r252.drug_overlap_count(r247, support_rows, qrows, local)
            if overlap:
                raise RuntimeError(f"Round252 support/query drug overlap for {cell}: {overlap}")
            support_q = r247.tensor_slice(support_item, "q", support_idx, device)
            support_y = r247.tensor_slice(support_item, "y", support_idx, device)
            support_w = r247.tensor_slice(support_item, "weights", support_idx, device)
            query_q = r247.tensor_slice(query_item, "q", qidx, device)
            delta, gate = adapter(query_q, support_q, support_y, support_w)
            scores = r252.score_with_query_delta(
                r247,
                model,
                query_q,
                delta,
                gate,
                float(selected["cfg"]["shift_scale"]),
            ).detach().cpu().numpy()
            score_df = pd.concat(
                [
                    qrows[["drug", "cell_line", "split", "n_profiles"]].reset_index(drop=True),
                    pd.DataFrame(scores.astype(np.float32), columns=labels),
                ],
                axis=1,
            )
            score_parts.append(score_df)
            audit_rows.append(
                {
                    "cell_line": cell,
                    "support_groups": int(len(support_all)),
                    "used_support_rows": int(len(support_idx)),
                    "query_groups": int(len(qidx)),
                    "support_query_drug_overlap": int(overlap),
                    "gate_mean": float(gate.mean().detach().cpu()),
                    "delta_norm_mean": float(torch.linalg.vector_norm(delta, dim=1).mean().detach().cpu()),
                }
            )
    matrix = pd.concat(score_parts, ignore_index=True)
    per = metric_records_from_score_df(matrix, labels, target_map, "MeTar", "seen_test")
    cells = summarize_per_cell(per, "MeTar")
    matrix.to_parquet(SOURCE / "round320_metar_round252_scoremean_drugcell_scores_668.parquet", index=False)
    per.to_csv(SOURCE / "round320_metar_round252_scoremean_drugcell_metrics.tsv", sep="\t", index=False)
    cells.to_csv(SOURCE / "round320_metar_round252_scoremean_drugcell_per_cell.tsv", sep="\t", index=False)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(SOURCE / "round320_metar_round252_support_audit.tsv", sep="\t", index=False)
    return per, cells, audit


def compare_metar_to_round252_summary(metar_cells: pd.DataFrame) -> pd.DataFrame:
    old = pd.read_csv(ROUND252 / "source_data" / "per_cell_metrics.tsv", sep="\t")
    old = old[
        old["cell_line"].isin(CELLS)
        & old["split"].eq("seen_test")
        & old["evaluation_unit"].eq("drug-mean")
        & old["method"].eq("Round252_EpiQGate")
        & old["variant"].eq(SELECTED_VARIANT)
    ].copy()
    merged = metar_cells.merge(old, on="cell_line", suffixes=("_strict", "_round252"), how="outer", indicator=True)
    rows = []
    for _, row in merged.iterrows():
        for metric in ["n_groups", "n_profiles", *METRIC_COLS]:
            left = row.get(f"{metric}_strict")
            right = row.get(f"{metric}_round252")
            rows.append(
                {
                    "cell_line": row["cell_line"],
                    "metric": metric,
                    "strict_value": left,
                    "round252_summary_value": right,
                    "diff": float(left) - float(right) if pd.notna(left) and pd.notna(right) else float("nan"),
                    "merge_status": row["_merge"],
                }
            )
    diff = pd.DataFrame(rows)
    diff.to_csv(SOURCE / "round320_metar_strict_vs_round252_summary_diff.tsv", sep="\t", index=False)
    return diff


def write_method_lock(extra: dict[str, Any]) -> None:
    payload = {
        "round": "round450_fig2_seen_cellline_extension",
        "scope": "Export expanded seen-cell strict score-mean drug-cell MeTar Round252 and SSGCN raw-score metrics for Fig.2 extension.",
        "cells": CELLS,
        "candidate_space": CANDIDATE_SPACE,
        "evaluation_unit": "score-mean drug-cell",
        "round252_selected_variant": SELECTED_VARIANT,
        "support_selector": SUPPORT_SELECTOR,
        "support_size": SUPPORT_SIZE,
        "inputs": {
            "round252_script": {"path": ROUND252_SCRIPT, "sha256": sha256_file(ROUND252_SCRIPT)},
            "adapter_configuration": "cfg stored with the selected adapter checkpoint",
            "round252_adapter_checkpoint": {"path": ROUND252_ADAPTER_CKPT, "sha256": sha256_file(ROUND252_ADAPTER_CKPT)},
            "ssgcn_score_cache": {"path": SSGCN_SCORE_PATH, "sha256": sha256_file(SSGCN_SCORE_PATH)},
            "old_metar_score_cache_not_used_for_main": {"path": METAR_OLD_SCORE_PATH, "sha256": sha256_file(METAR_OLD_SCORE_PATH)},
            "protocol_lock": {"path": LOCK_PATH, "sha256": sha256_file(LOCK_PATH)},
            "compoundinfo": {"path": COMPOUNDINFO, "sha256": sha256_file(COMPOUNDINFO)},
        },
        "extra": extra,
    }
    (ROUND_ROOT / "method_lock.json").write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")
    (SOURCE / "method_lock.json").write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells-file", default=None)
    parser.add_argument("--skip-metar", action="store_true")
    parser.add_argument("--skip-ssgcn", action="store_true")
    args = parser.parse_args()
    global CELLS
    CELLS = load_cells(args.cells_file)
    ensure_dirs()
    (ROUND_ROOT / "selected_seen_cells.txt").write_text("\n".join(CELLS) + "\n", encoding="utf-8")
    random.seed(20260602)
    np.random.seed(20260602)
    labels, target_map, protocol = load_target_context()
    if len(labels) != CANDIDATE_SPACE:
        raise RuntimeError("candidate-space audit failed")
    extra: dict[str, Any] = {
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "pandas": pd.__version__,
        "n_labels": len(labels),
        "n_target_map_drugs": len(target_map),
        "protocol_train_cells": len(protocol["cell_split"]["train_cells"]),
        "round252_seen_diagnostic_cells_requested": CELLS,
    }
    if not args.skip_ssgcn:
        ssgcn_per, ssgcn_cells = export_ssgcn(labels, target_map)
        extra["ssgcn_units"] = int(len(ssgcn_per))
        extra["ssgcn_cell_counts"] = ssgcn_cells[["cell_line", "n_groups", "n_profiles"]].to_dict(orient="records")
    if not args.skip_metar:
        metar_per, metar_cells, metar_audit = export_metar_round252(labels, target_map)
        diff = compare_metar_to_round252_summary(metar_cells)
        max_abs = diff["diff"].abs().max(skipna=True)
        extra["metar_units"] = int(len(metar_per))
        extra["metar_cell_counts"] = metar_cells[["cell_line", "n_groups", "n_profiles"]].to_dict(orient="records")
        extra["metar_max_abs_diff_vs_round252_summary"] = float(max_abs) if pd.notna(max_abs) else None
        extra["metar_support_audit"] = metar_audit.to_dict(orient="records")
    write_method_lock(extra)
    print("[round450] export complete", flush=True)


if __name__ == "__main__":
    main()
