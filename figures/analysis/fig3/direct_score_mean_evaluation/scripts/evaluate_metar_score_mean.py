#!/usr/bin/env python3
"""Round309 MeTar score-mean drug-cell evaluation for Fig.4.

This script does not train a model. It reloads the selected Round246 base model
and Round252 EpiQGate adapter, scores the official unseen profile rows, averages
668-target score vectors within each drug-cell unit, and recomputes metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - tested by py_compile only
    raise ModuleNotFoundError("torch is required; run this script on lwn or another MeTar torch environment.") from exc


ROUND_ROOT = Path(__file__).resolve().parents[1]
METAR_ROOT = ROUND_ROOT.parent
SOURCE = ROUND_ROOT / "source_data"
QA = ROUND_ROOT / "qa"
LOGS = ROUND_ROOT / "logs"

ROUND246_ROOT = METAR_ROOT / "claude_round246_joint_seen_target_train_pc3_a375_mcf7_20260519"
ROUND247_ROOT = METAR_ROOT / "claude_round247_support_curve_query_gated_adapter_20260520"
ROUND252_ROOT = METAR_ROOT / "claude_round252_epiqgate_pseudoval_sweep_20260520"
ROUND250_ROOT = METAR_ROOT / "claude_round250_epmeta_stable_tuning_20260520"

ROUND252_SCRIPT = ROUND252_ROOT / "scripts" / "run_round252_epiqgate_pseudoval_sweep.py"
ROUND247_SCRIPT = ROUND247_ROOT / "scripts" / "run_round247_support_curve_query_gated_adapter.py"
ROUND250_SCRIPT = ROUND250_ROOT / "scripts" / "run_round250_epmeta_stable_tuning.py"

ROUND246_NOADAPT_VARIANT = "r246_joint_lm0p5_sc0p15_mn2p5_official_unseen_no_adapt"
ROUND252_EPIQGATE_VARIANT = (
    "r252_epiqgate_lm1_lr0p00015_sc0p18_mn2p5_ba0p1_s32_q32_diverse_label_seed20260524"
    "_official_unseen_diverse_label_s64"
)
ROUND252_CORE_VARIANT = "r252_epiqgate_lm1_lr0p00015_sc0p18_mn2p5_ba0p1_s32_q32_diverse_label_seed20260524"
HIT_KS = [1, 5, 10, 20, 30, 50]
METRICS = [f"Hit@{k}" for k in HIT_KS] + ["AUROC", "NDCG@10", "MeanAP"]


def ensure_dirs() -> None:
    for path in [SOURCE, QA, LOGS]:
        path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def import_from_path(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_existing(path_text: str | None, local_root: Path, relative_tail: str) -> Path:
    candidates: list[Path] = []
    if path_text:
        raw = Path(path_text)
        candidates.append(raw)
        parts = raw.parts
        if "claude_round246_joint_seen_target_train_pc3_a375_mcf7_20260519" in parts:
            idx = parts.index("claude_round246_joint_seen_target_train_pc3_a375_mcf7_20260519")
            candidates.append(METAR_ROOT / Path(*parts[idx:]))
        if "claude_round252_epiqgate_pseudoval_sweep_20260520" in parts:
            idx = parts.index("claude_round252_epiqgate_pseudoval_sweep_20260520")
            candidates.append(METAR_ROOT / Path(*parts[idx:]))
        if "claude_round247_support_curve_query_gated_adapter_20260520" in parts:
            idx = parts.index("claude_round247_support_curve_query_gated_adapter_20260520")
            candidates.append(METAR_ROOT / Path(*parts[idx:]))
    candidates.append(local_root / relative_tail)
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError("No existing path among:\n" + "\n".join(map(str, candidates)))


def binary_auroc(y: np.ndarray, score: np.ndarray) -> float:
    y = y.astype(np.int8)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    return float((ranks[y > 0].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(score)[::-1]
    y_sorted = y[order].astype(np.float64)
    n_pos = y_sorted.sum()
    if n_pos <= 0:
        return float("nan")
    precision = np.cumsum(y_sorted) / (np.arange(len(y_sorted)) + 1.0)
    return float((precision * y_sorted).sum() / n_pos)


def ndcg_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float:
    order = np.argsort(score)[::-1][:k]
    gains = y[order].astype(np.float64)
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float((gains * discounts).sum())
    ideal = np.sort(y.astype(np.float64))[::-1][:k]
    idcg = float((ideal * discounts[: len(ideal)]).sum())
    return dcg / idcg if idcg > 0 else 0.0


def compute_unit_metrics(scores: np.ndarray, y: np.ndarray, rows: pd.DataFrame, method: str, variant: str, unit: str) -> pd.DataFrame:
    records = []
    for i in range(scores.shape[0]):
        s = np.nan_to_num(scores[i].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        yy = y[i].astype(np.int8)
        order = np.argsort(s)[::-1]
        positives = set(np.where(yy > 0)[0].tolist())
        rec = {
            "method": method,
            "variant": variant,
            "split": str(rows.iloc[i]["split"]),
            "evaluation_unit": unit,
            "drug": str(rows.iloc[i]["drug"]),
            "cell_line": str(rows.iloc[i]["cell_line"]),
            "n_profiles": int(rows.iloc[i]["n_profiles"]),
            "n_true_targets": int(yy.sum()),
        }
        for k in HIT_KS:
            rec[f"Hit@{k}"] = int(bool(set(order[:k]) & positives))
        rec["AUROC"] = binary_auroc(yy, s)
        rec["NDCG@10"] = ndcg_at_k(yy, s, 10)
        rec["MeanAP"] = average_precision(yy, s)
        rec["best_true_target_rank"] = int(min([np.where(order == p)[0][0] + 1 for p in positives])) if positives else -1
        rec["best_true_target_percentile"] = 100.0 * (1.0 - rec["best_true_target_rank"] / scores.shape[1]) if positives else float("nan")
        records.append(rec)
    return pd.DataFrame(records)


def summarize_from_per_cell(per_cell: pd.DataFrame, method: str, variant: str, split: str, unit: str, aggregation_scope: str) -> dict[str, Any]:
    out = {
        "method": method,
        "variant": variant,
        "split": split,
        "evaluation_unit": unit,
        "candidate_space": 668,
        "n_cells": int(per_cell["cell_line"].nunique()) if len(per_cell) else 0,
        "n_groups": int(per_cell["n_groups"].sum()) if "n_groups" in per_cell else int(len(per_cell)),
        "n_drugs": int(per_cell["n_drugs"].sum()) if "n_drugs" in per_cell else int(per_cell["drug"].nunique()) if len(per_cell) else 0,
        "n_profiles": int(per_cell["n_profiles"].sum()) if "n_profiles" in per_cell else 0,
        "aggregation_scope": aggregation_scope,
    }
    for metric in METRICS + ["best_true_target_percentile"]:
        out[metric] = float(per_cell[metric].mean()) if len(per_cell) and metric in per_cell else float("nan")
    return out


def per_cell_summary(per: pd.DataFrame, method: str, variant: str, unit: str) -> pd.DataFrame:
    rows = []
    for cell, grp in per.groupby("cell_line", sort=True):
        rec = {
            "method": method,
            "variant": variant,
            "split": str(grp["split"].iloc[0]) if len(grp) else "official_test",
            "evaluation_unit": unit,
            "candidate_space": 668,
            "n_cells": 1,
            "n_groups": int(len(grp)),
            "n_drugs": int(grp["drug"].nunique()),
            "n_profiles": int(grp["n_profiles"].sum()),
            "aggregation_scope": "scoremean_drugcell_per_cell",
            "cell_line": cell,
        }
        for metric in METRICS + ["best_true_target_percentile"]:
            rec[metric] = float(grp[metric].mean()) if metric in grp else float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)


def collapse_scores_to_drugcell(profile_scores: np.ndarray, profile_rows: pd.DataFrame, profile_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    profile_rows = profile_rows.reset_index(drop=True).copy()
    score_parts = []
    y_parts = []
    row_parts = []
    for (drug, cell), idx in profile_rows.groupby(["drug", "cell_line"], sort=True).indices.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        mean_score = profile_scores[idx_arr].mean(axis=0)
        y_rows = profile_y[idx_arr]
        if not np.all(y_rows == y_rows[0]):
            raise RuntimeError(f"Target labels differ within drug-cell unit: {drug}/{cell}")
        first = profile_rows.iloc[idx_arr[0]].copy()
        first["profile_id"] = str(drug)
        first["n_profiles"] = int(profile_rows.iloc[idx_arr]["n_profiles"].sum())
        score_parts.append(mean_score)
        y_parts.append(y_rows[0])
        row_parts.append(first)
    return np.vstack(score_parts), np.vstack(y_parts), pd.DataFrame(row_parts).reset_index(drop=True)


def new_query_adapter(r247_module, model, cfg: dict[str, Any], device: torch.device):
    adapter = r247_module.QueryGatedSupportAdapter(
        model.target_anchor.detach(),
        hidden_dim=int(cfg["hidden_dim"]),
        dropout=float(cfg["dropout"]),
        max_delta_norm=float(cfg["max_delta_norm"]),
    ).to(device)
    return adapter


def score_noadapt(r250, model, item: dict, device: torch.device) -> np.ndarray:
    return r250.predict_functional(model, item["x"], device, batch_size=1024)


def score_epiqgate(r252, r247, adapter, model, support_item: dict, query_item: dict, device: torch.device, support_size: int, selector: str, shift_scale: float) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    all_scores = []
    all_rows = []
    all_y = []
    audit_rows = []
    adapter.eval()
    model.eval()
    with torch.no_grad():
        for cell, qidx_raw in query_item["by_cell"].items():
            qidx = np.asarray(qidx_raw, dtype=np.int64)
            support_all = np.asarray(support_item["by_cell"].get(cell, np.asarray([], dtype=np.int64)), dtype=np.int64)
            if len(qidx) == 0 or len(support_all) == 0:
                continue
            support_rows = support_item["rows"].iloc[support_all].reset_index(drop=True)
            support_y_np = support_item["y"][support_all]
            local = r252.select_support_indices(r247, support_rows, support_y_np, support_size, selector, r252.stable_seed(r252.SEED, "r309", cell, selector, support_size))
            support_idx = support_all[local]
            qrows = query_item["rows"].iloc[qidx].reset_index(drop=True)
            overlap = r252.drug_overlap_count(r247, support_rows, qrows, local)
            if overlap:
                raise RuntimeError(f"Support/query drug overlap for {cell}: {overlap}")
            support_q = r247.tensor_slice(support_item, "q", support_idx, device)
            support_y = r247.tensor_slice(support_item, "y", support_idx, device)
            support_w = r247.tensor_slice(support_item, "weights", support_idx, device)
            query_q = r247.tensor_slice(query_item, "q", qidx, device)
            delta, gate = adapter(query_q, support_q, support_y, support_w)
            scores = r247.score_with_query_delta(model, query_q, delta, gate, float(shift_scale)).detach().cpu().numpy()
            all_scores.append(scores)
            all_rows.append(qrows)
            all_y.append(query_item["y"][qidx])
            audit = r247.support_audit(support_rows, support_y_np, local, selector, support_size)
            audit.update({"cell_line": cell, "support_query_drug_overlap": overlap, "used_support_rows": int(len(support_idx))})
            audit_rows.append(audit)
    return np.vstack(all_scores), pd.concat(all_rows, ignore_index=True), np.vstack(all_y), pd.DataFrame(audit_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-profile-score-matrix", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    r252 = import_from_path("round309_round252", ROUND252_SCRIPT)
    r247 = import_from_path("round309_round247", ROUND247_SCRIPT)
    r250 = import_from_path("round309_round250", ROUND250_SCRIPT)

    # Redirect imported modules to their original roots so local relative imports resolve.
    r252.ROUND_ROOT = ROUND252_ROOT
    r252.SOURCE = ROUND252_ROOT / "source_data"
    r252.QA = ROUND252_ROOT / "qa"
    r252.CHECKPOINTS = ROUND252_ROOT / "checkpoints"
    r252.LOGS = ROUND252_ROOT / "logs"
    r250.ROUND_ROOT = ROUND250_ROOT
    r250.SOURCE = ROUND250_ROOT / "source_data"
    r250.QA = ROUND250_ROOT / "qa"
    r250.CHECKPOINTS = ROUND250_ROOT / "checkpoints"
    r250.LOGS = ROUND250_ROOT / "logs"
    r247.ROUND_ROOT = ROUND247_ROOT
    r247.SOURCE = ROUND247_ROOT / "source_data"
    r247.QA = ROUND247_ROOT / "qa"
    r247.CHECKPOINTS = ROUND247_ROOT / "checkpoints"
    r247.LOGS = ROUND247_ROOT / "logs"

    context_args = argparse.Namespace(
        kg_checkpoint=None,
        pseudo_unseen_cells=",".join(r250.DEFAULT_R250_PSEUDO_UNSEEN_VAL_CELLS),
        official_unseen_cells="round215_heldout_panel",
        seen_diagnostic_cells=",".join(r250.FORCED_TRAIN_CELLS),
        support_baseline_variant=r250.BASE_REFERENCE["round246_current_joint"],
        support_checkpoint=None,
    )
    context = r252.load_context(r250, context_args, device)
    r243 = context["r243"]
    model = context["support_model"]
    for table in context["tables"].values():
        r243.precompute_q(model, table, device)

    adapter_ckpt = resolve_existing(None, ROUND252_ROOT, f"checkpoints/{ROUND252_CORE_VARIANT}/adapter_checkpoint.pth")
    payload = torch.load(adapter_ckpt, map_location="cpu", weights_only=False)
    selected = {"cfg": payload["cfg"], "checkpoint": str(adapter_ckpt)}
    adapter = new_query_adapter(r247, model, selected["cfg"], device)
    adapter.load_state_dict(payload["adapter_state_dict"], strict=True)
    adapter.eval()

    query_item = context["tables"]["official_test_profile"]
    support_item = context["tables"]["official_support"]

    no_profile_scores = score_noadapt(r250, model, query_item, device)
    no_drug_scores, no_drug_y, no_drug_rows = collapse_scores_to_drugcell(no_profile_scores, query_item["rows"], query_item["y"])
    no_per = compute_unit_metrics(no_drug_scores, no_drug_y, no_drug_rows, "Round246 no-adapt", ROUND246_NOADAPT_VARIANT, "score-mean drug-cell")
    no_cells = per_cell_summary(no_per, "Round246 no-adapt", ROUND246_NOADAPT_VARIANT, "score-mean drug-cell")

    epi_profile_scores, epi_profile_rows, epi_profile_y, support_audit = score_epiqgate(
        r252,
        r247,
        adapter,
        model,
        support_item,
        query_item,
        device,
        support_size=64,
        selector="diverse_label",
        shift_scale=float(selected["cfg"]["shift_scale"]),
    )
    epi_drug_scores, epi_drug_y, epi_drug_rows = collapse_scores_to_drugcell(epi_profile_scores, epi_profile_rows, epi_profile_y)
    epi_per = compute_unit_metrics(epi_drug_scores, epi_drug_y, epi_drug_rows, "Round252 EpiQGate diverse_label s64", ROUND252_EPIQGATE_VARIANT, "score-mean drug-cell")
    epi_cells = per_cell_summary(epi_per, "Round252 EpiQGate diverse_label s64", ROUND252_EPIQGATE_VARIANT, "score-mean drug-cell")

    # Historical metric-mean comparator from existing Round252 source data.
    round252_metric_path = ROUND252_ROOT / "source_data" / "metric_summary.tsv"
    round252_metric = pd.read_csv(round252_metric_path, sep="\t")
    metric_mask = (
        (round252_metric["source"].astype(str) == "Round252_EpiQGate")
        & (round252_metric["variant"].astype(str) == ROUND252_EPIQGATE_VARIANT)
        & (round252_metric["split"].astype(str) == "official_test")
        & (round252_metric["evaluation_unit"].astype(str) == "drug-mean")
    )
    metric_mean_row = round252_metric.loc[metric_mask].copy()
    if metric_mean_row.empty:
        raise RuntimeError(f"Cannot find Round252 official_test drug-mean metric-mean row in {round252_metric_path}")
    metric_mean_row = metric_mean_row.iloc[[0]].copy()
    metric_mean_row.insert(0, "comparison_mode", "metric-mean")
    metric_mean_row.to_csv(SOURCE / "Round252_metar_metricmean_summary.tsv", sep="\t", index=False)

    delta_metrics = [metric for metric in METRICS + ["best_true_target_percentile"] if metric in metric_mean_row.columns and metric in epi_per.columns]
    metricmean_vs_scoremean = pd.DataFrame(
        [
            {
                "method": "Round252 EpiQGate diverse_label s64",
                "comparison_mode": "metric-mean_minus_score-mean",
                "evaluation_unit": "drug-mean",
                **{metric: float(metric_mean_row.iloc[0][metric]) - float(epi_per[metric].mean()) for metric in delta_metrics},
            }
        ]
    )
    metricmean_vs_scoremean.to_csv(SOURCE / "Round252_metar_metricmean_vs_scoremean_delta.tsv", sep="\t", index=False)

    per = pd.concat([no_per, epi_per], ignore_index=True, sort=False)
    cells = pd.concat([no_cells, epi_cells], ignore_index=True, sort=False)
    summary = pd.DataFrame(
        [
            summarize_from_per_cell(no_cells, "Round246 no-adapt", ROUND246_NOADAPT_VARIANT, "official_test", "score-mean drug-cell", "cell_unweighted_mean"),
            summarize_from_per_cell(epi_cells, "Round252 EpiQGate diverse_label s64", ROUND252_EPIQGATE_VARIANT, "official_test", "score-mean drug-cell", "cell_unweighted_mean"),
        ]
    )
    summary.to_csv(SOURCE / "Round252_metar_scoremean_summary.tsv", sep="\t", index=False)

    per.to_csv(SOURCE / "Round252_metar_scoremean_drugcell_metrics.tsv", sep="\t", index=False)
    cells.to_csv(SOURCE / "Round252_metar_scoremean_per_cell_metrics.tsv", sep="\t", index=False)
    support_audit.to_csv(QA / "round309_scoremean_support_selection_audit.tsv", sep="\t", index=False)

    if not args.skip_profile_score_matrix:
        np.savez_compressed(
            SOURCE / "Round252_metar_official_profile_scores_scoremean_input.npz",
            noadapt_scores=no_profile_scores.astype(np.float32),
            epiqgate_scores=epi_profile_scores.astype(np.float32),
            y=query_item["y"].astype(np.int8),
        )
        query_item["rows"].to_csv(SOURCE / "Round252_metar_official_profile_score_rows.tsv", sep="\t", index=False)

    method_lock = {
        "round": "Round309",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task": "MeTar score-mean drug-cell evaluation for Fig.4 Round307 comparator update",
        "candidate_space": 668,
        "official_unseen_cells": sorted(query_item["rows"]["cell_line"].astype(str).unique().tolist()),
        "n_profile_rows": int(len(query_item["rows"])),
        "n_drugcell_units": int(len(epi_per)),
        "device": str(device),
        "round246_checkpoint": str(context["support_checkpoint"]),
        "round246_checkpoint_sha256": r243.sha256_file(context["support_checkpoint"]),
        "round252_adapter_checkpoint": str(adapter_ckpt),
        "round252_adapter_checkpoint_sha256": sha256_file(adapter_ckpt),
        "selected_round252": selected,
        "outputs": {
            "per_unit": str(SOURCE / "Round252_metar_scoremean_drugcell_metrics.tsv"),
            "per_cell": str(SOURCE / "Round252_metar_scoremean_per_cell_metrics.tsv"),
            "summary": str(SOURCE / "Round252_metar_scoremean_summary.tsv"),
        },
    }
    (ROUND_ROOT / "method_lock_scoremean_eval.json").write_text(json.dumps(json_ready(method_lock), indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "summary": summary.to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    main()
