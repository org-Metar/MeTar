#!/usr/bin/env python
"""Export full-space single-cell target score vectors for Fig. 7 consensus.

This script runs on the server inside the Round259/260 single-cell overlay.
It loads the selected primary checkpoint for A549, SW480 and NCI-H460,
recomputes the fixed 300-cell holdout evaluations, and writes the complete
16,583-label score vectors needed for drug-level three-cell consensus.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


CELL_DIRS = {
    "A549": "train_A549",
    "SW480": "train_SW480",
    "NCI-H460": "train_NCI_H460",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Round259/260 overlay root on the server")
    parser.add_argument(
        "--out-dir",
        default="outputs/fig7_round483_singlecell_scores",
        help="Output directory, relative to --root unless absolute",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device; falls back to CPU if CUDA is unavailable")
    return parser.parse_args()


def load_base_module(root: Path):
    script = root / "scripts" / "run_sc_label_bag_calibration_multicell_base.py"
    if not script.exists():
        raise FileNotFoundError(script)
    spec = importlib.util.spec_from_file_location("round260_sc_base", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rank_diagnostics(scores: np.ndarray, pos_idx: list[int]) -> dict[str, object]:
    order = np.argsort(-scores, kind="mergesort")
    ranks = [int(np.where(order == p)[0][0]) + 1 for p in pos_idx]
    n = int(len(scores))
    best_rank = min(ranks) if ranks else None
    best_percentile = (best_rank / float(n)) if best_rank is not None else np.nan
    return {
        "best_rank": best_rank,
        "best_percentile": best_percentile,
        "all_positive_ranks": ";".join(map(str, sorted(ranks))),
        "hit_top5pct": int(best_percentile <= 0.05) if best_rank is not None else 0,
        "hit_top10pct": int(best_percentile <= 0.10) if best_rank is not None else 0,
    }


def export_cell(
    base,
    root: Path,
    out_dir: Path,
    cell_line: str,
    train_dir_name: str,
    device: torch.device,
) -> tuple[pd.DataFrame, list[np.ndarray], list[str]]:
    train_dir = root / train_dir_name
    selected_path = train_dir / "selected_primary_result.json"
    if not selected_path.exists():
        selected_path = train_dir / "source_data" / "selected_primary_result.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))

    base.TARGET_EMBEDDING_PATH = Path(selected["target_embedding_path"])
    adata_path = Path(selected["adata_path"])
    ctx = base.SweepContext(device, train_dir, adata_path, cell_line)

    loss_mode = str(selected["loss_mode"])
    seed = int(selected["seed"])
    run_dir = train_dir / f"{loss_mode}__seed{seed}"
    checkpoint = run_dir / "checkpoint.pth"
    holdout_path = run_dir / "holdout_per_drug.tsv"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if not holdout_path.exists():
        raise FileNotFoundError(holdout_path)

    model = base.build_model(loss_mode, in_dim=ctx.adata.n_vars, out_dim=ctx.target_df.shape[1]).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    expected = pd.read_csv(holdout_path, sep="\t")
    expected = expected.sort_values(["drug_category", "drug"]).reset_index(drop=True)
    rows: list[dict[str, object]] = []
    score_blocks: list[np.ndarray] = []

    with torch.no_grad():
        for _, exp in expected.iterrows():
            drug = str(exp["drug"])
            pos_idx = list(ctx.holdout_idx_by_drug.get(drug, []))
            if not pos_idx or drug not in ctx.eval_cells_by_drug:
                raise RuntimeError(f"{cell_line}: cannot evaluate exported holdout drug {drug}")

            x = base.fetch_x_batch(ctx.adata, ctx.eval_cells_by_drug[drug], device).view(1, -1, ctx.adata.n_vars)
            scores = model.forward_logits_batch(x, ctx.w_all).detach().cpu().numpy()[0].astype(np.float32)
            diag = rank_diagnostics(scores, pos_idx)
            positive_targets = [ctx.all_genes[i] for i in pos_idx]

            rows.append(
                {
                    "cell_line": cell_line,
                    "drug": drug,
                    "drug_category": str(exp["drug_category"]),
                    "loss_mode": loss_mode,
                    "seed": seed,
                    "best_epoch": int(selected.get("best_epoch", -1)),
                    "n_eval_cells": int(len(ctx.eval_cells_by_drug[drug])),
                    "n_holdout_positive": int(len(pos_idx)),
                    "positive_targets": "|".join(sorted(positive_targets)),
                    "gene_universe_size": int(ctx.target_df.shape[0]),
                    **diag,
                    "saved_best_rank": int(exp["best_rank"]),
                    "saved_best_percentile": float(exp["best_percentile"]),
                    "saved_hit_top5pct": int(exp["hit_top5pct"]),
                    "saved_hit_top10pct": int(exp["hit_top10pct"]),
                }
            )
            score_blocks.append(scores)

    metrics = pd.DataFrame(rows)
    mismatch = metrics[
        (metrics["best_rank"].astype(int) != metrics["saved_best_rank"].astype(int))
        | (np.abs(metrics["best_percentile"].astype(float) - metrics["saved_best_percentile"].astype(float)) > 1e-10)
        | (metrics["hit_top5pct"].astype(int) != metrics["saved_hit_top5pct"].astype(int))
        | (metrics["hit_top10pct"].astype(int) != metrics["saved_hit_top10pct"].astype(int))
    ]
    if len(mismatch):
        mismatch.to_csv(out_dir / f"{cell_line.replace('-', '_')}_recompute_mismatch.tsv", sep="\t", index=False)
        raise RuntimeError(f"{cell_line}: recomputed scores do not match saved holdout metrics")

    target_labels = list(ctx.all_genes)
    del ctx, model
    return metrics, score_blocks, target_labels


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    base = load_base_module(root)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    all_metrics: list[pd.DataFrame] = []
    all_scores: list[np.ndarray] = []
    audit: dict[str, object] = {
        "root": str(root),
        "device": str(device),
        "cell_lines": list(CELL_DIRS.keys()),
        "purpose": "Fig.7 Round483 single-cell three-cell consensus score export",
    }

    target_labels: list[str] | None = None
    for cell_line, train_dir in CELL_DIRS.items():
        metrics, scores, labels = export_cell(base, root, out_dir, cell_line, train_dir, device)
        if target_labels is None:
            target_labels = labels
        elif labels != target_labels:
            raise RuntimeError(f"{cell_line}: target label order differs from previous cell lines")
        all_metrics.append(metrics)
        all_scores.extend(scores)
        del metrics, scores, labels
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metric_df = pd.concat(all_metrics, ignore_index=True)
    if target_labels is None:
        raise RuntimeError("No target labels exported")
    score_df = pd.DataFrame(np.vstack(all_scores), columns=target_labels)
    score_df = pd.concat([metric_df[["cell_line", "drug"]].reset_index(drop=True), score_df], axis=1)

    metric_df.to_csv(out_dir / "round260_singlecell_score_metrics.tsv", sep="\t", index=False)
    score_df.to_csv(out_dir / "round260_singlecell_fullspace_scores.tsv.gz", sep="\t", index=False, compression="gzip")
    pd.DataFrame({"target_label": target_labels, "target_index": np.arange(len(target_labels), dtype=int)}).to_csv(
        out_dir / "round260_singlecell_target_labels.tsv", sep="\t", index=False
    )

    audit.update(
        {
            "n_rows": int(len(metric_df)),
            "n_cell_lines": int(metric_df["cell_line"].nunique()),
            "n_shared_drugs": int(metric_df.groupby("drug")["cell_line"].nunique().eq(len(CELL_DIRS)).sum()),
            "target_universe_size": int(len(target_labels)),
            "top5_units": int(metric_df["hit_top5pct"].sum()),
            "top10_units": int(metric_df["hit_top10pct"].sum()),
        }
    )
    (out_dir / "round260_singlecell_export_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
