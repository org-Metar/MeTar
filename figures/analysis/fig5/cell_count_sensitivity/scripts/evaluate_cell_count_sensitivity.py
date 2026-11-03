#!/usr/bin/env python3
"""Round393 prediction-time single-cell number sensitivity extra points.

This script is evaluation-only. It reloads the selected Round260 single-cell
adapters, keeps target embeddings and holdout definitions fixed, and varies
only the number of single-cell profiles supplied at inference time.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


TARGET_EMBEDDING_PATH = Path(
    "/NFS2_home/NFS2_home_1/lwn/claude/manuscript/write/"
    "claude_round259_crosscell_fullspace_round252query_then_sc_proteomics_20260522/"
    "source_data/round259_target_embedding.csv"
)
PROTOCOL_JSON = Path(
    "/home/lwn/claude/manuscript/write/sc_agg_upgrade_20260321/"
    "derived_tables/sc_protocol_lock_v3_20260326.json"
)
DTI_PATH = Path("/home/lwn/claude/sc/data/tahoe_dti.csv")

ROUND259_SC_ROOT = Path(
    "/home/lwn/claude/manuscript/write/"
    "claude_round259_crosscell_fullspace_round252query_then_sc_proteomics_20260522"
)

CELL_CONFIGS: dict[str, dict[str, Any]] = {
    "A549": {
        "adata_path": Path("/home/lwn/claude/sc/data/tahoe_a549.h5ad"),
        "train_dir": ROUND259_SC_ROOT / "train_A549",
        "loss_mode": "drug_normalized_bce",
        "seed": 2043,
        "expected_top5": 21,
        "expected_top10": 23,
    },
    "SW480": {
        "adata_path": Path("/NFS2_home/NFS2_home_1/lwn/claude/sc/data/tahoe_sw480.h5ad"),
        "train_dir": ROUND259_SC_ROOT / "train_SW480",
        "loss_mode": "label_bag_retrieval",
        "seed": 2041,
        "expected_top5": 19,
        "expected_top10": 21,
    },
    "NCI-H460": {
        "adata_path": Path("/NFS2_home/NFS2_home_1/lwn/claude/sc/data/tahoe_nci_h460.h5ad"),
        "train_dir": ROUND259_SC_ROOT / "train_NCI_H460",
        "loss_mode": "drug_normalized_bce",
        "seed": 2042,
        "expected_top5": 18,
        "expected_top10": 21,
    },
}

LOSS_MODES = {
    "current_bce_baseline",
    "drug_normalized_bce",
    "label_bag_retrieval",
    "cap_positive_5_bce",
    "outlier_filter_gt10_bce",
}
HOLDOUT_PERCENTILES = [0.01, 0.05, 0.10, 0.20]
CELL_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 300]
REPEATS = 30
EVAL_CELLS = 300
SEED_EVAL_CELLS = 20260327
SEED_SUBSAMPLE = 20260611
KS = [1, 5, 10, 20, 30, 50]

HIDDEN_DIM = 512
LATENT_DIM = 256
GATE_HIDDEN = 64
DROPOUT = 0.25


def norm_drug(x: object) -> str:
    return str(x).strip().lower()


def stable_seed(name: str, base_seed: int) -> int:
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    return (base_seed + int(digest[:8], 16)) % (2**32 - 1)


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def parse_targets_cell(x: object) -> list[str]:
    s = str(x).strip()
    if s in ("", "nan", "none", "null"):
        return []
    try:
        value = ast.literal_eval(s)
    except Exception:
        value = s
    if isinstance(value, (list, tuple, set)):
        return [str(t).strip() for t in value if str(t).strip()]
    return [str(value).strip()]


def fetch_x_batch(adata, idxs: np.ndarray, device: torch.device) -> torch.Tensor:
    x_raw = adata.X[idxs]
    if hasattr(x_raw, "toarray"):
        x_raw = x_raw.toarray()
    x = np.asarray(x_raw, dtype=np.float32)
    return torch.tensor(x, device=device, dtype=torch.float32)


class BasicCellEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MeanResidualGateModel(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.cell_encoder = BasicCellEncoder(in_dim, out_dim)
        self.gate = nn.Sequential(
            nn.Linear(in_dim, GATE_HIDDEN),
            nn.GELU(),
            nn.Linear(GATE_HIDDEN, 1),
        )
        self.residual = nn.Linear(in_dim, out_dim)
        self.beta_logit = nn.Parameter(torch.tensor(0.0))
        self.logit_scale_raw = nn.Parameter(
            torch.tensor(np.log(np.exp(10.0) - 1.0), dtype=torch.float32)
        )

    def forward_query(self, x: torch.Tensor) -> torch.Tensor:
        h = self.cell_encoder(x)
        w = torch.softmax(self.gate(x).squeeze(-1), dim=1).unsqueeze(-1)
        pooled = (h * w).sum(dim=1)
        residual = self.residual(x.mean(dim=1))
        beta = torch.sigmoid(self.beta_logit)
        q = (1.0 - beta) * pooled + beta * residual
        return F.normalize(q, dim=-1)

    def forward_logits_batch(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        q = self.forward_query(x)
        scale = F.softplus(self.logit_scale_raw) + 1e-6
        return scale * q @ target.T


def build_model(loss_mode: str, in_dim: int, out_dim: int) -> nn.Module:
    if loss_mode not in LOSS_MODES:
        raise ValueError(f"Unknown loss_mode: {loss_mode}")
    return MeanResidualGateModel(in_dim, out_dim)


def build_cell_pool_from_adata(adata) -> dict[str, np.ndarray]:
    if "drug" not in adata.obs.columns:
        raise RuntimeError("Selected h5ad must include obs['drug']")
    drugs = adata.obs["drug"].astype(str).map(norm_drug).to_numpy()
    pools: dict[str, list[int]] = defaultdict(list)
    for idx, drug in enumerate(drugs):
        if drug and drug != "nan":
            pools[drug].append(idx)
    return {drug: np.asarray(indices, dtype=np.int64) for drug, indices in pools.items()}


def build_eval_cells(cell_pool_by_drug: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    eval_cells = {}
    for drug, pool in cell_pool_by_drug.items():
        m = min(len(pool), EVAL_CELLS)
        if m == len(pool):
            eval_cells[drug] = np.asarray(pool, dtype=np.int64)
        else:
            rng = np.random.RandomState(stable_seed(drug, SEED_EVAL_CELLS))
            eval_cells[drug] = np.sort(rng.choice(pool, size=m, replace=False))
    return eval_cells


def choose_subsample(base_eval_cells: np.ndarray, cell_line: str, drug: str, n_cells: int, repeat: int) -> np.ndarray:
    if len(base_eval_cells) < n_cells:
        raise ValueError(f"{cell_line} {drug} has only {len(base_eval_cells)} eval cells, requested {n_cells}")
    if len(base_eval_cells) == n_cells:
        return np.asarray(base_eval_cells, dtype=np.int64)
    token = f"{cell_line}::{drug}::{n_cells}::{repeat}"
    rng = np.random.RandomState(stable_seed(token, SEED_SUBSAMPLE))
    return np.sort(rng.choice(base_eval_cells, size=n_cells, replace=False))


def rank_metrics(scores: np.ndarray, pos_indices: list[int], candidate_space: int) -> dict[str, Any]:
    order = np.argsort(-scores)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(1, len(order) + 1)
    ranks = [int(inverse[p]) for p in pos_indices]
    best_rank = min(ranks)
    best_percentile = best_rank / float(candidate_space)
    out: dict[str, Any] = {
        "best_rank": best_rank,
        "best_percentile": best_percentile,
        "all_positive_ranks": ";".join(map(str, sorted(ranks))),
        "n_holdout_positive": int(len(pos_indices)),
    }
    for pct in HOLDOUT_PERCENTILES:
        cutoff = int(np.floor(candidate_space * pct))
        recovered = int(sum(rank <= cutoff for rank in ranks))
        label = f"top{int(pct * 100)}pct"
        out[f"hit_{label}"] = int(recovered > 0)
        out[f"recovered_positive_{label}"] = recovered
        out[f"recall_{label}"] = recovered / float(len(pos_indices))
        out[f"{label}_cutoff_rank"] = cutoff
    return out


class EvalContext:
    def __init__(self, cell_line: str, config: dict[str, Any], device: torch.device):
        self.cell_line = cell_line
        self.config = config
        self.device = device

        self.target_df = pd.read_csv(TARGET_EMBEDDING_PATH, index_col=0)
        self.target_df = self.target_df[~self.target_df.index.duplicated(keep="first")]
        self.target_df.index = self.target_df.index.astype(str)
        if tuple(self.target_df.shape) != (16583, 256):
            raise RuntimeError(f"Expected target table (16583, 256), got {self.target_df.shape}")
        self.all_genes = list(self.target_df.index)
        self.gene_to_global = {gene: idx for idx, gene in enumerate(self.all_genes)}
        self.w_all = F.normalize(torch.tensor(self.target_df.values.astype(np.float32), device=device), dim=1)

        with PROTOCOL_JSON.open("r", encoding="utf-8") as handle:
            self.protocol = json.load(handle)
        if self.protocol.get("protocol_tag") != "v3_corrected_gene_targets_fixed_cells":
            raise RuntimeError(f"Unexpected protocol tag: {self.protocol.get('protocol_tag')}")
        if self.protocol.get("target_column") != "id":
            raise RuntimeError(f"Expected target_column='id', got {self.protocol.get('target_column')}")

        self.adata = ad.read_h5ad(config["adata_path"])
        if self.adata.n_vars != 128:
            raise RuntimeError(f"{cell_line}: expected 128 Tahoe features, got {self.adata.n_vars}")
        self.adata.obs["drug"] = self.adata.obs["drug"].astype(str).map(norm_drug)
        self.cell_pool_by_drug = build_cell_pool_from_adata(self.adata)
        self.eval_cells_by_drug = build_eval_cells(self.cell_pool_by_drug)

        self.exclusive_drugs = np.asarray([norm_drug(d) for d in self.protocol["exclusive_drugs"]])
        self.mixed_drugs = np.asarray([norm_drug(d) for d in self.protocol["mixed_drugs"]])
        self.holdout_targets_set = set(str(g) for g in self.protocol["holdout_targets"])
        if set(self.protocol["seen_targets"]) & set(self.protocol["holdout_targets"]):
            raise RuntimeError("Seen and holdout targets overlap in protocol")

        self.scdrug = pd.read_csv(DTI_PATH, index_col=0)
        self.scdrug.columns = [c.lower() for c in self.scdrug.columns]
        if "drug" not in self.scdrug.columns or "id" not in self.scdrug.columns:
            raise RuntimeError("tahoe_dti.csv must include drug and id columns")

        drug2targets: dict[str, list[str]] = defaultdict(list)
        for _, row in self.scdrug.iterrows():
            drug = norm_drug(row["drug"])
            for target in parse_targets_cell(row["id"]):
                if target in self.gene_to_global and target not in drug2targets[drug]:
                    drug2targets[drug].append(target)
        self.drug2targets = {drug: sorted(targets) for drug, targets in drug2targets.items() if targets}
        self.holdout_idx_by_drug = {
            drug: [self.gene_to_global[t] for t in targets if t in self.holdout_targets_set]
            for drug, targets in self.drug2targets.items()
        }

        self.units: list[dict[str, Any]] = []
        for drug, category in [(d, "exclusive") for d in self.exclusive_drugs] + [(d, "mixed") for d in self.mixed_drugs]:
            holdout_idx = self.holdout_idx_by_drug.get(drug, [])
            if not holdout_idx or drug not in self.eval_cells_by_drug:
                continue
            self.units.append(
                {
                    "cell_line": cell_line,
                    "drug": drug,
                    "drug_category": category,
                    "holdout_idx": holdout_idx,
                    "n_available_cells": int(len(self.cell_pool_by_drug[drug])),
                    "n_base_eval_cells": int(len(self.eval_cells_by_drug[drug])),
                }
            )
        if len(self.units) != 30:
            raise RuntimeError(f"{cell_line}: expected 30 holdout units, got {len(self.units)}")

        run_dir = config["train_dir"] / f"{config['loss_mode']}__seed{config['seed']}"
        checkpoint_path = run_dir / "checkpoint.pth"
        result_path = run_dir / "result.json"
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        if not result_path.exists():
            raise FileNotFoundError(result_path)
        with result_path.open("r", encoding="utf-8") as handle:
            self.selected_result = json.load(handle)
        self.model = build_model(config["loss_mode"], in_dim=self.adata.n_vars, out_dim=self.target_df.shape[1]).to(device)
        state = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.checkpoint_path = checkpoint_path
        self.result_path = result_path


def eval_context(ctx: EvalContext, cell_counts: list[int], repeats: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unit_rows: list[dict[str, Any]] = []
    denominator_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for n_cells in cell_counts:
            for repeat in range(repeats):
                for unit in ctx.units:
                    drug = unit["drug"]
                    base_cells = ctx.eval_cells_by_drug[drug]
                    if len(base_cells) < n_cells:
                        denominator_rows.append(
                            {
                                "cell_line": ctx.cell_line,
                                "drug": drug,
                                "cell_count": n_cells,
                                "repeat": repeat,
                                "status": "insufficient_cells",
                                "n_base_eval_cells": int(len(base_cells)),
                                "n_available_cells": int(unit["n_available_cells"]),
                            }
                        )
                        continue
                    idxs = choose_subsample(base_cells, ctx.cell_line, drug, n_cells, repeat)
                    x = fetch_x_batch(ctx.adata, idxs, ctx.device).view(1, -1, ctx.adata.n_vars)
                    scores = ctx.model.forward_logits_batch(x, ctx.w_all).detach().cpu().numpy()[0]
                    metrics = rank_metrics(scores, unit["holdout_idx"], len(ctx.all_genes))
                    unit_rows.append(
                        {
                            "cell_line": ctx.cell_line,
                            "drug": drug,
                            "drug_category": unit["drug_category"],
                            "cell_count": n_cells,
                            "repeat": repeat,
                            "n_input_cells": int(len(idxs)),
                            "n_available_cells": int(unit["n_available_cells"]),
                            "n_base_eval_cells": int(len(base_cells)),
                            **metrics,
                        }
                    )
    return unit_rows, denominator_rows


def summarize_repeat_level(unit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(unit_rows)
    rows: list[dict[str, Any]] = []
    group_cols = ["cell_count", "repeat"]
    for keys, sub in df.groupby(group_cols, sort=True):
        cell_count, repeat = keys
        rows.extend(summarize_one_group(sub, int(cell_count), int(repeat), "pooled"))
    for keys, sub in df.groupby(["cell_line", "cell_count", "repeat"], sort=True):
        cell_line, cell_count, repeat = keys
        rows.extend(summarize_one_group(sub, int(cell_count), int(repeat), str(cell_line)))
    return rows


def summarize_one_group(sub: pd.DataFrame, cell_count: int, repeat: int, cell_line: str) -> list[dict[str, Any]]:
    rows = []
    base = {
        "cell_line": cell_line,
        "cell_count": cell_count,
        "repeat": repeat,
        "n_units": int(len(sub)),
        "n_holdout_positive_total": int(sub["n_holdout_positive"].sum()),
    }
    for pct in HOLDOUT_PERCENTILES:
        label = f"top{int(pct * 100)}pct"
        n_success = int(sub[f"hit_{label}"].sum())
        recovered_positive = int(sub[f"recovered_positive_{label}"].sum())
        total_positive = int(sub["n_holdout_positive"].sum())
        rows.append(
            {
                **base,
                "metric": f"anyhit_{label}",
                "threshold_percent": pct,
                "value": n_success / float(len(sub)) if len(sub) else np.nan,
                "n_success": n_success,
                "n_recovered_positive": recovered_positive,
                "recall_micro": recovered_positive / float(total_positive) if total_positive else np.nan,
                "recall_unit_mean": float(sub[f"recall_{label}"].mean()),
            }
        )
        rows.append(
            {
                **base,
                "metric": f"recall_{label}",
                "threshold_percent": pct,
                "value": recovered_positive / float(total_positive) if total_positive else np.nan,
                "n_success": n_success,
                "n_recovered_positive": recovered_positive,
                "recall_micro": recovered_positive / float(total_positive) if total_positive else np.nan,
                "recall_unit_mean": float(sub[f"recall_{label}"].mean()),
            }
        )
    rows.append(
        {
            **base,
            "metric": "median_best_percentile",
            "threshold_percent": "",
            "value": float(sub["best_percentile"].median()),
            "n_success": "",
            "n_recovered_positive": "",
            "recall_micro": "",
            "recall_unit_mean": "",
        }
    )
    return rows


def summarize_curve(repeat_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(repeat_rows)
    out_rows: list[dict[str, Any]] = []
    for keys, sub in df.groupby(["cell_line", "cell_count", "metric"], sort=True):
        cell_line, cell_count, metric = keys
        values = pd.to_numeric(sub["value"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        out_rows.append(
            {
                "cell_line": cell_line,
                "cell_count": int(cell_count),
                "metric": metric,
                "n_repeats": int(len(values)),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "sem": float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0,
                "q025": float(np.quantile(values, 0.025)),
                "q975": float(np.quantile(values, 0.975)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
    return out_rows


def write_tsv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("/home/lwn/claude/manuscript/write/claude_round393_fig5ef_refine_log2_20260611"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--cell-counts", default=",".join(map(str, CELL_COUNTS)))
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    random.seed(SEED_SUBSAMPLE)
    np.random.seed(SEED_SUBSAMPLE)
    torch.manual_seed(SEED_SUBSAMPLE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED_SUBSAMPLE)

    out_dir = args.out_dir
    source_dir = out_dir / "source_data"
    qa_dir = out_dir / "qa"
    source_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    cell_counts = [int(x) for x in args.cell_counts.split(",") if x.strip()]
    all_unit_rows: list[dict[str, Any]] = []
    denominator_rows: list[dict[str, Any]] = []
    context_locks = []
    t0 = time.time()
    for cell_line, config in CELL_CONFIGS.items():
        print(json.dumps({"event": "load_context", "cell_line": cell_line}, sort_keys=True), flush=True)
        ctx = EvalContext(cell_line, config, device)
        unit_rows, denom_rows = eval_context(ctx, cell_counts, args.repeats)
        all_unit_rows.extend(unit_rows)
        denominator_rows.extend(denom_rows)
        context_locks.append(
            {
                "cell_line": cell_line,
                "adata_path": str(config["adata_path"]),
                "adata_sha256": sha256_file(config["adata_path"]),
                "train_dir": str(config["train_dir"]),
                "loss_mode": config["loss_mode"],
                "seed": config["seed"],
                "checkpoint_path": str(ctx.checkpoint_path),
                "checkpoint_sha256": sha256_file(ctx.checkpoint_path),
                "result_path": str(ctx.result_path),
                "result_sha256": sha256_file(ctx.result_path),
                "n_units": len(ctx.units),
            }
        )
        print(json.dumps({"event": "done_cell", "cell_line": cell_line, "rows": len(unit_rows)}, sort_keys=True), flush=True)

    repeat_rows = summarize_repeat_level(all_unit_rows)
    curve_rows = summarize_curve(repeat_rows)

    unit_path = source_dir / "Figure5_R393_extra_cellnumber_unit_repeat_scores.tsv"
    repeat_path = source_dir / "Figure5_R393_extra_cellnumber_repeat_level_metrics.tsv"
    curve_path = source_dir / "Figure5_R393_extra_cellnumber_sensitivity_curve.tsv"
    denom_path = source_dir / "Figure5_R393_extra_cellnumber_denominator_audit.tsv"
    write_tsv(all_unit_rows, unit_path)
    write_tsv(repeat_rows, repeat_path)
    write_tsv(curve_rows, curve_path)
    write_tsv(denominator_rows, denom_path)

    repeat_df = pd.DataFrame(repeat_rows)
    has_300 = 300 in cell_counts
    if has_300:
        main = repeat_df[
            (repeat_df["cell_line"] == "pooled")
            & (repeat_df["cell_count"] == 300)
            & (repeat_df["repeat"] == 0)
            & (repeat_df["metric"].isin(["anyhit_top5pct", "anyhit_top10pct"]))
        ]
        top5 = float(main.loc[main["metric"] == "anyhit_top5pct", "value"].iloc[0])
        top10 = float(main.loc[main["metric"] == "anyhit_top10pct", "value"].iloc[0])
        if abs(top5 - 58 / 90) > 1e-9 or abs(top10 - 65 / 90) > 1e-9:
            raise RuntimeError(f"Reference mismatch at 300 cells: top5={top5}, top10={top10}")
    else:
        top5 = None
        top10 = None

    method_lock = {
        "round": "Round393",
        "purpose": "additional prediction-time input cell counts for Fig.5f log2-axis refinement",
        "evaluation_only": True,
        "training_performed": False,
        "target_embedding_path": str(TARGET_EMBEDDING_PATH),
        "target_embedding_sha256": sha256_file(TARGET_EMBEDDING_PATH),
        "target_shape": [16583, 256],
        "target_embeddings_frozen": True,
        "protocol_json": str(PROTOCOL_JSON),
        "protocol_sha256": sha256_file(PROTOCOL_JSON),
        "dti_path": str(DTI_PATH),
        "dti_sha256": sha256_file(DTI_PATH),
        "cell_counts": cell_counts,
        "repeats": args.repeats,
        "subsampling": "without replacement from the same deterministic 300-cell evaluation pool used by Round260",
        "seed_eval_cells": SEED_EVAL_CELLS,
        "seed_subsample": SEED_SUBSAMPLE,
        "device": str(device),
        "contexts": context_locks,
        "reference_check": {
            "performed": has_300,
            "pooled_300cell_top5": top5,
            "pooled_300cell_top10": top10,
            "expected_pooled_300cell_top5": 58 / 90,
            "expected_pooled_300cell_top10": 65 / 90,
        },
        "runtime_s": round(time.time() - t0, 2),
    }
    (out_dir / "method_lock.json").write_text(json.dumps(method_lock, indent=2), encoding="utf-8")

    print(json.dumps({"event": "complete", "out_dir": str(out_dir), "top5": top5, "top10": top10}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
