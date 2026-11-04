#!/usr/bin/env python3
"""Round386 X128 pseudo-bulk adapter comparison.

This diagnostic branch uses the existing Tahoe 128-dimensional single-cell
features as input. It first averages X128 features by `(cell_line, drug)`,
then trains a direct pseudo-bulk sample-side adapter against the fixed
16,583-gene target embedding exported by the full-space model. Inputs are
prepared X128 features, with outputs written to this comparison's directory.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


ROUND_ROOT = Path(__file__).resolve().parents[1]
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
CELL_LINE_TO_H5AD = {
    "A549": Path("/home/lwn/claude/sc/data/tahoe_a549.h5ad"),
    "SW480": Path("/NFS2_home/NFS2_home_1/lwn/claude/sc/data/tahoe_sw480.h5ad"),
    "NCI-H460": Path("/NFS2_home/NFS2_home_1/lwn/claude/sc/data/tahoe_nci_h460.h5ad"),
}
TARGET_CELLS = tuple(CELL_LINE_TO_H5AD)
PRIMARY_LOSS_MODES = [
    "current_bce_baseline",
    "drug_normalized_bce",
    "label_bag_retrieval",
]
SENSITIVITY_LOSS_MODES = [
    "cap_positive_5_bce",
    "outlier_filter_gt10_bce",
]
LOSS_MODES = PRIMARY_LOSS_MODES + SENSITIVITY_LOSS_MODES
SEEDS = [2041, 2042, 2043]
KS = [1, 5, 10, 20, 30, 50]
HOLDOUT_PERCENTILES = [0.01, 0.05, 0.10, 0.20]
BATCH_SIZE = 256
MAX_EPOCHS = 16
STEPS_PER_EPOCH = 200
EARLY_STOP_PATIENCE = 4
MIN_EPOCHS_BEFORE_STOP = 4
LR = 3e-4
WEIGHT_DECAY = 1e-5
HIDDEN_DIM = 512
DROPOUT = 0.25


def norm_drug(x: object) -> str:
    return str(x).strip().lower()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def decode_array(values) -> list[str]:
    out: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def parse_targets_cell(x: object) -> list[str]:
    s = str(x).strip()
    if s in ("", "nan", "none", "null"):
        return []
    try:
        v = ast.literal_eval(s)
    except Exception:
        v = s
    if isinstance(v, (list, tuple, set)):
        return [str(t).strip() for t in v if str(t).strip()]
    return [str(v).strip()]


def rank_diagnostics(scores: np.ndarray, pos_indices: list[int]) -> dict[str, Any]:
    order = np.argsort(-scores)
    ranks = [int(np.where(order == p)[0][0]) + 1 for p in pos_indices]
    n = len(scores)
    best_rank = min(ranks) if ranks else None
    best_percentile = best_rank / float(n) if best_rank is not None else None
    return {
        "best_rank": best_rank,
        "best_percentile": best_percentile,
        "all_positive_ranks": ";".join(map(str, sorted(ranks))),
    }


def write_tsv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def load_protocol() -> dict:
    with PROTOCOL_JSON.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    if protocol.get("protocol_tag") != "v3_corrected_gene_targets_fixed_cells":
        raise RuntimeError(f"Unexpected protocol tag: {protocol.get('protocol_tag')}")
    if protocol.get("target_column") != "id":
        raise RuntimeError(f"Unexpected target column: {protocol.get('target_column')}")
    return protocol


def protocol_drugs(protocol: dict) -> list[str]:
    keys = ["fit_drugs", "val_drugs", "test_drugs", "exclusive_drugs", "mixed_drugs"]
    return sorted({norm_drug(d) for key in keys for d in protocol[key]})


def split_label_by_drug(protocol: dict) -> dict[str, str]:
    rows: dict[str, list[str]] = defaultdict(list)
    for key in ["fit_drugs", "val_drugs", "test_drugs", "exclusive_drugs", "mixed_drugs"]:
        for drug in protocol[key]:
            rows[norm_drug(drug)].append(key.replace("_drugs", ""))
    return {drug: ";".join(sorted(set(labels))) for drug, labels in rows.items()}


def build_x128_pseudobulk(out_dir: Path, chunk_rows: int = 200000) -> None:
    protocol = load_protocol()
    drugs = protocol_drugs(protocol)
    split_label = split_label_by_drug(protocol)
    source_dir = out_dir / "source_data"
    logs_dir = out_dir / "logs"
    source_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    profile_x = []
    profile_cell = []
    profile_drug = []
    profile_count = []
    profile_id = []
    audit_rows = []

    for cell_line, h5ad_path in CELL_LINE_TO_H5AD.items():
        t0 = time.time()
        if not h5ad_path.exists():
            raise FileNotFoundError(h5ad_path)
        with h5py.File(h5ad_path, "r") as h5:
            x = h5["X"]
            if x.shape[1] != 128:
                raise RuntimeError(f"{h5ad_path} expected 128 features, got {x.shape[1]}")
            drug_group = h5["obs/drug"]
            categories = [norm_drug(v) for v in decode_array(drug_group["categories"][:])]
            codes_ds = drug_group["codes"]
            code_to_drug = {idx: drug for idx, drug in enumerate(categories) if drug in set(drugs)}
            drug_to_idx = {drug: i for i, drug in enumerate(drugs)}
            sums = np.zeros((len(drugs), x.shape[1]), dtype=np.float64)
            counts = np.zeros(len(drugs), dtype=np.int64)
            n_obs = int(x.shape[0])
            chunks_with_rows = 0
            for start in range(0, n_obs, chunk_rows):
                stop = min(start + chunk_rows, n_obs)
                codes = codes_ds[start:stop]
                mapped = np.full(stop - start, -1, dtype=np.int32)
                for code, drug in code_to_drug.items():
                    mask = codes == code
                    if np.any(mask):
                        mapped[mask] = drug_to_idx[drug]
                rows = np.flatnonzero(mapped >= 0)
                if rows.size == 0:
                    continue
                block = np.asarray(x[start:stop, :], dtype=np.float32)
                selected = block[rows, :]
                selected_idx = mapped[rows]
                for idx in np.unique(selected_idx):
                    sub = selected[selected_idx == idx, :]
                    sums[idx, :] += sub.sum(axis=0)
                    counts[idx] += sub.shape[0]
                chunks_with_rows += 1
            means = np.zeros_like(sums, dtype=np.float32)
            mask = counts > 0
            means[mask, :] = (sums[mask, :] / counts[mask, None]).astype(np.float32)

        for idx, drug in enumerate(drugs):
            if counts[idx] <= 0:
                continue
            profile_x.append(means[idx, :])
            profile_cell.append(cell_line)
            profile_drug.append(drug)
            profile_count.append(int(counts[idx]))
            profile_id.append(f"{cell_line}::{drug}")
            audit_rows.append({
                "profile_id": f"{cell_line}::{drug}",
                "cell_line": cell_line,
                "drug": drug,
                "protocol_split": split_label.get(drug, ""),
                "n_single_cells": int(counts[idx]),
                "entered_pseudobulk": True,
                "input_source": str(h5ad_path),
            })
        missing = [drug for idx, drug in enumerate(drugs) if counts[idx] <= 0]
        for drug in missing:
            audit_rows.append({
                "profile_id": f"{cell_line}::{drug}",
                "cell_line": cell_line,
                "drug": drug,
                "protocol_split": split_label.get(drug, ""),
                "n_single_cells": 0,
                "entered_pseudobulk": False,
                "input_source": str(h5ad_path),
            })
        print(json.dumps({
            "cell_line": cell_line,
            "h5ad_path": str(h5ad_path),
            "n_obs": n_obs,
            "n_protocol_drugs": len(drugs),
            "n_profiles": int(np.sum(counts > 0)),
            "chunks_with_rows": chunks_with_rows,
            "elapsed_s": round(time.time() - t0, 2),
        }, sort_keys=True), flush=True)

    x_all = np.stack(profile_x, axis=0).astype(np.float32)
    np.savez_compressed(
        source_dir / "Round386_x128_mean_pseudobulk_profiles.npz",
        x=x_all,
        cell_line=np.asarray(profile_cell, dtype=object),
        drug=np.asarray(profile_drug, dtype=object),
        profile_id=np.asarray(profile_id, dtype=object),
        n_single_cells=np.asarray(profile_count, dtype=np.int64),
        feature=np.asarray([f"x128_{i}" for i in range(x_all.shape[1])], dtype=object),
    )
    write_tsv(audit_rows, source_dir / "Round386_x128_pseudobulk_input_audit.tsv")

    lock = {
        "round": "Round386",
        "purpose": "X128 pseudo-bulk adapter comparison against Round260 single-cell adapter",
        "target_cells": list(TARGET_CELLS),
        "input_source": "existing Tahoe processed 128-dimensional single-cell h5ad files",
        "aggregation": "mean X128 feature vector by cell_line and normalized drug",
        "raw_tahoe_h5ad_used": False,
        "x128_substitution_used": True,
        "protocol_json": str(PROTOCOL_JSON),
        "protocol_sha256": sha256_file(PROTOCOL_JSON),
        "target_embedding_path": str(TARGET_EMBEDDING_PATH),
        "target_embedding_sha256": sha256_file(TARGET_EMBEDDING_PATH),
        "dti_path": str(DTI_PATH),
        "dti_sha256": sha256_file(DTI_PATH),
        "pseudobulk_npz": str(source_dir / "Round386_x128_mean_pseudobulk_profiles.npz"),
        "input_audit": str(source_dir / "Round386_x128_pseudobulk_input_audit.tsv"),
    }
    (out_dir / "method_lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")


class PseudoBulkAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, out_dim),
        )
        self.logit_scale_raw = nn.Parameter(torch.tensor(np.log(np.exp(10.0) - 1.0), dtype=torch.float32))

    def forward_query(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder(x), dim=-1)

    def forward_logits_batch(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        q = self.forward_query(x)
        scale = F.softplus(self.logit_scale_raw) + 1e-6
        return scale * q @ target.T


class TrainContext:
    def __init__(self, out_dir: Path, cell_line: str, device: torch.device):
        self.out_dir = out_dir
        self.cell_line = cell_line
        self.device = device
        self.protocol = load_protocol()
        self.target_df = pd.read_csv(TARGET_EMBEDDING_PATH, index_col=0)
        self.target_df = self.target_df[~self.target_df.index.duplicated(keep="first")]
        self.target_df.index = self.target_df.index.astype(str)
        if tuple(self.target_df.shape) != (16583, 256):
            raise RuntimeError(f"Expected target embedding (16583, 256), got {self.target_df.shape}")
        self.all_genes = list(self.target_df.index)
        self.gene_to_global = {g: i for i, g in enumerate(self.all_genes)}
        self.w_all = F.normalize(torch.tensor(self.target_df.values.astype(np.float32), device=device), dim=1)

        npz_path = out_dir / "source_data" / "Round386_x128_mean_pseudobulk_profiles.npz"
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)
        npz = np.load(npz_path, allow_pickle=True)
        cells = npz["cell_line"].astype(str)
        drugs = npz["drug"].astype(str)
        x = npz["x"].astype(np.float32)
        mask = cells == cell_line
        self.x_by_drug = {drugs[i]: x[i] for i in np.flatnonzero(mask)}
        self.n_cells_by_drug = {drugs[i]: int(npz["n_single_cells"][i]) for i in np.flatnonzero(mask)}
        if not self.x_by_drug:
            raise RuntimeError(f"No X128 pseudo-bulk profiles for {cell_line}")
        self.in_dim = int(x.shape[1])

        seen_targets_raw = [str(g) for g in self.protocol["seen_targets"]]
        self.seen_targets = [g for g in seen_targets_raw if g in self.gene_to_global]
        self.seen_name2idx = {g: i for i, g in enumerate(self.seen_targets)}
        seen_global_idx = torch.tensor([self.gene_to_global[g] for g in self.seen_targets], device=device)
        self.w_seen = F.normalize(self.w_all[seen_global_idx], dim=1)

        self.fit_drugs = np.asarray([norm_drug(d) for d in self.protocol["fit_drugs"]])
        self.val_drugs = np.asarray([norm_drug(d) for d in self.protocol["val_drugs"]])
        self.test_drugs = np.asarray([norm_drug(d) for d in self.protocol["test_drugs"]])
        self.exclusive_drugs = np.asarray([norm_drug(d) for d in self.protocol["exclusive_drugs"]])
        self.mixed_drugs = np.asarray([norm_drug(d) for d in self.protocol["mixed_drugs"]])
        self.holdout_targets_set = set([str(g) for g in self.protocol["holdout_targets"]])
        assert len(set(self.fit_drugs) & set(self.test_drugs)) == 0
        assert len(set(self.protocol["seen_targets"]) & set(self.protocol["holdout_targets"])) == 0

        dti = pd.read_csv(DTI_PATH, index_col=0)
        dti.columns = [c.lower() for c in dti.columns]
        if "drug" not in dti.columns or "id" not in dti.columns:
            raise RuntimeError("tahoe_dti.csv must include drug and id columns")
        drug2targets = defaultdict(list)
        for _, row in dti.iterrows():
            drug = norm_drug(row["drug"])
            for target in parse_targets_cell(row["id"]):
                if target in self.gene_to_global and target not in drug2targets[drug]:
                    drug2targets[drug].append(target)
        self.drug2targets = {d: sorted(ts) for d, ts in drug2targets.items() if ts}

        self.y_seen_by_drug = {}
        self.holdout_idx_by_drug = {}
        for drug, targets in self.drug2targets.items():
            vec = np.zeros(len(self.seen_targets), dtype=np.float32)
            for target in targets:
                if target in self.seen_name2idx:
                    vec[self.seen_name2idx[target]] = 1.0
            self.y_seen_by_drug[drug] = torch.tensor(vec, device=device)
            self.holdout_idx_by_drug[drug] = [
                self.gene_to_global[t] for t in targets
                if t in self.gene_to_global and t in self.holdout_targets_set
            ]

        self.available_fit_drugs = np.asarray([
            d for d in self.fit_drugs if d in self.x_by_drug and d in self.y_seen_by_drug
        ])
        self.fit_drugs_no_gt10 = np.asarray([
            d for d in self.available_fit_drugs if int(self.y_seen_by_drug[d].sum().item()) <= 10
        ])
        if len(self.available_fit_drugs) == 0:
            raise RuntimeError(f"No fit drugs available for {cell_line}")
        fit_label_matrix = np.stack([self.y_seen_by_drug[d].detach().cpu().numpy() for d in self.available_fit_drugs])
        pos = np.clip(fit_label_matrix.mean(axis=0), 1e-4, 1.0)
        pos_weight = np.clip((1.0 - pos) / pos, 1.0, 20.0)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device, dtype=torch.float32))
        self.qa = {
            "cell_line": cell_line,
            "input_dim": self.in_dim,
            "target_universe_size": int(self.target_df.shape[0]),
            "target_embedding_dim": int(self.target_df.shape[1]),
            "seen_targets": int(len(self.seen_targets)),
            "pseudobulk_drug_profiles": int(len(self.x_by_drug)),
            "fit_available_for_training": int(len(self.available_fit_drugs)),
            "fit_no_gt10_seen_targets": int(len(self.fit_drugs_no_gt10)),
            "val": int(len(self.val_drugs)),
            "test": int(len(self.test_drugs)),
            "exclusive": int(len(self.exclusive_drugs)),
            "mixed": int(len(self.mixed_drugs)),
        }


def sample_batch(ctx: TrainContext, drugs_arr: np.ndarray, rng: np.random.RandomState) -> tuple[torch.Tensor, torch.Tensor]:
    drugs = rng.choice(drugs_arr, size=BATCH_SIZE, replace=True)
    xb = np.stack([ctx.x_by_drug[d] for d in drugs], axis=0).astype(np.float32)
    yb = torch.stack([ctx.y_seen_by_drug[d] for d in drugs], dim=0)
    return torch.tensor(xb, device=ctx.device, dtype=torch.float32), yb


def compute_adapter_loss(ctx: TrainContext, logits: torch.Tensor, labels: torch.Tensor, loss_mode: str,
                         rng: np.random.RandomState) -> torch.Tensor:
    if loss_mode == "current_bce_baseline" or loss_mode == "outlier_filter_gt10_bce":
        return ctx.loss_fn(logits, labels)
    if loss_mode == "cap_positive_5_bce":
        sampled = torch.zeros_like(labels)
        for i in range(labels.shape[0]):
            pos = torch.where(labels[i] > 0.5)[0]
            if len(pos) > 5:
                keep_np = rng.choice(pos.detach().cpu().numpy(), size=5, replace=False)
                keep = torch.tensor(keep_np, device=labels.device, dtype=torch.long)
            else:
                keep = pos
            sampled[i, keep] = 1.0
        return ctx.loss_fn(logits, sampled)
    if loss_mode == "drug_normalized_bce":
        pos_mask = labels > 0.5
        neg_mask = ~pos_mask
        pos_loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(labels), reduction="none")
        neg_loss = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(labels), reduction="none")
        pos_branch = (pos_loss * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
        neg_branch = (neg_loss * neg_mask).sum(dim=1) / neg_mask.sum(dim=1).clamp(min=1)
        return (pos_branch + neg_branch).mean()
    if loss_mode == "label_bag_retrieval":
        log_probs = F.log_softmax(logits, dim=1)
        masked = log_probs.masked_fill(labels <= 0.5, -1e9)
        return -torch.logsumexp(masked, dim=1).mean()
    raise ValueError(loss_mode)


def eval_seen_hitk(ctx: TrainContext, model: nn.Module, drugs_arr: np.ndarray, split_name: str) -> tuple[dict, list[dict]]:
    model.eval()
    hits = {k: 0 for k in KS}
    rows = []
    total = 0
    skipped = 0
    with torch.no_grad():
        for drug in drugs_arr:
            positives = [t for t in ctx.drug2targets.get(drug, []) if t in ctx.seen_name2idx]
            if not positives or drug not in ctx.x_by_drug:
                skipped += 1
                rows.append({
                    "split": split_name,
                    "drug": drug,
                    "status": "skipped",
                    "n_seen_positive": len(positives),
                    "n_single_cells_aggregated": int(ctx.n_cells_by_drug.get(drug, 0)),
                })
                continue
            pos_idx = [ctx.seen_name2idx[t] for t in positives]
            xb = torch.tensor(ctx.x_by_drug[drug][None, :], device=ctx.device, dtype=torch.float32)
            scores = model.forward_logits_batch(xb, ctx.w_seen).detach().cpu().numpy()[0]
            order = np.argsort(-scores)
            row = {
                "split": split_name,
                "drug": drug,
                "status": "evaluated",
                "n_seen_positive": len(pos_idx),
                "n_single_cells_aggregated": int(ctx.n_cells_by_drug.get(drug, 0)),
            }
            row.update(rank_diagnostics(scores, pos_idx))
            top_sets = {k: set(order[:k]) for k in KS}
            for k in KS:
                hit = int(any(p in top_sets[k] for p in pos_idx))
                hits[k] += hit
                row[f"hit@{k}"] = hit
            rows.append(row)
            total += 1
    metrics = {
        f"{split_name}_n_total": int(len(drugs_arr)),
        f"{split_name}_n_eval": int(total),
        f"{split_name}_n_skipped": int(skipped),
    }
    for k in KS:
        metrics[f"{split_name}_Hit@{k}"] = round(hits[k] / float(total), 4) if total else 0.0
    return metrics, rows


def eval_holdout_percentiles(ctx: TrainContext, model: nn.Module) -> tuple[dict, list[dict]]:
    model.eval()
    hits = {pct: 0 for pct in HOLDOUT_PERCENTILES}
    rows = []
    total = 0
    all_drugs = [(d, "exclusive") for d in ctx.exclusive_drugs] + [(d, "mixed") for d in ctx.mixed_drugs]
    with torch.no_grad():
        for drug, category in all_drugs:
            hs = ctx.holdout_idx_by_drug.get(drug, [])
            if not hs or drug not in ctx.x_by_drug:
                continue
            xb = torch.tensor(ctx.x_by_drug[drug][None, :], device=ctx.device, dtype=torch.float32)
            scores = model.forward_logits_batch(xb, ctx.w_all).detach().cpu().numpy()[0]
            diag = rank_diagnostics(scores, hs)
            row = {
                "drug": drug,
                "drug_category": category,
                "n_holdout_positive": int(len(hs)),
                "n_single_cells_aggregated": int(ctx.n_cells_by_drug.get(drug, 0)),
                **diag,
            }
            for pct in HOLDOUT_PERCENTILES:
                hit = int(diag["best_percentile"] <= pct)
                hits[pct] += hit
                row[f"hit_top{int(pct * 100)}pct"] = hit
            rows.append(row)
            total += 1
    out = {"holdout_all_n_eval": int(total)}
    for pct in HOLDOUT_PERCENTILES:
        key = f"holdout_all_top{int(pct * 100)}pct"
        out[key] = round(hits[pct] / float(total), 4) if total else 0.0
        out[f"{key}_random"] = pct
    return out, rows


def train_one(ctx: TrainContext, loss_mode: str, seed: int, run_dir: Path) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    model = PseudoBulkAdapter(ctx.in_dim, ctx.target_df.shape[1]).to(ctx.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    rng = np.random.RandomState(seed)
    best_state = None
    best_score = -1.0
    best_epoch = 0
    no_improve = 0
    history = []
    train_drugs = ctx.fit_drugs_no_gt10 if loss_mode == "outlier_filter_gt10_bce" else ctx.available_fit_drugs
    train_start = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        t0 = time.time()
        for _ in range(STEPS_PER_EPOCH):
            xb, yb = sample_batch(ctx, train_drugs, rng)
            optimizer.zero_grad()
            logits = model.forward_logits_batch(xb, ctx.w_seen)
            loss = compute_adapter_loss(ctx, logits, yb, loss_mode, rng)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()
        val_metrics, _ = eval_seen_hitk(ctx, model, ctx.val_drugs, "val")
        score = val_metrics["val_Hit@10"]
        row = {
            "loss_mode": loss_mode,
            "seed": seed,
            "epoch": epoch,
            "train_loss": round(float(np.mean(losses)), 6),
            "epoch_time_s": round(time.time() - t0, 2),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        row.update(val_metrics)
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if score > best_score + 1e-9:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if epoch >= MIN_EPOCHS_BEFORE_STOP and no_improve >= EARLY_STOP_PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), run_dir / "checkpoint.pth")
    train_time = time.time() - train_start
    val_metrics, val_rows = eval_seen_hitk(ctx, model, ctx.val_drugs, "val")
    test_metrics, test_rows = eval_seen_hitk(ctx, model, ctx.test_drugs, "test")
    holdout_metrics, holdout_rows = eval_holdout_percentiles(ctx, model)
    write_tsv(val_rows + test_rows, run_dir / "seen_per_drug.tsv")
    write_tsv(holdout_rows, run_dir / "holdout_per_drug.tsv")
    (run_dir / "history.json").write_text(json.dumps({"history": history}, indent=2), encoding="utf-8")
    (run_dir / "train.log").write_text("\n".join(json.dumps(r, sort_keys=True) for r in history) + "\n", encoding="utf-8")

    result = {
        "cell_line": ctx.cell_line,
        "loss_mode": loss_mode,
        "seed": seed,
        "model_class": model.__class__.__name__,
        "train_fit_drugs_used": int(len(train_drugs)),
        "input_dim": ctx.in_dim,
        "target_embedding_path": str(TARGET_EMBEDDING_PATH),
        "target_embedding_shape": list(ctx.target_df.shape),
        "target_embeddings_frozen": True,
        "protocol_json": str(PROTOCOL_JSON),
        "dti_path": str(DTI_PATH),
        "device": str(ctx.device),
        "best_epoch": int(best_epoch),
        "n_params": int(n_params),
        "train_time_s": round(float(train_time), 1),
        "gene_universe_size": int(ctx.target_df.shape[0]),
        "n_seen_targets": int(len(ctx.seen_targets)),
        "n_test_total": int(len(ctx.test_drugs)),
        "n_test_drugs": int(test_metrics["test_n_eval"]),
        "n_test_skipped": int(test_metrics["test_n_skipped"]),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "holdout_metrics": holdout_metrics,
        "qa_target_space": ctx.qa,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def flatten_result(result: dict) -> dict:
    row = {
        "cell_line": result["cell_line"],
        "loss_mode": result["loss_mode"],
        "seed": result["seed"],
        "model_class": result["model_class"],
        "train_fit_drugs_used": result["train_fit_drugs_used"],
        "best_epoch": result["best_epoch"],
        "n_params": result["n_params"],
        "train_time_s": result["train_time_s"],
        "n_test_total": result["n_test_total"],
        "n_test_eval": result["n_test_drugs"],
        "n_test_skipped": result["n_test_skipped"],
        "n_seen_targets": result["n_seen_targets"],
        "gene_universe_size": result["gene_universe_size"],
    }
    for pct in HOLDOUT_PERCENTILES:
        key = f"holdout_all_top{int(pct * 100)}pct"
        row[key] = result["holdout_metrics"].get(key, np.nan)
    for k in KS:
        row[f"val_Hit@{k}"] = result["val_metrics"].get(f"val_Hit@{k}", np.nan)
        row[f"test_Hit@{k}"] = result["test_metrics"].get(f"test_Hit@{k}", np.nan)
    return row


def select_primary(rows: list[dict]) -> dict:
    primary = [r for r in rows if r["loss_mode"] in PRIMARY_LOSS_MODES]
    return sorted(
        primary,
        key=lambda r: (r["val_Hit@10"], r["val_Hit@5"], r["val_Hit@20"], -r["n_params"]),
        reverse=True,
    )[0]


def train_cell(out_dir: Path, cell_line: str, device_name: str, force: bool) -> None:
    cell_dir = out_dir / f"train_{cell_line.replace('-', '_')}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name if torch.cuda.is_available() and device_name.startswith("cuda") else "cpu")
    ctx = TrainContext(out_dir=out_dir, cell_line=cell_line, device=device)
    rows = []
    for loss_mode in LOSS_MODES:
        for seed in SEEDS:
            run_dir = cell_dir / f"{loss_mode}__seed{seed}"
            result_path = run_dir / "result.json"
            if result_path.exists() and not force:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                print(f"START cell={cell_line} loss_mode={loss_mode} seed={seed}", flush=True)
                result = train_one(ctx, loss_mode, seed, run_dir)
            rows.append(flatten_result(result))
    write_tsv(rows, cell_dir / "sweep_summary.tsv")
    selected = select_primary(rows)
    selected_dir = cell_dir / f"{selected['loss_mode']}__seed{selected['seed']}"
    selected_result = json.loads((selected_dir / "result.json").read_text(encoding="utf-8"))
    (cell_dir / "selected_primary_result.json").write_text(json.dumps(selected_result, indent=2), encoding="utf-8")
    shutil.copy2(selected_dir / "holdout_per_drug.tsv", cell_dir / "selected_holdout_per_drug.tsv")
    shutil.copy2(selected_dir / "seen_per_drug.tsv", cell_dir / "selected_seen_per_drug.tsv")
    source_dir = out_dir / "source_data"
    source_dir.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(source_dir / f"Round386_{cell_line.replace('-', '_')}_x128_pseudobulk_sweep_summary.tsv", sep="\t", index=False)
    selected_holdout = pd.read_csv(selected_dir / "holdout_per_drug.tsv", sep="\t")
    selected_holdout["cell_line"] = cell_line
    selected_holdout["loss_mode"] = selected_result["loss_mode"]
    selected_holdout["seed"] = selected_result["seed"]
    selected_holdout["best_epoch"] = selected_result["best_epoch"]
    selected_holdout.to_csv(source_dir / f"Round386_{cell_line.replace('-', '_')}_selected_holdout_per_drug.tsv", sep="\t", index=False)
    (cell_dir / "qa_target_space_check.json").write_text(json.dumps(ctx.qa, indent=2), encoding="utf-8")
    print("SELECTED", json.dumps(selected, sort_keys=True), flush=True)


def run_all(out_dir: Path, device: str, force: bool) -> None:
    build_x128_pseudobulk(out_dir)
    for cell_line in TARGET_CELLS:
        train_cell(out_dir, cell_line, device, force)
    summarize(out_dir)


def summarize(out_dir: Path) -> None:
    source_dir = out_dir / "source_data"
    rows = []
    holdout_long = []
    for cell_line in TARGET_CELLS:
        cell_dir = out_dir / f"train_{cell_line.replace('-', '_')}"
        selected = json.loads((cell_dir / "selected_primary_result.json").read_text(encoding="utf-8"))
        row = {
            "method": "X128 pseudo-bulk adapter",
            "cell_line": cell_line,
            "loss_mode": selected["loss_mode"],
            "seed": selected["seed"],
            "best_epoch": selected["best_epoch"],
            "seen_test_Hit@5": selected["test_metrics"]["test_Hit@5"],
            "seen_test_Hit@10": selected["test_metrics"]["test_Hit@10"],
            "seen_test_Hit@20": selected["test_metrics"]["test_Hit@20"],
            "seen_test_Hit@50": selected["test_metrics"]["test_Hit@50"],
            "holdout_n_eval": selected["holdout_metrics"]["holdout_all_n_eval"],
        }
        for pct in HOLDOUT_PERCENTILES:
            row[f"holdout_top{int(pct * 100)}pct"] = selected["holdout_metrics"][f"holdout_all_top{int(pct * 100)}pct"]
        rows.append(row)
        h = pd.read_csv(cell_dir / "selected_holdout_per_drug.tsv", sep="\t")
        h["cell_line"] = cell_line
        h["method"] = "X128 pseudo-bulk adapter"
        holdout_long.append(h)
    metric = pd.DataFrame(rows)
    metric.to_csv(source_dir / "Round386_x128_pseudobulk_metric_summary.tsv", sep="\t", index=False)
    holdout = pd.concat(holdout_long, ignore_index=True)
    holdout.to_csv(source_dir / "Round386_x128_pseudobulk_holdout_per_drug.tsv", sep="\t", index=False)

    curve_rows = []
    for pct in HOLDOUT_PERCENTILES:
        col = f"hit_top{int(pct * 100)}pct"
        curve_rows.append({
            "method": "X128 pseudo-bulk adapter",
            "cell_line": "pooled",
            "threshold_percent": pct,
            "n_units": int(len(holdout)),
            "n_success": int(holdout[col].sum()),
            "recovery": float(holdout[col].mean()),
            "candidate_space": 16583,
        })
        for cell, sub in holdout.groupby("cell_line"):
            curve_rows.append({
                "method": "X128 pseudo-bulk adapter",
                "cell_line": cell,
                "threshold_percent": pct,
                "n_units": int(len(sub)),
                "n_success": int(sub[col].sum()),
                "recovery": float(sub[col].mean()),
                "candidate_space": 16583,
            })
    pd.DataFrame(curve_rows).to_csv(source_dir / "Round386_x128_pseudobulk_recovery_curve.tsv", sep="\t", index=False)
    print(metric.to_string(index=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "train-cell", "train-all", "summarize", "run-all"])
    parser.add_argument("--out-dir", default=str(ROUND_ROOT))
    parser.add_argument("--cell-line", choices=list(TARGET_CELLS))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "build":
        build_x128_pseudobulk(out_dir)
    elif args.command == "train-cell":
        if not args.cell_line:
            raise RuntimeError("--cell-line is required for train-cell")
        train_cell(out_dir, args.cell_line, args.device, args.force)
    elif args.command == "train-all":
        for cell_line in TARGET_CELLS:
            train_cell(out_dir, cell_line, args.device, args.force)
    elif args.command == "summarize":
        summarize(out_dir)
    elif args.command == "run-all":
        run_all(out_dir, args.device, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
