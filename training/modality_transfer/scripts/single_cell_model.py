#!/usr/bin/env python
"""Single-cell label-bag calibration against fixed target embeddings."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from metar.runtime import report_metrics, report_progress, report_step, resolve_device


DEFAULT_OUT_DIR = Path("/home/lwn/claude/manuscript/write/sc_tahoe_multicell_reprocess_20260509")
TARGET_EMBEDDING_PATH = Path(
    "/home/lwn/claude/manuscript/write/transfer_direct_h07_20260330/"
    "artifacts/full_raw_v1/label_repr_h07_transfer_full.csv"
)
PROTOCOL_JSON = Path(
    "/home/lwn/claude/manuscript/write/sc_agg_upgrade_20260321/"
    "derived_tables/sc_protocol_lock_v3_20260326.json"
)
DTI_PATH = Path("/home/lwn/claude/sc/data/tahoe_dti.csv")
FORBIDDEN_TARGET_REPR = Path("/home/lwn/claude/sc/data/label_repr_all.csv")

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
HOLDOUT_PERCENTILES = [0.05, 0.10]

CELLS_PER_DRUG = 128
EVAL_CELLS = 300
BATCH_SIZE = 256
MAX_EPOCHS = 16
STEPS_PER_EPOCH = 200
EARLY_STOP_PATIENCE = 4
MIN_EPOCHS_BEFORE_STOP = 4
LR = 3e-4
WEIGHT_DECAY = 1e-5
HIDDEN_DIM = 512
LATENT_DIM = 256
GATE_HIDDEN = 64
DROPOUT = 0.25
SEED_EVAL_CELLS = 20260327

ROUND61_BASELINE = {
    "Hit@1": 0.0833,
    "Hit@5": 0.0833,
    "Hit@10": 0.25,
    "Hit@20": 0.3333,
    "Hit@30": 0.3333,
    "Hit@50": 0.5,
}


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
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


def norm_drug(x) -> str:
    return str(x).strip().lower()


def stable_seed(name: str, base_seed: int) -> int:
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    return (base_seed + int(digest[:8], 16)) % (2**32 - 1)


def parse_targets_cell(x) -> list[str]:
    s = str(x).strip()
    if s in ("", "nan", "none", "null"):
        return []
    try:
        import ast

        v = ast.literal_eval(s)
    except Exception:
        v = s
    if isinstance(v, (list, tuple, set)):
        return [str(t).strip() for t in v if str(t).strip()]
    return [str(v).strip()]


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
        self.logit_scale_raw = nn.Parameter(torch.tensor(np.log(np.exp(10.0) - 1.0), dtype=torch.float32))

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
    for i, drug in enumerate(drugs):
        if drug and drug != "nan":
            pools[drug].append(i)
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


def sample_batch(adata, drugs_arr, cell_pool_by_drug, y_seen_by_drug, rng, device):
    drugs = rng.choice(drugs_arr, size=BATCH_SIZE, replace=True)
    cell_blocks = []
    y = []
    for drug in drugs:
        pool = cell_pool_by_drug[drug]
        replace = len(pool) < CELLS_PER_DRUG
        idxs = rng.choice(pool, size=CELLS_PER_DRUG, replace=replace)
        cell_blocks.append(np.asarray(idxs, dtype=np.int64))
        y.append(y_seen_by_drug[drug])
    cell_blocks = np.stack(cell_blocks, axis=0)
    x = fetch_x_batch(adata, cell_blocks.reshape(-1), device).view(BATCH_SIZE, CELLS_PER_DRUG, -1)
    return x, torch.stack(y, dim=0), drugs


def rank_diagnostics(scores: np.ndarray, pos_indices: list[int]) -> dict:
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


def bin_target_count(n: int) -> str:
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 3:
        return "2-3"
    if n <= 5:
        return "4-5"
    if n <= 10:
        return "6-10"
    return ">10"


def compute_adapter_loss(ctx, logits: torch.Tensor, labels: torch.Tensor, loss_mode: str,
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

    raise ValueError(f"Unknown loss_mode: {loss_mode}")


def eval_seen_hitk(model, adata, drugs_arr, eval_cells_by_drug, drug2targets, seen_name2idx,
                   gene_to_global, w_seen, split_name, device):
    model.eval()
    hits = {k: 0 for k in KS}
    rows = []
    total = 0
    skipped = 0
    with torch.no_grad():
        for completed, drug in enumerate(drugs_arr, start=1):
            positives = [t for t in drug2targets.get(drug, []) if t in seen_name2idx]
            if not positives or drug not in eval_cells_by_drug:
                skipped += 1
                rows.append({
                    "split": split_name,
                    "drug": drug,
                    "status": "skipped",
                    "n_seen_positive": len(positives),
                    "n_eval_cells": int(len(eval_cells_by_drug.get(drug, []))),
                })
                report_progress(completed, len(drugs_arr), label=f"single-cell {split_name} evaluation drugs")
                continue
            pos_idx = [seen_name2idx[t] for t in positives]
            x = fetch_x_batch(adata, eval_cells_by_drug[drug], device).view(1, -1, adata.n_vars)
            scores = model.forward_logits_batch(x, w_seen).detach().cpu().numpy()[0]
            order = np.argsort(-scores)
            row = {
                "split": split_name,
                "drug": drug,
                "status": "evaluated",
                "n_seen_positive": len(pos_idx),
                "n_eval_cells": int(len(eval_cells_by_drug[drug])),
            }
            row.update(rank_diagnostics(scores, pos_idx))
            for k in KS:
                hit = int(any(p in set(order[:k]) for p in pos_idx))
                hits[k] += hit
                row[f"hit@{k}"] = hit
            rows.append(row)
            total += 1
            report_progress(completed, len(drugs_arr), label=f"single-cell {split_name} evaluation drugs")
    metrics = {
        f"{split_name}_n_total": int(len(drugs_arr)),
        f"{split_name}_n_eval": int(total),
        f"{split_name}_n_skipped": int(skipped),
    }
    for k in KS:
        metrics[f"{split_name}_Hit@{k}"] = round(hits[k] / float(total), 4) if total else 0.0
    return metrics, rows


def eval_holdout_all_percentiles(model, adata, exclusive_drugs, mixed_drugs, eval_cells_by_drug,
                                 holdout_idx_by_drug, w_all, device):
    model.eval()
    hits = {pct: 0 for pct in HOLDOUT_PERCENTILES}
    per_drug = []
    total = 0
    all_drugs = [(d, "exclusive") for d in exclusive_drugs] + [(d, "mixed") for d in mixed_drugs]
    with torch.no_grad():
        for completed, (drug, category) in enumerate(all_drugs, start=1):
            hs = holdout_idx_by_drug.get(drug, [])
            if not hs or drug not in eval_cells_by_drug:
                report_progress(completed, len(all_drugs), label="single-cell holdout evaluation drugs")
                continue
            x = fetch_x_batch(adata, eval_cells_by_drug[drug], device).view(1, -1, adata.n_vars)
            scores = model.forward_logits_batch(x, w_all).detach().cpu().numpy()[0]
            diag = rank_diagnostics(scores, hs)
            total += 1
            row = {
                "drug": drug,
                "drug_category": category,
                "n_holdout_positive": int(len(hs)),
                "n_eval_cells": int(len(eval_cells_by_drug[drug])),
                **diag,
            }
            for pct in HOLDOUT_PERCENTILES:
                hit = int(diag["best_percentile"] <= pct)
                hits[pct] += hit
                row[f"hit_top{int(pct * 100)}pct"] = hit
            per_drug.append(row)
            report_progress(completed, len(all_drugs), label="single-cell holdout evaluation drugs")
    out = {
        "holdout_all_n_eval": int(total),
    }
    for pct in HOLDOUT_PERCENTILES:
        key = f"holdout_all_top{int(pct * 100)}pct"
        out[key] = round(hits[pct] / float(total), 4) if total else 0.0
        out[f"{key}_random"] = pct
    return out, per_drug


class SweepContext:
    def __init__(self, device: torch.device, out_dir: Path, adata_path: Path, cell_line: str):
        self.device = device
        self.out_dir = out_dir
        self.adata_path = adata_path
        self.cell_line = cell_line
        report_step(f"Reading single-cell target embeddings: {TARGET_EMBEDDING_PATH}")
        self.target_df = pd.read_csv(TARGET_EMBEDDING_PATH, index_col=0)
        self.target_df = self.target_df[~self.target_df.index.duplicated(keep="first")]
        self.target_df.index = self.target_df.index.astype(str)
        if tuple(self.target_df.shape) != (16583, 256):
            raise RuntimeError(f"Locked target table must be (16583, 256), got {self.target_df.shape}")
        if TARGET_EMBEDDING_PATH.resolve() == FORBIDDEN_TARGET_REPR.resolve():
            raise RuntimeError("Forbidden old label representation selected as target table")
        self.all_genes = list(self.target_df.index)
        self.gene_to_global = {g: i for i, g in enumerate(self.all_genes)}
        self.w_all = F.normalize(torch.tensor(self.target_df.values.astype(np.float32), device=device), dim=1)

        report_step(f"Reading locked single-cell protocol: {PROTOCOL_JSON}")
        with PROTOCOL_JSON.open("r", encoding="utf-8") as f:
            self.protocol = json.load(f)
        if self.protocol.get("target_column") != "id":
            raise RuntimeError(f"Expected target_column='id', got {self.protocol.get('target_column')}")
        if self.protocol.get("protocol_tag") != "v3_corrected_gene_targets_fixed_cells":
            raise RuntimeError(f"Unexpected protocol tag: {self.protocol.get('protocol_tag')}")

        report_step(f"Reading single-cell input: {adata_path}")
        self.adata = ad.read_h5ad(adata_path)
        if self.adata.n_vars != 128:
            raise RuntimeError(f"Expected 128 Tahoe features, got {self.adata.n_vars}")
        self.adata.obs["drug"] = self.adata.obs["drug"].astype(str).map(norm_drug)
        self.cell_pool_by_drug = build_cell_pool_from_adata(self.adata)
        self.eval_cells_by_drug = build_eval_cells(self.cell_pool_by_drug)

        seen_targets_raw = [str(g) for g in self.protocol["seen_targets"]]
        self.seen_targets = [g for g in seen_targets_raw if g in self.gene_to_global]
        self.missing_seen_targets = [g for g in seen_targets_raw if g not in self.gene_to_global]
        self.seen_name2idx = {g: i for i, g in enumerate(self.seen_targets)}
        seen_global_idx = torch.tensor([self.gene_to_global[g] for g in self.seen_targets], device=device)
        self.w_seen = F.normalize(self.w_all[seen_global_idx], dim=1)

        self.train_drugs_outer = np.asarray([norm_drug(d) for d in self.protocol["train_drugs_outer"]])
        self.fit_drugs = np.asarray([norm_drug(d) for d in self.protocol["fit_drugs"]])
        self.val_drugs = np.asarray([norm_drug(d) for d in self.protocol["val_drugs"]])
        self.test_drugs = np.asarray([norm_drug(d) for d in self.protocol["test_drugs"]])
        self.exclusive_drugs = np.asarray([norm_drug(d) for d in self.protocol["exclusive_drugs"]])
        self.mixed_drugs = np.asarray([norm_drug(d) for d in self.protocol["mixed_drugs"]])
        self.holdout_targets_set = set([str(g) for g in self.protocol["holdout_targets"]])
        assert len(set(self.train_drugs_outer) & set(self.test_drugs)) == 0
        assert len(set(self.protocol["seen_targets"]) & set(self.protocol["holdout_targets"])) == 0

        report_step(f"Building single-cell labels from {DTI_PATH}")
        self.scdrug = pd.read_csv(DTI_PATH, index_col=0)
        self.scdrug.columns = [c.lower() for c in self.scdrug.columns]
        if "drug" not in self.scdrug.columns or "id" not in self.scdrug.columns:
            raise RuntimeError("tahoe_dti.csv must include drug and id columns")

        drug2targets = defaultdict(list)
        for _, row in self.scdrug.iterrows():
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
                self.gene_to_global[t] for t in targets if t in self.gene_to_global and t in self.holdout_targets_set
            ]

        self.available_fit_drugs = np.asarray([d for d in self.fit_drugs if d in self.cell_pool_by_drug and d in self.y_seen_by_drug])
        self.missing_fit_drugs = [d for d in self.fit_drugs if d not in set(self.available_fit_drugs)]
        if len(self.available_fit_drugs) == 0:
            raise RuntimeError(f"No fit drugs with cells and labels for {cell_line}")

        fit_label_matrix = np.stack([self.y_seen_by_drug[d].detach().cpu().numpy() for d in self.available_fit_drugs], axis=0)
        pos = np.clip(fit_label_matrix.mean(axis=0), 1e-4, 1.0)
        pos_weight = np.clip((1.0 - pos) / pos, 1.0, 20.0)
        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, device=device, dtype=torch.float32)
        )
        self.pos_weight_tensor = torch.tensor(pos_weight, device=device, dtype=torch.float32)
        self.fit_drugs_no_gt10 = np.asarray([
            d for d in self.available_fit_drugs if int(self.y_seen_by_drug[d].sum().item()) <= 10
        ])

        split_availability = {}
        for split, drugs in [
            ("fit", self.fit_drugs),
            ("val", self.val_drugs),
            ("test", self.test_drugs),
            ("exclusive", self.exclusive_drugs),
            ("mixed", self.mixed_drugs),
        ]:
            with_cells = [d for d in drugs if d in self.cell_pool_by_drug]
            with_cells_and_labels = [d for d in drugs if d in self.cell_pool_by_drug and d in self.drug2targets]
            split_availability[split] = {
                "protocol_n": int(len(drugs)),
                "with_cells": int(len(with_cells)),
                "with_cells_and_labels": int(len(with_cells_and_labels)),
                "missing_cells": [d for d in drugs if d not in self.cell_pool_by_drug],
            }

        self.qa = {
            "cell_line": cell_line,
            "adata_path": str(adata_path),
            "adata_shape": [int(self.adata.n_obs), int(self.adata.n_vars)],
            "drug_pool_count": int(len(self.cell_pool_by_drug)),
            "target_universe_size": int(self.target_df.shape[0]),
            "target_embedding_dim": int(self.target_df.shape[1]),
            "protocol_seen_targets_raw": int(len(seen_targets_raw)),
            "protocol_seen_targets_in_full_target_table": int(len(self.seen_targets)),
            "protocol_seen_targets_missing": self.missing_seen_targets,
            "split_counts": {
                "fit": int(len(self.fit_drugs)),
                "fit_available_for_training": int(len(self.available_fit_drugs)),
                "fit_no_gt10_seen_targets": int(len(self.fit_drugs_no_gt10)),
                "val": int(len(self.val_drugs)),
                "test": int(len(self.test_drugs)),
                "exclusive": int(len(self.exclusive_drugs)),
                "mixed": int(len(self.mixed_drugs)),
            },
            "split_availability": split_availability,
            "cell_pool_source": "rebuilt from selected h5ad obs['drug']; A549 locked cell-pool indices were not used",
        }

    def write_global_lock(self):
        lock = {
            "purpose": "multi-cell-line single-cell adapter evaluation against locked full bulk target embeddings",
            "cell_line": self.cell_line,
            "primary_loss_modes": PRIMARY_LOSS_MODES,
            "sensitivity_loss_modes": SENSITIVITY_LOSS_MODES,
            "seeds": SEEDS,
            "target_embedding_path": str(TARGET_EMBEDDING_PATH),
            "target_embedding_sha256": sha256_file(TARGET_EMBEDDING_PATH),
            "target_shape": list(self.target_df.shape),
            "target_embeddings_frozen": True,
            "forbidden_target_repr": str(FORBIDDEN_TARGET_REPR),
            "protocol_json": str(PROTOCOL_JSON),
            "protocol_sha256": sha256_file(PROTOCOL_JSON),
            "cell_pool_json_gz": None,
            "cell_pool_sha256": None,
            "cell_pool_source": "rebuilt from selected h5ad obs['drug']",
            "adata_path": str(self.adata_path),
            "adata_sha256": sha256_file(self.adata_path),
            "dti_path": str(DTI_PATH),
            "dti_sha256": sha256_file(DTI_PATH),
            "target_column": "id",
            "loss_modes": {
                "current_bce_baseline": "BCEWithLogitsLoss + fixed class pos_weight",
                "drug_normalized_bce": "drug-level positive mean plus negative mean",
                "label_bag_retrieval": "bag-level logsumexp retrieval loss over seen targets",
                "cap_positive_5_bce": "BCE sensitivity with at most five positive targets sampled per drug per step",
                "outlier_filter_gt10_bce": "BCE sensitivity excluding fit drugs with more than ten seen targets",
            },
            "selection_rule": "val_Hit@10, then val_Hit@5, then val_Hit@20, then smaller n_params",
            "fixed_train_params": {
                "cells_per_drug_train": CELLS_PER_DRUG,
                "cells_per_drug_eval": EVAL_CELLS,
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "steps_per_epoch": STEPS_PER_EPOCH,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "early_stop_metric": "val_Hit@10",
            },
            "qa": self.qa,
        }
        (self.out_dir / "method_lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
        (self.out_dir / "qa_target_space_check.json").write_text(json.dumps(self.qa, indent=2), encoding="utf-8")


def train_one(ctx: SweepContext, loss_mode: str, seed: int, run_dir: Path) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    model = build_model(loss_mode, in_dim=ctx.adata.n_vars, out_dim=ctx.target_df.shape[1]).to(ctx.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=(ctx.device.type == "cuda"))
    rng = np.random.RandomState(seed)

    best_state = None
    best_score = -1.0
    best_epoch = 0
    no_improve = 0
    history = []
    train_start = time.time()
    train_drugs = ctx.fit_drugs_no_gt10 if loss_mode == "outlier_filter_gt10_bce" else ctx.available_fit_drugs
    run_label = f"single-cell {ctx.cell_line} {loss_mode} seed={seed}"
    report_step(f"Training {run_label}; device={ctx.device} max_epochs={MAX_EPOCHS} steps_per_epoch={STEPS_PER_EPOCH} batch_size={BATCH_SIZE} lr={LR}")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        t0 = time.time()
        for step in range(1, STEPS_PER_EPOCH + 1):
            xb, yb, _ = sample_batch(ctx.adata, train_drugs, ctx.cell_pool_by_drug, ctx.y_seen_by_drug, rng, ctx.device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(ctx.device.type == "cuda")):
                logits = model.forward_logits_batch(xb, ctx.w_seen)
                loss = compute_adapter_loss(ctx, logits, yb, loss_mode, rng)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
            report_progress(step, STEPS_PER_EPOCH, label=f"{run_label} epoch {epoch} batches", metrics={"loss": losses[-1]})
        scheduler.step()

        report_step(f"Validating {run_label} epoch {epoch}")
        val_metrics, _ = eval_seen_hitk(
            model, ctx.adata, ctx.val_drugs, ctx.eval_cells_by_drug, ctx.drug2targets,
            ctx.seen_name2idx, ctx.gene_to_global, ctx.w_seen, "val", ctx.device
        )
        score = val_metrics["val_Hit@10"]
        row = {
            "loss_mode": loss_mode,
            "seed": seed,
            "epoch": epoch,
            "train_loss": round(float(np.mean(losses)), 6),
            "epoch_time_s": round(time.time() - t0, 2),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        if hasattr(model, "beta_logit"):
            row["beta"] = round(float(torch.sigmoid(model.beta_logit).item()), 6)
        row.update(val_metrics)
        history.append(row)
        report_progress(epoch, MAX_EPOCHS, label=f"{run_label} epochs", metrics={"train_loss": row["train_loss"], "val_Hit@10": score})
        print(json.dumps(row, sort_keys=True), flush=True)
        if score > best_score + 1e-9:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if epoch >= MIN_EPOCHS_BEFORE_STOP and no_improve >= EARLY_STOP_PATIENCE:
            print(f"Early stop {loss_mode} seed={seed} at epoch {epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), run_dir / "checkpoint.pth")
    train_time = time.time() - train_start

    report_step(f"Evaluating selected {run_label}; best_epoch={best_epoch}")
    val_metrics, val_rows = eval_seen_hitk(
        model, ctx.adata, ctx.val_drugs, ctx.eval_cells_by_drug, ctx.drug2targets,
        ctx.seen_name2idx, ctx.gene_to_global, ctx.w_seen, "val", ctx.device
    )
    test_metrics, test_rows = eval_seen_hitk(
        model, ctx.adata, ctx.test_drugs, ctx.eval_cells_by_drug, ctx.drug2targets,
        ctx.seen_name2idx, ctx.gene_to_global, ctx.w_seen, "test", ctx.device
    )
    holdout_metrics, holdout_rows = eval_holdout_all_percentiles(
        model, ctx.adata, ctx.exclusive_drugs, ctx.mixed_drugs, ctx.eval_cells_by_drug,
        ctx.holdout_idx_by_drug, ctx.w_all, ctx.device
    )
    pd.DataFrame(val_rows + test_rows).to_csv(run_dir / "seen_per_drug.tsv", sep="\t", index=False)
    pd.DataFrame(holdout_rows).to_csv(run_dir / "holdout_per_drug.tsv", sep="\t", index=False)
    (run_dir / "history.json").write_text(json.dumps({"loss_mode": loss_mode, "seed": seed, "history": history}, indent=2), encoding="utf-8")
    (run_dir / "train.log").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in history) + "\n",
        encoding="utf-8",
    )

    metar = {f"Hit@{k}": test_metrics[f"test_Hit@{k}"] for k in KS}
    random_base = {f"Hit@{k}": round(k / float(len(ctx.seen_targets)), 4) for k in KS}
    result = {
        "cell_line": ctx.cell_line,
        "loss_mode": loss_mode,
        "seed": seed,
        "model_class": model.__class__.__name__,
        "train_fit_drugs_used": int(len(train_drugs)),
        "target_embedding_path": str(TARGET_EMBEDDING_PATH),
        "target_embedding_shape": list(ctx.target_df.shape),
        "target_embeddings_frozen": True,
        "used_forbidden_old_target_repr": False,
        "protocol_json": str(PROTOCOL_JSON),
        "cell_pool_source": "rebuilt from selected h5ad obs['drug']",
        "adata_path": str(ctx.adata_path),
        "target_column": "id",
        "device": str(ctx.device),
        "best_epoch": best_epoch,
        "n_params": int(n_params),
        "train_time_s": round(float(train_time), 1),
        "gene_universe_size": int(ctx.target_df.shape[0]),
        "n_seen_targets": int(len(ctx.seen_targets)),
        "n_test_total": int(len(ctx.test_drugs)),
        "n_test_drugs": int(test_metrics["test_n_eval"]),
        "n_test_skipped": int(test_metrics["test_n_skipped"]),
        "metar": metar,
        "random": random_base,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "holdout_metrics": holdout_metrics,
        "qa_target_space": ctx.qa,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for split, metrics in (("validation", val_metrics), ("test", test_metrics), ("holdout", holdout_metrics)):
        report_metrics(
            {key: value for key, value in metrics.items()
             if key.endswith(("Hit@5", "Hit@10", "Hit@20", "top5pct", "top10pct"))},
            label=f"{run_label} selected {split} metrics",
        )
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
        row[key] = result["holdout_metrics"][key]
    for k in KS:
        row[f"val_Hit@{k}"] = result["val_metrics"][f"val_Hit@{k}"]
        row[f"test_Hit@{k}"] = result["test_metrics"][f"test_Hit@{k}"]
        row[f"random_Hit@{k}"] = result["random"][f"Hit@{k}"]
    return row


def select_primary(rows: list[dict]) -> dict:
    primary_rows = [r for r in rows if r["loss_mode"] in PRIMARY_LOSS_MODES]
    return sorted(
        primary_rows,
        key=lambda r: (
            r["val_Hit@10"],
            r["val_Hit@5"],
            r["val_Hit@20"],
            -r["n_params"],
        ),
        reverse=True,
    )[0]


def write_burden_stratified_recovery(ctx: SweepContext, holdout_path: Path, out_path: Path) -> None:
    holdout = pd.read_csv(holdout_path, sep="\t")
    full_counts = {drug: len([t for t in targets if t in ctx.gene_to_global])
                   for drug, targets in ctx.drug2targets.items()}
    holdout["n_full_targets"] = holdout["drug"].map(full_counts).fillna(0).astype(int)
    holdout["full_target_bin"] = holdout["n_full_targets"].map(bin_target_count)
    holdout["holdout_positive_bin"] = holdout["n_holdout_positive"].astype(int).map(bin_target_count)
    rows = []
    for bin_col in ["drug_category", "holdout_positive_bin", "full_target_bin"]:
        for value, sub in holdout.groupby(bin_col):
            row = {
                "bin_type": bin_col,
                "bin": value,
                "drugs": int(len(sub)),
                "mean_full_targets": float(sub["n_full_targets"].mean()),
                "mean_holdout_positive": float(sub["n_holdout_positive"].mean()),
            }
            for pct in HOLDOUT_PERCENTILES:
                col = f"hit_top{int(pct * 100)}pct"
                row[f"top{int(pct * 100)}pct_recovery"] = float(sub[col].mean())
                row[f"random_top{int(pct * 100)}pct"] = pct
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)


def write_summaries(out_dir: Path, rows: list[dict], ctx: SweepContext) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sweep_summary.tsv", sep="\t", index=False)

    metric_cols = [
        c for c in df.columns
        if c.startswith("test_Hit@") or c.startswith("val_Hit@") or c.startswith("holdout_all_top")
    ]
    grouped_rows = []
    for loss_mode, g in df.groupby("loss_mode"):
        row = {
            "loss_mode": loss_mode,
            "run_type": "primary" if loss_mode in PRIMARY_LOSS_MODES else "sensitivity",
            "n_runs": int(len(g)),
        }
        for col in metric_cols:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=0))
        grouped_rows.append(row)
    pd.DataFrame(grouped_rows).to_csv(out_dir / "loss_mode_mean_std.tsv", sep="\t", index=False)

    selected = select_primary(rows)
    selected_dir = out_dir / f"{selected['loss_mode']}__seed{selected['seed']}"
    selected_result = json.loads((selected_dir / "result.json").read_text(encoding="utf-8"))
    (out_dir / "selected_primary_result.json").write_text(json.dumps(selected_result, indent=2), encoding="utf-8")
    write_burden_stratified_recovery(ctx, selected_dir / "holdout_per_drug.tsv", out_dir / "target_burden_stratified_recovery.tsv")

    passes = (
        selected["test_Hit@5"] >= ROUND61_BASELINE["Hit@5"]
        and (
            selected["test_Hit@10"] > ROUND61_BASELINE["Hit@10"]
            or selected["test_Hit@20"] > ROUND61_BASELINE["Hit@20"]
        )
    )
    lines = [
        f"# Round178 {ctx.cell_line} single-cell label-bag calibration report",
        "",
        "## Selection rule",
        "",
        "Primary selection used validation only: `val_Hit@10`, then `val_Hit@5`, then `val_Hit@20`, then smaller parameter count.",
        "",
        "## Selected primary",
        "",
        f"- Loss mode: `{selected['loss_mode']}`",
        f"- Seed: `{selected['seed']}`",
        f"- Val Hit@10: {selected['val_Hit@10']:.4f}",
        f"- Test Hit@5: {selected['test_Hit@5']:.4f}",
        f"- Test Hit@10: {selected['test_Hit@10']:.4f}",
        f"- Test Hit@20: {selected['test_Hit@20']:.4f}",
        f"- Holdout top-5%: {selected['holdout_all_top5pct']:.4f}",
        f"- Holdout top-10%: {selected['holdout_all_top10pct']:.4f}",
        f"- Acceptance criterion met relative to A549 Round61 closed-seen threshold: {'yes' if passes else 'no'}",
        "",
        "## Data availability audit",
        "",
        f"- Adata: `{ctx.adata_path}`",
        f"- Adata shape: `{ctx.adata.n_obs} x {ctx.adata.n_vars}`",
        f"- Fit drugs available for training: `{len(ctx.available_fit_drugs)}/{len(ctx.fit_drugs)}`",
        f"- Cell pools rebuilt from selected h5ad: `yes`",
        "",
        "## Interpretation guardrail",
        "",
        "All loss modes use the same locked target embeddings, split and evaluation code. Holdout/open-world metrics were not used for model selection.",
    ]
    (out_dir / f"round178_{ctx.cell_line.lower().replace('-', '_')}_sc_label_bag_calibration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata-path", required=True)
    parser.add_argument("--cell-line", required=True)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--loss-mode", choices=LOSS_MODES + ["all"], default="all")
    parser.add_argument("--seed", type=int, default=-1, help="-1 means all registered seeds")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(resolve_device(args.device))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_script = Path(__file__).resolve()
    dst_script = (out_dir / "run_sc_label_bag_calibration_multicell.py").resolve()
    if src_script != dst_script:
        shutil.copy2(src_script, dst_script)
    print(f"device={device}", flush=True)
    print(f"out_dir={out_dir}", flush=True)
    print(f"adata_path={args.adata_path}", flush=True)

    ctx = SweepContext(device=device, out_dir=out_dir, adata_path=Path(args.adata_path), cell_line=args.cell_line)
    ctx.write_global_lock()
    print("target/protocol lock ok", json.dumps(ctx.qa, sort_keys=True), flush=True)

    loss_modes = LOSS_MODES if args.loss_mode == "all" else [args.loss_mode]
    seeds = SEEDS if args.seed < 0 else [args.seed]
    rows = []

    for loss_mode in loss_modes:
        for seed in seeds:
            run_dir = out_dir / f"{loss_mode}__seed{seed}"
            result_path = run_dir / "result.json"
            if result_path.exists() and not args.force:
                print(f"SKIP existing {run_dir}", flush=True)
                result = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                print(f"START loss_mode={loss_mode} seed={seed}", flush=True)
                result = train_one(ctx, loss_mode, seed, run_dir)
                print("RESULT", json.dumps(flatten_result(result), sort_keys=True), flush=True)
            rows.append(flatten_result(result))

    write_summaries(out_dir, rows, ctx)
    print(f"saved sweep summary to {out_dir / 'sweep_summary.tsv'}", flush=True)
    print(f"saved selected primary to {out_dir / 'selected_primary_result.json'}", flush=True)


if __name__ == "__main__":
    main()
