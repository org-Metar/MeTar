#!/usr/bin/env python3
"""Train a response encoder and PPI target tower over the full candidate space."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score
from torch import nn


ROUND_ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROUND_ROOT / "source_data"
QA = ROUND_ROOT / "qa"
CHECKPOINTS = ROUND_ROOT / "checkpoints"

REMOTE_WRITE_ROOT = Path("/home/lwn/claude/manuscript/write")
H07_WRAPPER = REMOTE_WRITE_ROOT / "bulk_sharedsample_indunseenlabel_20260330_h07" / "scripts" / "train_bulk_sharedsample_indunseenlabel_v1_20260330.py"
H07_CACHE = REMOTE_WRITE_ROOT / "claude_round209_fullspace_a01_extrapolation_20260517" / "cache" / "h07_prepared_data.pt"
H07_CKPT = (
    REMOTE_WRITE_ROOT
    / "bulk_sharedsample_indunseenlabel_20260330_h07_w060_u8e4_med"
    / "checkpoints"
    / "best_unseen_h07_indlabel_linear_e04init_w060_u8e4_med.pth"
)

CELL_LINE = "PC3"
SEED = 20260517
EXPECTED_COUNTS = {
    "n_genes_full_universe": 16583,
    "n_train_label_genes": 668,
    "pc3_inner_train": 17103,
    "pc3_val": 1555,
    "pc3_test": 3829,
    "pc3_holdout_val": 1113,
    "pc3_holdout_test": 2288,
}

VARIANTS: dict[str, dict[str, Any]] = {
    "m0_fixed_trained_target": {
        "target_mode": "trained_target_embedding",
        "target_tower_train": False,
        "sample_hidden": 512,
        "sample_dropout": 0.25,
        "lr": 2.0e-4,
        "sample_lr": 2.0e-4,
        "scale_lr": 2.0e-4,
        "weight_decay": 1.0e-5,
        "max_epochs": 80,
        "min_epochs": 8,
        "patience": 14,
        "bce_weight": 0.20,
        "sample_l2": 0.0,
        "target_tower_l2": 0.0,
        "target_init_drift_l2": 0.0,
        "batch_size": 256,
        "ablation": "scratch_mlp_fixed_trained_metar_target_embedding",
    },
    "m1_raw_trained_target_tower": {
        "target_mode": "trained_raw_gene_repr",
        "target_tower_train": True,
        "target_tower_init": "trained_label_tower",
        "sample_hidden": 512,
        "sample_dropout": 0.25,
        "lr": 2.0e-4,
        "sample_lr": 2.0e-4,
        "target_tower_lr": 3.0e-5,
        "scale_lr": 2.0e-4,
        "weight_decay": 1.0e-5,
        "max_epochs": 80,
        "min_epochs": 8,
        "patience": 14,
        "bce_weight": 0.20,
        "sample_l2": 0.0,
        "target_tower_l2": 1.0e-5,
        "target_init_drift_l2": 0.02,
        "batch_size": 256,
        "ablation": "scratch_mlp_trained_metar_raw_gene_repr_plus_target_tower",
    },
    "m2_raw_trained_target_tower_frozen": {
        "target_mode": "trained_raw_gene_repr",
        "target_tower_train": False,
        "target_tower_init": "trained_label_tower",
        "sample_hidden": 512,
        "sample_dropout": 0.25,
        "lr": 2.0e-4,
        "sample_lr": 2.0e-4,
        "scale_lr": 2.0e-4,
        "weight_decay": 1.0e-5,
        "max_epochs": 80,
        "min_epochs": 8,
        "patience": 14,
        "bce_weight": 0.20,
        "sample_l2": 0.0,
        "target_tower_l2": 0.0,
        "target_init_drift_l2": 0.0,
        "batch_size": 256,
        "ablation": "scratch_mlp_trained_metar_raw_gene_repr_frozen_target_tower",
    },
    "m3_raw_untrained_target_tower": {
        "target_mode": "untrained_raw_gene_repr",
        "target_tower_train": True,
        "target_tower_init": "untrained_label_tower",
        "sample_hidden": 512,
        "sample_dropout": 0.25,
        "lr": 2.0e-4,
        "sample_lr": 2.0e-4,
        "target_tower_lr": 3.0e-5,
        "scale_lr": 2.0e-4,
        "weight_decay": 1.0e-5,
        "max_epochs": 80,
        "min_epochs": 8,
        "patience": 14,
        "bce_weight": 0.20,
        "sample_l2": 0.0,
        "target_tower_l2": 1.0e-5,
        "target_init_drift_l2": 0.05,
        "batch_size": 256,
        "ablation": "scratch_mlp_untrained_raw_gene_repr_plus_target_tower",
    },
}


def log(message: str) -> None:
    print(f"[round230] {message}", flush=True)


def ensure_dirs() -> None:
    for path in [SOURCE, QA, CHECKPOINTS]:
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_h07_wrapper():
    os.environ.setdefault("METAR_BULK_PROJECT_ROOT", str(ROUND_ROOT / "h07_import_tmp"))
    spec = importlib.util.spec_from_file_location("round230_h07_wrapper", H07_WRAPPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import h07 wrapper from {H07_WRAPPER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_h07_data() -> tuple[object, dict, dict]:
    wrapper = import_h07_wrapper()
    cached = torch.load(H07_CACHE, map_location="cpu", weights_only=False)
    return wrapper, cached["data"], cached["protocol"]


def move_sparse(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if tensor.device == device:
        return tensor
    return tensor.coalesce().to(device) if tensor.is_sparse else tensor.to(device)


def subset_split_by_cell(split: dict, cell_line: str) -> dict:
    mask = split["df"]["cell_line"].astype(str).eq(cell_line).to_numpy()
    return {
        "df": split["df"].loc[mask].reset_index(drop=True),
        "x": split["x"][mask].astype(np.float32, copy=False),
        "y": split["y"][mask].astype(np.float32, copy=False),
    }


def subset_holdout_by_cell(bundle: dict, cell_line: str) -> dict:
    mask = bundle["df"]["cell_line"].astype(str).eq(cell_line).to_numpy()
    idx = np.flatnonzero(mask)
    return {
        "df": bundle["df"].iloc[idx].reset_index(drop=True),
        "x": bundle["x"][idx].astype(np.float32, copy=False),
        "true_indices": [bundle["true_indices"][int(i)] for i in idx],
    }


def assert_expected_counts(data: dict, splits: dict, holdouts: dict) -> None:
    observed = {
        "n_genes_full_universe": len(data["gene_list"]),
        "n_train_label_genes": len(data["train_label_genes"]),
        "pc3_inner_train": len(splits["inner_train"]["x"]),
        "pc3_val": len(splits["val"]["x"]),
        "pc3_test": len(splits["test"]["x"]),
        "pc3_holdout_val": len(holdouts["holdout_val"]["x"]),
        "pc3_holdout_test": len(holdouts["holdout_test"]["x"]),
    }
    bad = {k: (observed[k], v) for k, v in EXPECTED_COUNTS.items() if observed[k] != v}
    if bad:
        raise RuntimeError(f"Unexpected h07/PC3 counts: {bad}")


def scatter_local_targets_to_full(y_local: torch.Tensor, train_idx: torch.Tensor, n_genes: int) -> torch.Tensor:
    y_full = torch.zeros((y_local.shape[0], n_genes), dtype=y_local.dtype, device=y_local.device)
    y_full[:, train_idx] = y_local
    return y_full


def build_h07_model(wrapper: object, data: dict, device: torch.device, load_checkpoint: bool) -> tuple[nn.Module, dict]:
    gnn = wrapper.bulk_base.choose_gnn("baseline_clean_v6", len(data["gene_list"])).to(device)
    model = wrapper.SharedSampleIndependentUnseenLabelMetaTwoTowerGNN(
        gnn=gnn,
        n_expr_genes=len(data["expr_cols"]),
        hidden=512,
        out_dim=256,
        shared_sample_variant="residual",
        shared_label_variant="residual",
        unseen_label_variant="linear",
    ).to(device)
    audit: dict[str, Any] = {"checkpoint_loaded": False}
    if load_checkpoint:
        state = torch.load(H07_CKPT, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(state["model_state_dict"], strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Unexpected h07 checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        audit = {
            "checkpoint_loaded": True,
            "h07_checkpoint": str(H07_CKPT),
            "h07_checkpoint_sha256": sha256_file(H07_CKPT),
            "checkpoint_epoch": int(state["epoch"]),
            "checkpoint_variant": str(state["variant"]),
            "checkpoint_head": str(state["head"]),
            "checkpoint_unseen_score": float(state["unseen_score"]),
            "checkpoint_val_holdout_top5pct": float(state["val_holdout_top5pct"]),
        }
    return model, audit


@torch.no_grad()
def extract_target_bundle(wrapper: object, data: dict, device: torch.device, trained: bool) -> dict[str, Any]:
    set_seed(SEED if trained else 20260518)
    model, audit = build_h07_model(wrapper, data, device, load_checkpoint=trained)
    model.eval()
    adj = move_sparse(data["adj_norm"], device)
    gene_repr = model.gnn(adj)
    target = F.normalize(model.label_forward(gene_repr, head="unseen"), dim=1, eps=1e-8)
    label_state = {k: v.detach().cpu().clone() for k, v in model.label_unseen_private.state_dict().items()}
    audit.update(
        {
            "target_source": "trained_h07_ppi" if trained else "untrained_h07_ppi",
            "raw_gene_repr_shape": [int(gene_repr.shape[0]), int(gene_repr.shape[1])],
            "target_shape": [int(target.shape[0]), int(target.shape[1])],
            "target_norm_min": float(target.norm(dim=1).min().item()),
            "target_norm_max": float(target.norm(dim=1).max().item()),
        }
    )
    return {
        "raw_gene_repr": gene_repr.detach().cpu().float().numpy().astype(np.float32),
        "target_embedding": target.detach().cpu().float().numpy().astype(np.float32),
        "label_state": label_state,
        "audit": audit,
    }


class SampleEncoderResidual(nn.Module):
    def __init__(self, in_dim: int = 978, hidden: int = 512, out_dim: int = 256, dropout: float = 0.25) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.fc3 = nn.Linear(hidden, out_dim)
        self.skip = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.ln1(self.fc1(x)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.gelu(self.ln2(self.fc2(h)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.fc3(h) + self.skip(x)


class LabelEncoderLinear(nn.Module):
    def __init__(self, in_dim: int = 128, out_dim: int = 256) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ScratchMLPTargetModel(nn.Module):
    def __init__(self, target_input_full: np.ndarray, train_idx: np.ndarray, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = dict(cfg)
        self.target_mode = str(cfg["target_mode"])
        self.register_buffer("target_full_anchor", torch.tensor(target_input_full, dtype=torch.float32))
        self.register_buffer("target_init_reference", torch.empty(0, dtype=torch.float32))
        self.register_buffer("train_idx", torch.tensor(train_idx, dtype=torch.long))
        out_dim = 256
        self.sample_encoder = SampleEncoderResidual(
            in_dim=978,
            hidden=int(cfg.get("sample_hidden", 512)),
            out_dim=out_dim,
            dropout=float(cfg.get("sample_dropout", 0.25)),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0), dtype=torch.float32))
        if self.target_mode == "trained_target_embedding":
            self.target_tower = nn.Identity()
        else:
            self.target_tower = LabelEncoderLinear(in_dim=int(target_input_full.shape[1]), out_dim=out_dim)

    def initialize_target_tower(self, state: dict[str, torch.Tensor] | None, fixed_reference: np.ndarray | None = None) -> dict[str, Any]:
        if isinstance(self.target_tower, nn.Identity):
            return {"loaded": False, "reason": "fixed_target_embedding"}
        if state is None:
            raise RuntimeError("Target tower variant requires a label tower state")
        self.target_tower.load_state_dict(state, strict=True)
        with torch.no_grad():
            init = F.normalize(self.target_tower(self.target_full_anchor), dim=1, eps=1e-8).detach().clone()
            self.target_init_reference = init
            audit = {"loaded": True, "init_reference_shape": list(init.shape)}
            if fixed_reference is not None:
                ref = torch.tensor(fixed_reference, dtype=torch.float32)
                audit["equivalence_max_abs_to_label_forward"] = float((init.cpu() - ref).abs().max().item())
                audit["equivalence_mean_cos_to_label_forward"] = float((init.cpu() * ref).sum(dim=1).mean().item())
        return audit

    def target_full(self) -> torch.Tensor:
        if isinstance(self.target_tower, nn.Identity):
            target = self.target_full_anchor
        else:
            target = self.target_tower(self.target_full_anchor)
        return F.normalize(target, dim=1, eps=1e-8)

    def query(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.sample_encoder(x), dim=1, eps=1e-8)

    def logits_full(self, x: torch.Tensor) -> torch.Tensor:
        return (self.query(x) @ self.target_full().T) * torch.exp(torch.clamp(self.logit_scale, -2.0, 4.0))

    def logits_seen(self, x: torch.Tensor) -> torch.Tensor:
        return (self.query(x) @ self.target_full().index_select(0, self.train_idx).T) * torch.exp(torch.clamp(self.logit_scale, -2.0, 4.0))

    def regularization(self) -> torch.Tensor:
        reg = torch.zeros((), dtype=self.logit_scale.dtype, device=self.logit_scale.device)
        target_l2 = float(self.cfg.get("target_tower_l2", 0.0))
        if target_l2 > 0 and not isinstance(self.target_tower, nn.Identity):
            for param in self.target_tower.parameters():
                reg = reg + target_l2 * torch.square(param).sum()
        drift = float(self.cfg.get("target_init_drift_l2", 0.0))
        if drift > 0 and self.target_init_reference.numel() > 0:
            target = self.target_full()
            ref = self.target_init_reference.to(target.device)
            reg = reg + drift * (1.0 - (target * ref).sum(dim=1)).mean()
        sample_l2 = float(self.cfg.get("sample_l2", 0.0))
        if sample_l2 > 0:
            for param in self.sample_encoder.parameters():
                reg = reg + sample_l2 * torch.square(param).sum()
        return reg


def bag_softmax_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    masked_pos = logits.masked_fill(targets <= 0, float("-inf"))
    log_pos = torch.logsumexp(masked_pos, dim=1)
    log_all = torch.logsumexp(logits, dim=1)
    valid = targets.sum(dim=1) > 0
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return -(log_pos[valid] - log_all[valid]).mean()


def drug_normalized_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pos_mask = targets > 0
    neg_mask = ~pos_mask
    raw = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pos = (raw * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp_min(1)
    neg = (raw * neg_mask).sum(dim=1) / neg_mask.sum(dim=1).clamp_min(1)
    return 0.5 * (pos + neg).mean()


def compute_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    order = np.argsort(-scores, axis=1)
    hits: dict[str, float] = {}
    for k in [1, 5, 10, 20, 30, 50]:
        topk = order[:, :k]
        hits[f"Hit@{k}"] = float((np.take_along_axis(labels, topk, axis=1).sum(axis=1) > 0).mean())
    valid_auc = (labels.sum(axis=1) > 0) & (labels.sum(axis=1) < labels.shape[1])
    auroc = float(roc_auc_score(labels[valid_auc], scores[valid_auc], average="samples")) if bool(valid_auc.any()) else 0.5
    return {
        **hits,
        "AUROC": auroc,
        "NDCG@10": float(ndcg_score(labels, scores, k=10)),
        "MeanAP": float(average_precision_score(labels, scores, average="samples")),
        "n_profiles": int(labels.shape[0]),
    }


@torch.no_grad()
def predict_seen(model: ScratchMLPTargetModel, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(x), batch_size):
        xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
        out.append(model.logits_seen(xb).detach().cpu())
    return torch.cat(out, dim=0).numpy()


@torch.no_grad()
def evaluate_holdout_top5(
    model: ScratchMLPTargetModel,
    bundle: dict,
    device: torch.device,
    batch_size: int,
    exclude_seen: bool = False,
    return_rows: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    model.eval()
    n_genes = int(model.target_full_anchor.shape[0])
    k = max(1, int(math.floor(n_genes * 0.05)))
    seen_mask = None
    if exclude_seen:
        seen_mask = torch.zeros(n_genes, dtype=torch.bool, device=device)
        seen_mask[model.train_idx.to(device)] = True
    successes = 0
    rows = []
    for start in range(0, len(bundle["x"]), batch_size):
        xb = torch.tensor(bundle["x"][start : start + batch_size], dtype=torch.float32, device=device)
        logits = model.logits_full(xb)
        if seen_mask is not None:
            logits[:, seen_mask] = float("-inf")
        topk = torch.topk(logits, k=k, dim=1).indices.detach().cpu().numpy()
        logits_cpu = logits.detach().cpu().numpy()
        for offset, (row_topk, true_idx) in enumerate(zip(topk, bundle["true_indices"][start : start + batch_size])):
            true_idx = [int(t) for t in true_idx]
            success = any(t in set(row_topk.tolist()) for t in true_idx)
            successes += int(success)
            if return_rows:
                valid_true = [t for t in true_idx if np.isfinite(logits_cpu[offset, t])]
                ranks = [int((logits_cpu[offset] > logits_cpu[offset, t]).sum()) + 1 for t in valid_true]
                rows.append(
                    {
                        "row_index": int(start + offset),
                        "drug": str(bundle["df"].iloc[start + offset]["drug"]),
                        "cell_line": str(bundle["df"].iloc[start + offset]["cell_line"]),
                        "success_top5pct": bool(success),
                        "best_rank": int(min(ranks)) if ranks else -1,
                        "n_true_targets": int(len(true_idx)),
                    }
                )
    return (
        {
            "top5pct_success": successes / len(bundle["x"]) if len(bundle["x"]) else 0.0,
            "n_profiles": int(len(bundle["x"])),
            "candidate_space": int(n_genes),
            "exclude_seen_labels": bool(exclude_seen),
        },
        pd.DataFrame(rows) if return_rows else None,
    )


def better_candidate(candidate: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    if incumbent is None:
        return True
    for key in ["holdout_top5pct", "holdout_top5pct_remove_seen", "epoch_preference"]:
        if candidate[key] > incumbent[key] + 1e-12:
            return True
        if candidate[key] < incumbent[key] - 1e-12:
            return False
    return False


def evaluate_variant(
    run_id: str,
    model: ScratchMLPTargetModel,
    splits: dict,
    holdouts: dict,
    device: torch.device,
    best_epoch: int,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    rows: list[dict[str, Any]] = []
    for split_name in ["val", "test"]:
        scores = predict_seen(model, splits[split_name]["x"], device, batch_size=256)
        metrics = compute_metrics(scores, splits[split_name]["y"])
        rows.append({"variant": run_id, "split": split_name, "evaluation_family": "seen_668", "evaluation_unit": "profile", **metrics, "best_epoch": best_epoch})
    per_rows = []
    for split_name in ["holdout_val", "holdout_test"]:
        metrics, per = evaluate_holdout_top5(model, holdouts[split_name], device, batch_size=256, exclude_seen=False, return_rows=(split_name == "holdout_test"))
        metrics_rm, _ = evaluate_holdout_top5(model, holdouts[split_name], device, batch_size=256, exclude_seen=True, return_rows=False)
        rows.append({"variant": run_id, "split": split_name, "evaluation_family": "holdout_fullspace_top5", "evaluation_unit": "profile", **metrics, "best_epoch": best_epoch})
        rows.append({"variant": run_id, "split": split_name, "evaluation_family": "holdout_fullspace_top5_remove_seen", "evaluation_unit": "profile", **metrics_rm, "best_epoch": best_epoch})
        if per is not None:
            per.insert(0, "variant", run_id)
            per.insert(1, "split", split_name)
            per_rows.append(per)
    return pd.DataFrame(rows), pd.concat(per_rows, ignore_index=True) if per_rows else None


def train_variant(
    variant: str,
    cfg: dict[str, Any],
    model: ScratchMLPTargetModel,
    splits: dict,
    holdouts: dict,
    train_idx: np.ndarray,
    device: torch.device,
    smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    model = model.to(device)
    for p in model.target_tower.parameters():
        p.requires_grad_(bool(cfg.get("target_tower_train", False)))
    sample_params = list(model.sample_encoder.parameters())
    scale_params = [model.logit_scale]
    groups = [
        {"params": sample_params, "lr": float(cfg.get("sample_lr", cfg["lr"]))},
        {"params": scale_params, "lr": float(cfg.get("scale_lr", cfg["lr"]))},
    ]
    target_params = [p for p in model.target_tower.parameters() if p.requires_grad]
    if target_params:
        groups.append({"params": target_params, "lr": float(cfg.get("target_tower_lr", cfg["lr"]))})
    trainable_params = [p for group in groups for p in group["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=float(cfg["weight_decay"]))
    tensors = {name: {"x": torch.tensor(item["x"], dtype=torch.float32, device=device), "y": torch.tensor(item["y"], dtype=torch.float32, device=device)} for name, item in splits.items()}
    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    rng = np.random.default_rng(SEED)
    n_train = tensors["inner_train"]["x"].shape[0]
    max_epochs = 1 if smoke else int(cfg["max_epochs"])
    min_epochs = 1 if smoke else int(cfg["min_epochs"])
    patience = int(cfg["patience"])
    batch_size = int(cfg["batch_size"])
    history: list[dict[str, Any]] = []
    best = None
    best_state = None
    best_epoch = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(n_train)
        losses = []
        for start in range(0, n_train, batch_size):
            idx = torch.tensor(order[start : start + batch_size], dtype=torch.long, device=device)
            xb = tensors["inner_train"]["x"][idx]
            yb_local = tensors["inner_train"]["y"][idx]
            yb_full = scatter_local_targets_to_full(yb_local, train_idx_t, int(model.target_full_anchor.shape[0]))
            logits = model.logits_full(xb)
            loss = bag_softmax_loss(logits, yb_full) + float(cfg["bce_weight"]) * drug_normalized_bce_loss(logits, yb_full) + model.regularization()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 2.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        seen_scores = predict_seen(model, splits["val"]["x"], device, batch_size=256)
        seen = compute_metrics(seen_scores, splits["val"]["y"])
        holdout, _ = evaluate_holdout_top5(model, holdouts["holdout_val"], device, batch_size=256, exclude_seen=False)
        holdout_rm, _ = evaluate_holdout_top5(model, holdouts["holdout_val"], device, batch_size=256, exclude_seen=True)
        rec = {
            "variant": variant,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "seen_Hit@10": float(seen["Hit@10"]),
            "seen_MeanAP": float(seen["MeanAP"]),
            "seen_NDCG@10": float(seen["NDCG@10"]),
            "holdout_top5pct": float(holdout["top5pct_success"]),
            "holdout_top5pct_remove_seen": float(holdout_rm["top5pct_success"]),
            "epoch_preference": -float(epoch),
            "seed": SEED,
            "target_mode": str(cfg["target_mode"]),
            "ablation": str(cfg["ablation"]),
            "sample_hidden": int(cfg["sample_hidden"]),
            "sample_dropout": float(cfg["sample_dropout"]),
            "target_tower_train": bool(cfg.get("target_tower_train", False)),
        }
        history.append(rec)
        if better_candidate(rec, best):
            best = rec
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0:
            log(f"{variant} epoch={epoch} loss={rec['train_loss']:.4f} val_hit10={rec['seen_Hit@10']:.4f} holdout_top5={rec['holdout_top5pct']:.4f}")
        if epoch >= min_epochs and best is not None and epoch - best_epoch >= patience:
            log(f"{variant} early stop epoch={epoch}; best_epoch={best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    metrics, per = evaluate_variant(variant, model, splits, holdouts, device, best_epoch)
    ckpt_dir = CHECKPOINTS / variant
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "best_epoch": best_epoch, "cfg": cfg}, ckpt_dir / "selected_checkpoint.pth")
    return metrics, pd.DataFrame(history), per


def build_model_for_variant(variant: str, cfg: dict[str, Any], bundles: dict[str, dict[str, Any]], train_idx: np.ndarray) -> tuple[ScratchMLPTargetModel, dict[str, Any]]:
    if cfg["target_mode"] == "trained_target_embedding":
        target_input = bundles["trained"]["target_embedding"]
        fixed_ref = None
        label_state = None
    elif cfg["target_mode"] == "trained_raw_gene_repr":
        target_input = bundles["trained"]["raw_gene_repr"]
        fixed_ref = bundles["trained"]["target_embedding"]
        label_state = bundles["trained"]["label_state"]
    elif cfg["target_mode"] == "untrained_raw_gene_repr":
        target_input = bundles["untrained"]["raw_gene_repr"]
        fixed_ref = bundles["untrained"]["target_embedding"]
        label_state = bundles["untrained"]["label_state"]
    else:
        raise ValueError(f"Unknown target mode: {cfg['target_mode']}")
    model = ScratchMLPTargetModel(target_input, train_idx, cfg)
    init = model.initialize_target_tower(label_state, fixed_reference=fixed_ref)
    audit = {"variant": variant, "target_mode": cfg["target_mode"], "target_tower_init": init, "target_shape": list(target_input.shape)}
    return model, audit


def write_report(metrics: pd.DataFrame, history: pd.DataFrame, audits: list[dict[str, Any]]) -> None:
    shown = metrics[metrics["split"].isin(["test", "holdout_test"])].copy()
    cols = ["variant", "target_mode", "split", "evaluation_family", "Hit@10", "MeanAP", "AUROC", "NDCG@10", "top5pct_success", "exclude_seen_labels", "best_epoch"]
    table = shown[[c for c in cols if c in shown.columns]].to_string(index=False)
    val = metrics[metrics["split"].eq("holdout_val")][[c for c in cols if c in metrics.columns]].to_string(index=False)
    text = f"""# Round230 scratch MLP + MeTar target tower report

## Scope

This round does not use the A01 sample encoder. The sample side is a newly initialized residual MLP trained on PC3 h07 data. Target-side variants use MeTar h07 PPI target embeddings or raw PPI gene representations plus a MeTar-style target tower.

## Test results

```text
{table}
```

## Holdout-validation results

```text
{val}
```

## Target initialization audit

```json
{json.dumps(audits, indent=2)}
```
"""
    (ROUND_ROOT / "round230_result_report_zh.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=str, default=",".join(VARIANTS.keys()))
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")
    wrapper, data, protocol = load_h07_data()
    splits = {name: subset_split_by_cell(data[name], CELL_LINE) for name in ["inner_train", "val", "test"]}
    holdouts = {name: subset_holdout_by_cell(data[name], CELL_LINE) for name in ["holdout_val", "holdout_test"]}
    assert_expected_counts(data, splits, holdouts)
    train_idx = data["train_gene_global_idx"].detach().cpu().numpy().astype(np.int64)

    bundles = {
        "trained": extract_target_bundle(wrapper, data, device, trained=True),
        "untrained": extract_target_bundle(wrapper, data, device, trained=False),
    }
    metrics_all: list[pd.DataFrame] = []
    history_all: list[pd.DataFrame] = []
    per_all: list[pd.DataFrame] = []
    audits = []
    for variant in [x for x in args.runs.split(",") if x]:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        set_seed(SEED)
        cfg = dict(VARIANTS[variant])
        model, audit = build_model_for_variant(variant, cfg, bundles, train_idx)
        audits.append(audit)
        metrics, hist, per = train_variant(variant, cfg, model, splits, holdouts, train_idx, device, smoke=bool(args.smoke))
        for df in [metrics, hist]:
            df["target_mode"] = cfg["target_mode"]
            df["ablation"] = cfg["ablation"]
        metrics_all.append(metrics)
        history_all.append(hist)
        if per is not None:
            per["target_mode"] = cfg["target_mode"]
            per_all.append(per)

    metrics_df = pd.concat(metrics_all, ignore_index=True)
    history_df = pd.concat(history_all, ignore_index=True)
    metrics_df.to_csv(SOURCE / "metric_summary.tsv", sep="\t", index=False)
    history_df.to_csv(SOURCE / "training_history.tsv", sep="\t", index=False)
    if per_all:
        pd.concat(per_all, ignore_index=True).to_csv(SOURCE / "per_profile_holdout_metrics.tsv.gz", sep="\t", index=False, compression="gzip")
    (SOURCE / "target_bundle_audit.json").write_text(json.dumps({k: v["audit"] for k, v in bundles.items()}, indent=2), encoding="utf-8")
    (SOURCE / "target_mode_audit.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    method = {
        "round": "round230_scratch_mlp_metar_target_tower_pc3",
        "cell_line": CELL_LINE,
        "seed": SEED,
        "counts": EXPECTED_COUNTS,
        "variants": VARIANTS,
        "h07_cache": str(H07_CACHE),
        "h07_checkpoint": str(H07_CKPT),
        "protocol_keys": sorted(protocol.keys()) if isinstance(protocol, dict) else [],
    }
    (ROUND_ROOT / "method_lock.json").write_text(json.dumps(method, indent=2), encoding="utf-8")
    (QA / "leakage_and_alignment_audit.md").write_text(
        f"""# Round230 audit

- No A01 sample encoder checkpoint is loaded.
- Sample encoder is newly initialized for every run.
- Trained MeTar h07 PPI target artifacts are used only in variants whose target mode explicitly starts with `trained`.
- Training labels are restricted to the original 668 seen-label genes scattered into the 16,583-gene universe.
- Holdout target labels are used only for validation/test evaluation.
- PC3 counts match h07 expected counts: `{EXPECTED_COUNTS}`.
""",
        encoding="utf-8",
    )
    write_report(metrics_df, history_df, audits)
    log(f"wrote {SOURCE / 'metric_summary.tsv'}")


if __name__ == "__main__":
    main()
