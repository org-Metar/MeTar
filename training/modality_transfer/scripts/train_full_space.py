#!/usr/bin/env python3
"""Train the full-candidate target-ranking model with cell-disjoint validation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn


ROUND_ROOT = Path(__file__).resolve().parents[1]
REMOTE_WRITE_ROOT = Path("/home/lwn/claude/manuscript/write")
ROUND230_ROOT = REMOTE_WRITE_ROOT / "claude_round230_scratch_mlp_metar_target_tower_pc3_20260518"
ROUND230_SCRIPT = ROUND230_ROOT / "scripts" / "run_round230_scratch_mlp_metar_target_tower_pc3.py"
ROUND246_CKPT = (
    REMOTE_WRITE_ROOT
    / "claude_round246_joint_seen_target_train_pc3_a375_mcf7_20260519"
    / "checkpoints"
    / "r246_joint_lm0p5_sc0p15_mn2p5"
    / "whole_model_checkpoint.pth"
)
ROUND252_PROTOCOL_AUDIT = (
    REMOTE_WRITE_ROOT
    / "claude_round252_epiqgate_pseudoval_sweep_20260520"
    / "source_data"
    / "cell_splits.json"
)
VARIANTS: dict[str, dict[str, Any]] = {
    "e8_enc105e6": {
        "encoder_train": "all",
        "encoder_lr": 1.05e-4,
        "gnn_lr": 4.0e-5,
        "label_lr": 5.0e-5,
        "scale_lr": 1.0e-4,
        "weight_decay": 3.0e-5,
        "bce_weight": 0.0,
        "target_drift_l2": 0.003,
        "encoder_l2": 0.0,
        "query_adapter_rank": 0,
        "max_epochs": 35,
        "min_epochs": 8,
        "patience": 12,
        "batch_size": 192,
    },
    "e10_enc102e6": {
        "encoder_train": "all",
        "encoder_lr": 1.02e-4,
        "gnn_lr": 4.0e-5,
        "label_lr": 5.0e-5,
        "scale_lr": 1.0e-4,
        "weight_decay": 3.0e-5,
        "bce_weight": 0.0,
        "target_drift_l2": 0.003,
        "encoder_l2": 0.0,
        "query_adapter_rank": 0,
        "max_epochs": 35,
        "min_epochs": 8,
        "patience": 12,
        "batch_size": 192,
    },
    "e11_enc103e6": {
        "encoder_train": "all",
        "encoder_lr": 1.03e-4,
        "gnn_lr": 4.0e-5,
        "label_lr": 5.0e-5,
        "scale_lr": 1.0e-4,
        "weight_decay": 3.0e-5,
        "bce_weight": 0.0,
        "target_drift_l2": 0.003,
        "encoder_l2": 0.0,
        "query_adapter_rank": 0,
        "max_epochs": 35,
        "min_epochs": 8,
        "patience": 12,
        "batch_size": 192,
    },
    "e12_enc104e6": {
        "encoder_train": "all",
        "encoder_lr": 1.04e-4,
        "gnn_lr": 4.0e-5,
        "label_lr": 5.0e-5,
        "scale_lr": 1.0e-4,
        "weight_decay": 3.0e-5,
        "bce_weight": 0.0,
        "target_drift_l2": 0.003,
        "encoder_l2": 0.0,
        "query_adapter_rank": 0,
        "max_epochs": 35,
        "min_epochs": 8,
        "patience": 12,
        "batch_size": 192,
    },
    "e13_bce001": {
        "encoder_train": "all",
        "encoder_lr": 1.0e-4,
        "gnn_lr": 4.0e-5,
        "label_lr": 5.0e-5,
        "scale_lr": 1.0e-4,
        "weight_decay": 3.0e-5,
        "bce_weight": 0.01,
        "target_drift_l2": 0.003,
        "encoder_l2": 0.0,
        "query_adapter_rank": 0,
        "max_epochs": 35,
        "min_epochs": 8,
        "patience": 12,
        "batch_size": 192,
    },
    "e14_adapter16": {
        "encoder_train": "all",
        "encoder_lr": 1.0e-4,
        "gnn_lr": 4.0e-5,
        "label_lr": 5.0e-5,
        "scale_lr": 1.0e-4,
        "weight_decay": 3.0e-5,
        "bce_weight": 0.0,
        "target_drift_l2": 0.003,
        "encoder_l2": 0.0,
        "query_adapter_rank": 16,
        "query_adapter_scale": 0.5,
        "query_adapter_l2": 1.0e-6,
        "max_epochs": 35,
        "min_epochs": 8,
        "patience": 12,
        "batch_size": 192,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_round230_module():
    spec = importlib.util.spec_from_file_location("round230_base", ROUND230_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import Round230 base script from {ROUND230_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_round252_protocol_audit() -> dict[str, Any]:
    if not ROUND252_PROTOCOL_AUDIT.exists():
        raise RuntimeError(f"Missing Round252 protocol audit: {ROUND252_PROTOCOL_AUDIT}")
    audit = json.loads(ROUND252_PROTOCOL_AUDIT.read_text(encoding="utf-8"))
    required = ["base_train_cells", "pseudo_unseen_val_cells", "official_unseen_cells"]
    missing = [key for key in required if key not in audit]
    if missing:
        raise RuntimeError(f"Round252 protocol audit missing keys: {missing}")
    leakage = audit.get("leakage_assertions", {})
    bad = {
        key: value
        for key, value in leakage.items()
        if key.endswith("_overlap") or key.startswith(("official_unseen_intersects", "pseudo_unseen_intersects"))
        if value
    }
    if bad:
        raise RuntimeError(f"Round252 protocol audit has non-empty leakage assertions: {bad}")
    return audit


def subset_split_by_cells(split: dict[str, Any], cells: list[str], split_label: str) -> dict[str, Any]:
    cell_set = set(map(str, cells))
    df = split["df"].copy()
    mask = df["cell_line"].astype(str).isin(cell_set).to_numpy()
    out_df = df.loc[mask].reset_index(drop=True)
    out_df["round259_cell_split"] = split_label
    return {
        "df": out_df,
        "x": split["x"][mask].astype(np.float32, copy=False),
        "y": split["y"][mask].astype(np.float32, copy=False),
    }


def subset_holdout_by_cells(bundle: dict[str, Any], cells: list[str], split_label: str) -> dict[str, Any]:
    cell_set = set(map(str, cells))
    df = bundle["df"].copy()
    mask = df["cell_line"].astype(str).isin(cell_set).to_numpy()
    idx = np.flatnonzero(mask)
    out_df = df.iloc[idx].reset_index(drop=True)
    out_df["round259_cell_split"] = split_label
    return {
        "df": out_df,
        "x": bundle["x"][idx].astype(np.float32, copy=False),
        "true_indices": [bundle["true_indices"][int(i)] for i in idx],
    }


def count_bundle(bundle: dict[str, Any]) -> dict[str, int]:
    df = bundle["df"]
    return {
        "n_profiles": int(len(bundle["x"])),
        "n_cells": int(df["cell_line"].astype(str).nunique()) if len(df) else 0,
        "n_drugs": int(df["drug"].astype(str).nunique()) if len(df) and "drug" in df else 0,
    }


def build_round259_splits(module: Any, data: dict[str, Any], protocol_audit: dict[str, Any]) -> tuple[dict[str, dict], dict[str, dict], dict[str, Any]]:
    cell_sets = {
        "base_train_cells": list(map(str, protocol_audit["base_train_cells"])),
        "pseudo_unseen_val_cells": list(map(str, protocol_audit["pseudo_unseen_val_cells"])),
        "official_unseen_cells": list(map(str, protocol_audit["official_unseen_cells"])),
    }
    overlap_audit = {
        "official_unseen_intersects_train": sorted(set(cell_sets["official_unseen_cells"]).intersection(cell_sets["base_train_cells"])),
        "pseudo_unseen_intersects_train": sorted(set(cell_sets["pseudo_unseen_val_cells"]).intersection(cell_sets["base_train_cells"])),
        "official_unseen_intersects_pseudo": sorted(set(cell_sets["official_unseen_cells"]).intersection(cell_sets["pseudo_unseen_val_cells"])),
    }
    bad_overlap = {k: v for k, v in overlap_audit.items() if v}
    if bad_overlap:
        raise RuntimeError(f"Round259 cell split leakage: {bad_overlap}")

    splits = {
        "inner_train": subset_split_by_cells(data["inner_train"], cell_sets["base_train_cells"], "base_train"),
        "val": subset_split_by_cells(data["val"], cell_sets["pseudo_unseen_val_cells"], "pseudo_unseen_selection"),
        "test": subset_split_by_cells(data["test"], cell_sets["official_unseen_cells"], "official_unseen_final"),
        "pc3_test": module.subset_split_by_cell(data["test"], "PC3"),
    }
    holdouts = {
        "holdout_val": subset_holdout_by_cells(data["holdout_val"], cell_sets["pseudo_unseen_val_cells"], "pseudo_unseen_selection"),
        "holdout_test": subset_holdout_by_cells(data["holdout_test"], cell_sets["official_unseen_cells"], "official_unseen_final"),
        "pc3_holdout_test": module.subset_holdout_by_cell(data["holdout_test"], "PC3"),
    }
    counts = {
        "splits": {name: count_bundle(item) for name, item in splits.items()},
        "holdouts": {name: count_bundle(item) for name, item in holdouts.items()},
        "cell_sets": {key: {"n_cells": len(value), "cells": value} for key, value in cell_sets.items()},
        "leakage": overlap_audit,
        "n_genes_full_universe": int(len(data["gene_list"])),
        "n_train_label_genes": int(len(data["train_label_genes"])),
    }
    if counts["n_genes_full_universe"] != 16583 or counts["n_train_label_genes"] != 668:
        raise RuntimeError(f"Unexpected full-space dimensions: {counts}")
    if not len(splits["inner_train"]["x"]) or not len(splits["val"]["x"]) or not len(splits["test"]["x"]):
        raise RuntimeError(f"Empty Round259 split detected: {counts}")
    if not len(holdouts["holdout_val"]["x"]) or not len(holdouts["holdout_test"]["x"]):
        raise RuntimeError(f"Empty Round259 holdout split detected: {counts}")
    return splits, holdouts, counts


def load_round246_query_state() -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    state = torch.load(ROUND246_CKPT, map_location="cpu", weights_only=False)
    model_state = state.get("readout_state_dict", state.get("model_state_dict", state.get("state_dict", state)))
    query_state = {
        k: v.detach().clone().float()
        for k, v in model_state.items()
        if k.startswith("query_encoder.") or k.startswith("query_norm.")
    }
    required = {
        "query_encoder.0.weight",
        "query_encoder.0.bias",
        "query_encoder.1.weight",
        "query_encoder.1.bias",
        "query_encoder.4.weight",
        "query_encoder.4.bias",
        "query_encoder.5.weight",
        "query_encoder.5.bias",
        "query_norm.weight",
        "query_norm.bias",
    }
    missing = sorted(required.difference(query_state))
    if missing:
        raise RuntimeError(f"Round246 query encoder checkpoint missing keys: {missing}")
    audit = {
        "round246_checkpoint": str(ROUND246_CKPT),
        "round246_checkpoint_sha256": sha256_file(ROUND246_CKPT),
        "checkpoint_top_keys": sorted(state.keys()) if isinstance(state, dict) else [],
        "loaded_query_keys": sorted(query_state),
        "state_shapes": {k: list(v.shape) for k, v in query_state.items()},
    }
    return query_state, audit


class Round246QueryEncoder(nn.Module):
    def __init__(self, query_state: dict[str, torch.Tensor], cfg: dict[str, Any]) -> None:
        super().__init__()
        hidden = int(query_state["query_encoder.0.weight"].shape[0])
        dim = int(query_state["query_encoder.4.weight"].shape[0])
        self.query_adapter_rank = int(cfg.get("query_adapter_rank", 0))
        self.query_adapter_scale = float(cfg.get("query_adapter_scale", 1.0))
        self._query_adapter_l2 = float(cfg.get("query_adapter_l2", 0.0))
        self.query_encoder = nn.Sequential(
            nn.Linear(978, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(hidden, dim),
            nn.LayerNorm(dim),
        )
        self.query_norm = nn.LayerNorm(dim)
        if self.query_adapter_rank > 0:
            self.query_adapter_down = nn.Linear(dim, self.query_adapter_rank, bias=False)
            self.query_adapter_up = nn.Linear(self.query_adapter_rank, dim, bias=False)
            nn.init.xavier_uniform_(self.query_adapter_down.weight)
            nn.init.zeros_(self.query_adapter_up.weight)
        else:
            self.query_adapter_down = None
            self.query_adapter_up = None
        missing, unexpected = self.load_state_dict(query_state, strict=False)
        allowed_missing: set[str] = set()
        if self.query_adapter_rank > 0:
            allowed_missing.update({"query_adapter_down.weight", "query_adapter_up.weight"})
        bad_missing = [x for x in missing if x not in allowed_missing]
        if bad_missing or unexpected:
            raise RuntimeError(f"Round246 query load mismatch: missing={bad_missing}, unexpected={unexpected}")
        self.reference = {name: param.detach().clone() for name, param in self.named_parameters()}
        train_mode = str(cfg.get("encoder_train", "all"))
        for param in self.parameters():
            param.requires_grad_(False)
        if train_mode == "all":
            for param in self.parameters():
                param.requires_grad_(True)
        elif train_mode == "last":
            for name, param in self.named_parameters():
                if name.startswith("query_encoder.4.") or name.startswith("query_encoder.5.") or name.startswith("query_norm."):
                    param.requires_grad_(True)
        elif train_mode == "none":
            pass
        else:
            raise ValueError(f"Unknown encoder_train mode: {train_mode}")
        if self.query_adapter_rank > 0:
            for name, param in self.named_parameters():
                if name.startswith("query_adapter_"):
                    param.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.query_encoder[3].eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query_norm(self.query_encoder(x))
        if self.query_adapter_down is not None and self.query_adapter_up is not None:
            q = q + self.query_adapter_scale * self.query_adapter_up(F.gelu(self.query_adapter_down(q)))
            q = self.query_norm(q)
        return q

    def drift_loss(self, weight: float) -> torch.Tensor:
        params = list(self.parameters())
        if not params:
            raise RuntimeError("Round246QueryEncoder has no parameters")
        reg = params[0].sum() * 0.0
        for name, param in self.named_parameters():
            if param.requires_grad and weight > 0 and not name.startswith("query_adapter_"):
                reg = reg + weight * torch.square(param - self.reference[name].to(param.device)).sum()
            if param.requires_grad and name.startswith("query_adapter_"):
                reg = reg + float(self.query_adapter_l2()) * torch.square(param).sum()
        return reg

    def query_adapter_l2(self) -> float:
        return float(getattr(self, "_query_adapter_l2", 0.0))

    def set_query_adapter_l2(self, value: float) -> None:
        self._query_adapter_l2 = float(value)

    def base_trainable_parameters(self) -> list[nn.Parameter]:
        return [
            param
            for name, param in self.named_parameters()
            if param.requires_grad and not name.startswith("query_adapter_")
        ]

    def adapter_trainable_parameters(self) -> list[nn.Parameter]:
        return [
            param
            for name, param in self.named_parameters()
            if param.requires_grad and name.startswith("query_adapter_")
        ]


class A01EncoderTrainablePPIModel(nn.Module):
    def __init__(
        self,
        h07_model: nn.Module,
        a01_encoder: Round246QueryEncoder,
        adj_norm: torch.Tensor,
        train_idx: np.ndarray,
        cfg: dict[str, Any],
        n_genes: int,
    ) -> None:
        super().__init__()
        self.h07_model = h07_model
        self.a01_encoder = a01_encoder
        self.cfg = dict(cfg)
        self.register_buffer("adj_norm", adj_norm.coalesce() if adj_norm.is_sparse else adj_norm)
        self.register_buffer("train_idx", torch.tensor(train_idx, dtype=torch.long))
        self.register_buffer("target_full_anchor", torch.empty(n_genes, 1))
        with torch.no_grad():
            target0 = self.target_full().detach().cpu()
            query_weight0 = torch.cat([p.detach().flatten().cpu() for p in self.a01_encoder.parameters()])
        self.register_buffer("target_init_reference", target0)
        self.register_buffer("query_init_flat", query_weight0)

    def target_full(self) -> torch.Tensor:
        gene_repr = self.h07_model.gnn(self.adj_norm)
        target = self.h07_model.label_forward(gene_repr, head="unseen")
        return F.normalize(target, dim=1, eps=1e-8)

    def query(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.a01_encoder(x), dim=1, eps=1e-8)

    def logit_scale(self) -> torch.Tensor:
        return torch.exp(torch.clamp(self.h07_model.logit_scale_unseen, -2.0, 4.0))

    def logits_full(self, x: torch.Tensor) -> torch.Tensor:
        return (self.query(x) @ self.target_full().T) * self.logit_scale()

    def logits_seen(self, x: torch.Tensor) -> torch.Tensor:
        target = self.target_full().index_select(0, self.train_idx)
        return (self.query(x) @ target.T) * self.logit_scale()

    def regularization(self) -> torch.Tensor:
        target_weight = float(self.cfg.get("target_drift_l2", 0.0))
        target = self.target_full()
        ref = self.target_init_reference.to(target.device)
        reg = target_weight * (1.0 - (target * ref).sum(dim=1)).mean()
        reg = reg + self.a01_encoder.drift_loss(float(self.cfg.get("encoder_l2", 0.0)))
        return reg

    @torch.no_grad()
    def drift_stats(self) -> dict[str, float]:
        target = self.target_full().detach().cpu()
        target_drift = float((1.0 - (target * self.target_init_reference.cpu()).sum(dim=1)).mean().item())
        qflat = torch.cat([p.detach().flatten().cpu() for p in self.a01_encoder.parameters()])
        qref = self.query_init_flat.detach().cpu()
        qdenom = torch.norm(qref).item() + 1e-8
        query_drift = float(torch.norm(qflat - qref).item() / qdenom)
        return {"target_norm_drift": target_drift, "query_relative_drift": query_drift}


def build_model(module: Any, data: dict, device: torch.device, train_idx: np.ndarray, cfg: dict[str, Any], query_state: dict[str, torch.Tensor]) -> A01EncoderTrainablePPIModel:
    h07_model, audit = module.build_h07_model(module.import_h07_wrapper(), data, device, load_checkpoint=False)
    for param in h07_model.parameters():
        param.requires_grad_(False)
    for param in h07_model.gnn.parameters():
        param.requires_grad_(True)
    for param in h07_model.label_unseen_private.parameters():
        param.requires_grad_(True)
    h07_model.logit_scale_unseen.requires_grad_(True)
    a01_encoder = Round246QueryEncoder(query_state, cfg)
    adj = module.move_sparse(data["adj_norm"], device)
    model = A01EncoderTrainablePPIModel(h07_model, a01_encoder, adj, train_idx, cfg, n_genes=len(data["gene_list"]))
    model.build_audit = audit
    return model


def better_round259_candidate(candidate: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    if incumbent is None:
        return True
    for key in ["holdout_top5pct", "holdout_top5pct_remove_seen", "seen_Hit@10", "seen_NDCG@10", "epoch_preference"]:
        if candidate[key] > incumbent[key] + 1e-12:
            return True
        if candidate[key] < incumbent[key] - 1e-12:
            return False
    return False


def train_variant(
    module: Any,
    variant: str,
    cfg: dict[str, Any],
    model: A01EncoderTrainablePPIModel,
    splits: dict,
    holdouts: dict,
    train_idx: np.ndarray,
    device: torch.device,
    smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    model = model.to(device)
    encoder_params = model.a01_encoder.base_trainable_parameters()
    adapter_params = model.a01_encoder.adapter_trainable_parameters()
    groups = [
        {"params": encoder_params, "lr": float(cfg["encoder_lr"]), "weight_decay": float(cfg["weight_decay"])},
        {"params": adapter_params, "lr": float(cfg.get("adapter_lr", cfg["encoder_lr"])), "weight_decay": float(cfg["weight_decay"])},
        {"params": list(model.h07_model.gnn.parameters()), "lr": float(cfg["gnn_lr"]), "weight_decay": float(cfg["weight_decay"])},
        {"params": list(model.h07_model.label_unseen_private.parameters()), "lr": float(cfg["label_lr"]), "weight_decay": float(cfg["weight_decay"])},
        {"params": [model.h07_model.logit_scale_unseen], "lr": float(cfg["scale_lr"]), "weight_decay": 0.0},
    ]
    groups = [g for g in groups if len(g["params"]) > 0]
    trainable_params = [p for group in groups for p in group["params"]]
    opt = torch.optim.AdamW(groups)
    tensors = {name: {"x": torch.tensor(item["x"], dtype=torch.float32, device=device), "y": torch.tensor(item["y"], dtype=torch.float32, device=device)} for name, item in splits.items()}
    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    rng = np.random.default_rng(module.SEED)
    n_train = tensors["inner_train"]["x"].shape[0]
    max_epochs = 1 if smoke else int(cfg["max_epochs"])
    min_epochs = 1 if smoke else int(cfg["min_epochs"])
    patience = int(cfg["patience"])
    batch_size = int(cfg["batch_size"])
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
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
            yb_full = module.scatter_local_targets_to_full(yb_local, train_idx_t, int(model.target_full_anchor.shape[0]))
            logits = model.logits_full(xb)
            loss = (
                module.bag_softmax_loss(logits, yb_full)
                + float(cfg["bce_weight"]) * module.drug_normalized_bce_loss(logits, yb_full)
                + model.regularization()
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.5)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        seen_scores = module.predict_seen(model, splits["val"]["x"], device, batch_size=256)
        seen = module.compute_metrics(seen_scores, splits["val"]["y"])
        holdout, _ = module.evaluate_holdout_top5(model, holdouts["holdout_val"], device, batch_size=128, exclude_seen=False)
        holdout_rm, _ = module.evaluate_holdout_top5(model, holdouts["holdout_val"], device, batch_size=128, exclude_seen=True)
        drift = model.drift_stats()
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
            "target_norm_drift": drift["target_norm_drift"],
            "query_relative_drift": drift["query_relative_drift"],
            "logit_scale": float(model.logit_scale().detach().cpu().item()),
            "seed": module.SEED,
            **{k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))},
        }
        history.append(rec)
        if better_round259_candidate(rec, best):
            best = rec
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0:
            module.log(
                f"{variant} epoch={epoch} loss={rec['train_loss']:.4f} "
                f"val_hit10={rec['seen_Hit@10']:.4f} holdout_top5={rec['holdout_top5pct']:.4f} "
                f"rm_seen={rec['holdout_top5pct_remove_seen']:.4f} "
                f"tdrift={rec['target_norm_drift']:.4f} qdrift={rec['query_relative_drift']:.4f}"
            )
        if epoch >= min_epochs and best is not None and epoch - best_epoch >= patience:
            module.log(f"{variant} early stop epoch={epoch}; best_epoch={best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    metrics, per = evaluate_round259_variant(module, variant, model, splits, holdouts, device, best_epoch)
    ckpt_dir = module.CHECKPOINTS / variant
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "best_epoch": best_epoch, "cfg": cfg}, ckpt_dir / "selected_checkpoint.pth")
    return metrics, pd.DataFrame(history), per


@torch.no_grad()
def evaluate_holdout_percent(
    model: A01EncoderTrainablePPIModel,
    bundle: dict[str, Any],
    device: torch.device,
    batch_size: int,
    percent: float,
    exclude_seen: bool = False,
    return_rows: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    model.eval()
    n_genes = int(model.target_full_anchor.shape[0])
    k = max(1, int(math.floor(n_genes * percent)))
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
                        f"success_top{int(percent * 100)}pct": bool(success),
                        "best_rank": int(min(ranks)) if ranks else -1,
                        "n_true_targets": int(len(true_idx)),
                    }
                )
    key = f"top{int(percent * 100)}pct_success"
    return (
        {
            key: successes / len(bundle["x"]) if len(bundle["x"]) else 0.0,
            "n_profiles": int(len(bundle["x"])),
            "candidate_space": int(n_genes),
            "exclude_seen_labels": bool(exclude_seen),
        },
        pd.DataFrame(rows) if return_rows else None,
    )


def evaluate_round259_variant(
    module: Any,
    run_id: str,
    model: A01EncoderTrainablePPIModel,
    splits: dict[str, dict],
    holdouts: dict[str, dict],
    device: torch.device,
    best_epoch: int,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    rows: list[dict[str, Any]] = []
    split_roles = {
        "val": "pseudo_unseen_selection",
        "test": "official_unseen_final",
        "pc3_test": "pc3_seen_cell_diagnostic",
    }
    for split_name in ["val", "test", "pc3_test"]:
        scores = module.predict_seen(model, splits[split_name]["x"], device, batch_size=256)
        metrics = module.compute_metrics(scores, splits[split_name]["y"])
        rows.append(
            {
                "variant": run_id,
                "split": split_name,
                "cell_split_role": split_roles[split_name],
                "evaluation_family": "seen_668",
                "evaluation_unit": "profile",
                **metrics,
                "best_epoch": best_epoch,
            }
        )
    per_rows = []
    holdout_roles = {
        "holdout_val": "pseudo_unseen_selection",
        "holdout_test": "official_unseen_final",
        "pc3_holdout_test": "pc3_seen_cell_diagnostic",
    }
    for split_name in ["holdout_val", "holdout_test", "pc3_holdout_test"]:
        metrics, per = evaluate_holdout_percent(
            model,
            holdouts[split_name],
            device,
            batch_size=256,
            percent=0.05,
            exclude_seen=False,
            return_rows=(split_name in {"holdout_test", "pc3_holdout_test"}),
        )
        metrics10, _ = evaluate_holdout_percent(
            model,
            holdouts[split_name],
            device,
            batch_size=256,
            percent=0.10,
            exclude_seen=False,
            return_rows=False,
        )
        metrics_rm, _ = evaluate_holdout_percent(
            model,
            holdouts[split_name],
            device,
            batch_size=256,
            percent=0.05,
            exclude_seen=True,
            return_rows=False,
        )
        rows.append(
            {
                "variant": run_id,
                "split": split_name,
                "cell_split_role": holdout_roles[split_name],
                "evaluation_family": "holdout_fullspace_top5",
                "evaluation_unit": "profile",
                **metrics,
                **metrics10,
                "best_epoch": best_epoch,
            }
        )
        rows.append(
            {
                "variant": run_id,
                "split": split_name,
                "cell_split_role": holdout_roles[split_name],
                "evaluation_family": "holdout_fullspace_top5_remove_seen",
                "evaluation_unit": "profile",
                **metrics_rm,
                "best_epoch": best_epoch,
            }
        )
        if per is not None:
            per.insert(0, "variant", run_id)
            per.insert(1, "split", split_name)
            per.insert(2, "cell_split_role", holdout_roles[split_name])
            per_rows.append(per)
    return pd.DataFrame(rows), pd.concat(per_rows, ignore_index=True) if per_rows else None


def select_export_variant(history: pd.DataFrame) -> str:
    if history.empty:
        raise RuntimeError("Cannot select export variant from empty history")
    last_best = (
        history.sort_values(
            ["holdout_top5pct", "holdout_top5pct_remove_seen", "seen_Hit@10", "seen_NDCG@10", "epoch_preference"],
            ascending=[False, False, False, False, False],
        )
        .iloc[0]
        .to_dict()
    )
    return str(last_best["variant"])


@torch.no_grad()
def export_selected_target_embedding(
    module: Any,
    model: A01EncoderTrainablePPIModel,
    data: dict[str, Any],
    selected_variant: str,
    selected_ckpt: Path,
    extra_audit: dict[str, Any],
) -> dict[str, Any]:
    model.eval()
    target = model.target_full().detach().cpu()
    genes = [str(g) for g in data["gene_list"]]
    if tuple(target.shape) != (len(genes), 256):
        raise RuntimeError(f"Unexpected Round259 target embedding shape: {tuple(target.shape)} for {len(genes)} genes")
    out_csv = module.SOURCE / "round259_target_embedding.csv"
    df = pd.DataFrame(
        target.numpy().astype(np.float32),
        index=genes,
        columns=[str(i) for i in range(target.shape[1])],
    )
    df.index.name = None
    df.to_csv(out_csv)
    audit = {
        "round": "Round259",
        "selected_variant": selected_variant,
        "selected_checkpoint": str(selected_ckpt),
        "selected_checkpoint_sha256": sha256_file(selected_ckpt),
        "target_csv": str(out_csv),
        "target_csv_sha256": sha256_file(out_csv),
        "target_shape": [int(target.shape[0]), int(target.shape[1])],
        "target_norm_min": float(target.norm(dim=1).min().item()),
        "target_norm_max": float(target.norm(dim=1).max().item()),
        "gene_count": int(len(genes)),
        "first_genes": genes[:5],
        "last_genes": genes[-5:],
        **extra_audit,
    }
    (module.SOURCE / "round259_target_embedding_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def write_report(module: Any, metrics: pd.DataFrame, history: pd.DataFrame, audits: list[dict[str, Any]]) -> None:
    shown = metrics[metrics["split"].isin(["test", "holdout_test"])].copy()
    cols = [
        "variant",
        "split",
        "evaluation_family",
        "Hit@10",
        "MeanAP",
        "AUROC",
        "NDCG@10",
        "top5pct_success",
        "top10pct_success",
        "exclude_seen_labels",
        "n_profiles",
        "candidate_space",
        "best_epoch",
    ]
    table = shown[[c for c in cols if c in shown.columns]].to_string(index=False)
    hbest = (
        history.groupby("variant", as_index=False)
        .agg(
            max_val_holdout=("holdout_top5pct", "max"),
            max_val_remove_seen=("holdout_top5pct_remove_seen", "max"),
            max_seen_hit10=("seen_Hit@10", "max"),
            min_loss=("train_loss", "min"),
            max_target_drift=("target_norm_drift", "max"),
            max_query_drift=("query_relative_drift", "max"),
            epochs=("epoch", "max"),
        )
        .to_string(index=False)
    )
    text = f"""# Round259 cross-cell full-space target embedding report

## Scope

Sample side is initialized from the Round246/Round252-family `query_encoder/query_norm` and finetuned. The Round252 adapter is not required for the final full-space forward. Target side keeps the Round255/Round240 full-space architecture: untrained MeTar h07 PPI GCN plus untrained unseen label tower trained from scratch. Training uses Round252 base-train cells, pseudo-unseen cells only for selection, and official unseen cells only for final reporting. Whole-model checkpoints are selected by one validation epoch; encoder and decoder states are never mixed across epochs.

## Test results

```text
{table}
```

## Training-history maxima

```text
{hbest}
```

## Build audit

```json
{json.dumps(audits, indent=2)}
```
"""
    (module.ROUND_ROOT / "round259_result_report_zh.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=str, default=",".join(VARIANTS.keys()))
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module = load_round230_module()
    module.ROUND_ROOT = ROUND_ROOT
    module.SOURCE = ROUND_ROOT / "source_data"
    module.QA = ROUND_ROOT / "qa"
    module.CHECKPOINTS = ROUND_ROOT / "checkpoints"
    module.ensure_dirs()
    module.set_seed(module.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.log(f"round259 device={device}")
    wrapper, data, protocol = module.load_h07_data()
    protocol_audit = load_round252_protocol_audit()
    splits, holdouts, split_counts = build_round259_splits(module, data, protocol_audit)
    train_idx = data["train_gene_global_idx"].detach().cpu().numpy().astype(np.int64)
    query_state, query_audit = load_round246_query_state()

    metrics_all = []
    history_all = []
    per_all = []
    audits = []
    trained_models: dict[str, A01EncoderTrainablePPIModel] = {}
    for variant in [x for x in args.runs.split(",") if x]:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        module.set_seed(module.SEED)
        cfg = deepcopy(VARIANTS[variant])
        cfg["target_mode"] = "round259_round246query_crosscell_round240_fullspace_ppi_t4"
        cfg["ablation"] = f"round259_{variant}_round246query_crosscell_fullspace_ppi_t4"
        model = build_model(module, data, device, train_idx, cfg, query_state)
        audits.append(
            {
                "variant": variant,
                "round246_query_encoder_initialized": True,
                "query_audit": query_audit,
                "h07_checkpoint_loaded": False,
                "target_side": "trainable untrained MeTar h07 PPI GCN + trainable untrained unseen label tower",
                "trainable_modules": ["round246_query_encoder_selected_layers", "optional_query_adapter", "gnn", "label_unseen_private", "logit_scale_unseen"],
                "single_epoch_checkpoint_policy": "whole model state selected at one validation epoch",
                "cfg": cfg,
                "build_h07_audit": getattr(model, "build_audit", {}),
                "round252_cell_protocol": split_counts,
            }
        )
        metrics, history, per = train_variant(module, variant, cfg, model, splits, holdouts, train_idx, device, smoke=bool(args.smoke))
        trained_models[variant] = model
        for df in [metrics, history]:
            df["target_mode"] = cfg["target_mode"]
            df["ablation"] = cfg["ablation"]
        metrics_all.append(metrics)
        history_all.append(history)
        if per is not None:
            per["target_mode"] = cfg["target_mode"]
            per_all.append(per)

    metrics_df = pd.concat(metrics_all, ignore_index=True)
    history_df = pd.concat(history_all, ignore_index=True)
    metrics_df.to_csv(module.SOURCE / "metric_summary.tsv", sep="\t", index=False)
    metrics_df.to_csv(module.SOURCE / "metric_summary_by_cell_split.tsv", sep="\t", index=False)
    history_df.to_csv(module.SOURCE / "training_history.tsv", sep="\t", index=False)
    if per_all:
        pd.concat(per_all, ignore_index=True).to_csv(module.SOURCE / "per_profile_holdout_metrics.tsv.gz", sep="\t", index=False, compression="gzip")
    (module.SOURCE / "target_mode_audit.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    (module.SOURCE / "round246_query_checkpoint_audit.json").write_text(json.dumps(query_audit, indent=2), encoding="utf-8")
    (module.SOURCE / "round252_cell_protocol_audit.json").write_text(json.dumps(split_counts, indent=2), encoding="utf-8")
    selected_variant = select_export_variant(history_df)
    selected_ckpt = module.CHECKPOINTS / selected_variant / "selected_checkpoint.pth"
    embedding_audit = export_selected_target_embedding(
        module,
        trained_models[selected_variant],
        data,
        selected_variant,
        selected_ckpt,
        {"round252_protocol_audit_path": str(ROUND252_PROTOCOL_AUDIT), "split_counts": split_counts},
    )
    (module.ROUND_ROOT / "method_lock.json").write_text(
        json.dumps(
            {
                "round": "round259_crosscell_fullspace_round252query",
                "seed": module.SEED,
                "counts": split_counts,
                "variants": VARIANTS,
                "query_initializer": "Round246 whole-model checkpoint readout_state_dict query_encoder/query_norm",
                "round246_checkpoint": str(ROUND246_CKPT),
                "round246_checkpoint_sha256": query_audit["round246_checkpoint_sha256"],
                "round252_protocol_audit": str(ROUND252_PROTOCOL_AUDIT),
                "selected_export_variant": selected_variant,
                "round259_target_embedding_audit": embedding_audit,
                "h07_checkpoint_loaded": False,
                "single_epoch_checkpoint_policy": "selected checkpoint saves encoder and decoder together from the same epoch",
                "round230_base_script": str(ROUND230_SCRIPT),
                "protocol_keys": sorted(protocol.keys()) if isinstance(protocol, dict) else [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    audit_text = f"""# Round259 audit

- Sample encoder is initialized from Round246 whole-model `query_encoder/query_norm`: `{ROUND246_CKPT}`.
- Round252 adapter is not used in the final full-space forward or export.
- No trained MeTar h07 checkpoint is loaded.
- PPI GCN and unseen label tower are newly initialized and trainable.
- Query encoder and target-side modules are optimized jointly within each epoch.
- Training uses Round252 base-train cells only.
- Pseudo-unseen cells are used only for epoch/model selection.
- Official unseen cells are used only for final metrics and never enter training or selection.
- Checkpoint selection saves and restores the whole model state from one validation-selected epoch; encoder and decoder are not mixed across different epochs.
- Loss uses bag-softmax with the same optional BCE term as Round255/Round240.
- Training labels are restricted to the original 668 seen-label genes scattered into the 16,583-gene universe.
- Holdout target labels are used only for validation/test evaluation.
- Exported target embedding: `{module.SOURCE / 'round259_target_embedding.csv'}`.

```json
{json.dumps(split_counts, indent=2)}
```
"""
    (module.QA / "leakage_and_alignment_audit.md").write_text(audit_text, encoding="utf-8")
    (module.ROUND_ROOT / "leakage_and_denominator_audit.md").write_text(audit_text, encoding="utf-8")
    write_report(module, metrics_df, history_df, audits)
    module.log(f"round259 wrote {module.SOURCE / 'metric_summary.tsv'}")


if __name__ == "__main__":
    main()
