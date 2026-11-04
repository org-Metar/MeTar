#!/usr/bin/env python3
"""Initialize and evaluate query-gated support adaptation from a fixed bulk model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from metar.runtime import report_metrics, report_progress, report_step, resolve_device

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None


ROUND_ROOT = Path(__file__).resolve().parents[1]
METAR_ROOT = ROUND_ROOT.parent
SOURCE = ROUND_ROOT / "source_data"
QA = ROUND_ROOT / "qa"
CHECKPOINTS = ROUND_ROOT / "checkpoints"
LOGS = ROUND_ROOT / "logs"

ROUND246_ROOT = METAR_ROOT / "claude_round246_joint_seen_target_train_pc3_a375_mcf7_20260519"
ROUND246_SCRIPT = ROUND246_ROOT / "scripts" / "run_round246_joint_seen_target_train_pc3_a375_mcf7.py"
ROUND246_SELECTED = ROUND246_ROOT / "source_data" / "selected_r246_joint_lm0p5_sc0p15_mn2p5.json"

SEED = 20260520
SUPPORT_SIZES = [0, 8, 16, 32, 64, 128, 256, -1]
SELECTORS = ["top_count", "random", "diverse_label"]
HIT_KS = [1, 5, 10, 20, 30, 50]

ADAPTER_CFG = {
    "variant": "r247_query_gated_h128_sc0p15_mn2p5",
    "hidden_dim": 128,
    "dropout": 0.10,
    "max_delta_norm": 2.5,
    "shift_scale": 0.15,
    "support_size": 128,
    "query_size": 32,
    "tasks_per_epoch": 240,
    "max_epochs": 50,
    "min_epochs": 8,
    "patience": 8,
    "lr": 2.0e-4,
    "weight_decay": 1.0e-5,
    "delta_l2": 1.0e-3,
    "gate_l2": 1.0e-3,
    "base_margin_weight": 0.10,
}


def require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise ModuleNotFoundError("torch is required; run on the remote MeTar/PertKGE environment.")


def log(message: str) -> None:
    print(f"[round247] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def stable_seed(*parts: Any) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:12], 16) % (2**31 - 1)


def ensure_dirs() -> None:
    for path in [SOURCE, QA, CHECKPOINTS, LOGS]:
        path.mkdir(parents=True, exist_ok=True)


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


def load_round246():
    module = import_from_path("round247_round246", ROUND246_SCRIPT)
    module.ROUND_ROOT = ROUND_ROOT
    module.SOURCE = SOURCE
    module.QA = QA
    module.CHECKPOINTS = CHECKPOINTS
    module.LOGS = LOGS
    return module


def load_context(device: torch.device):
    report_step('Loading checkpoints and preparing bulk split tables')
    r246 = load_round246()
    args = argparse.Namespace(
        kg_checkpoint=None,
        pseudo_unseen_cells=",".join(r246.DEFAULT_R246_PSEUDO_UNSEEN_VAL_CELLS),
        official_unseen_cells="round215_heldout_panel",
        seen_diagnostic_cells=",".join(r246.FORCED_TRAIN_CELLS),
    )
    r243, r211, r208, labels, kg_path, kg_ckpt, tables, data_audit = r246.load_context(args, device)
    r243.write_round243_audit_tables(tables, data_audit)
    selected = json.loads(ROUND246_SELECTED.read_text(encoding="utf-8"))
    selected_checkpoint = Path(selected["checkpoint"])
    if not selected_checkpoint.is_absolute():
        selected_checkpoint = ROUND246_ROOT / selected_checkpoint
    ckpt = torch.load(selected_checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["joint_cfg"]
    readout_cfg = ckpt["readout_cfg"]
    target_init, target_audit = r211.target_embeddings_from_ckpt(kg_ckpt, labels, cfg["variant"])
    model = r211.DiagReadout(target_init, readout_cfg).to(device)
    model.load_state_dict(ckpt["readout_state_dict"], strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    cell_adapter = CellLevelSupportAdapter(
        model.target_anchor.detach(),
        hidden_dim=int(cfg["support_hidden_dim"]),
        dropout=float(cfg["support_dropout"]),
        max_delta_norm=float(cfg["max_delta_norm"]),
    ).to(device)
    cell_adapter.load_state_dict(ckpt["support_adapter_state_dict"], strict=True)
    cell_adapter.eval()
    for param in cell_adapter.parameters():
        param.requires_grad_(False)
    for table in tables.values():
        r243.precompute_q(model, table, device)
    context = {
        "r246": r246,
        "r243": r243,
        "r211": r211,
        "r208": r208,
        "labels": labels,
        "kg_path": kg_path,
        "kg_checkpoint_sha256": r243.sha256_file(kg_path),
        "round246_selected": selected,
        "round246_ckpt": ckpt,
        "target_audit": target_audit,
        "tables": tables,
        "data_audit": data_audit,
        "model": model,
        "round246_cell_adapter": cell_adapter,
    }
    return context


def metric_cols() -> list[str]:
    return [f"Hit@{k}" for k in HIT_KS] + ["AUROC", "NDCG@10", "MeanAP"]


def current_target_tensor(model: nn.Module) -> torch.Tensor:
    return F.normalize(model.target_anchor + model.target_delta, dim=1)


def tensor_slice(item: dict, key: str, idx: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(item[key][idx], dtype=torch.float32, device=device)


def support_size_value(size: int, available: int) -> int:
    if size < 0:
        return int(available)
    if size == 0:
        return 0
    return int(min(size, available))


def select_support_indices(
    rows: pd.DataFrame,
    y: np.ndarray,
    size: int,
    selector: str,
    seed: int,
) -> np.ndarray:
    n = int(len(rows))
    k = support_size_value(size, n)
    if k <= 0:
        return np.zeros((0,), dtype=np.int64)
    if k >= n:
        return np.arange(n, dtype=np.int64)
    if selector == "top_count":
        order = np.lexsort(
            (
                rows["profile_id"].astype(str).to_numpy(),
                rows["drug"].astype(str).to_numpy(),
                rows["n_profiles"].to_numpy(dtype=np.int64) * -1,
            )
        )
        return order[:k].astype(np.int64)
    if selector == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(np.arange(n, dtype=np.int64), size=k, replace=False)).astype(np.int64)
    if selector == "diverse_label":
        target_sets = [set(np.where(y[i] > 0)[0].tolist()) for i in range(n)]
        selected: list[int] = []
        covered: set[int] = set()
        remaining = set(range(n))
        weights = rows["n_profiles"].to_numpy(dtype=np.float64)
        while remaining and len(selected) < k:
            best_i = None
            best_key = None
            for i in remaining:
                gain = len(target_sets[i] - covered)
                total = len(target_sets[i])
                key = (gain, total, float(weights[i]), -i)
                if best_key is None or key > best_key:
                    best_key = key
                    best_i = i
            assert best_i is not None
            selected.append(best_i)
            covered.update(target_sets[best_i])
            remaining.remove(best_i)
        return np.asarray(selected, dtype=np.int64)
    raise ValueError(f"Unknown selector: {selector}")


def support_audit(rows: pd.DataFrame, y: np.ndarray, idx: np.ndarray, selector: str, requested_size: int) -> dict[str, Any]:
    if len(idx) == 0:
        return {
            "selector": selector,
            "requested_support_size": int(requested_size),
            "actual_support_size": 0,
            "support_profiles": 0,
            "support_target_coverage": 0,
            "support_target_entropy": 0.0,
            "profile_count_gini": 0.0,
        }
    selected_rows = rows.iloc[idx]
    y_sel = y[idx]
    target_counts = y_sel.sum(axis=0).astype(np.float64)
    positive_counts = target_counts[target_counts > 0]
    probs = positive_counts / max(float(positive_counts.sum()), 1.0)
    entropy = float(-(probs * np.log2(np.clip(probs, 1.0e-12, 1.0))).sum()) if len(probs) else 0.0
    profile_counts = selected_rows["n_profiles"].to_numpy(dtype=np.float64)
    if len(profile_counts) <= 1 or np.isclose(profile_counts.sum(), 0.0):
        gini = 0.0
    else:
        sorted_counts = np.sort(profile_counts)
        n = len(sorted_counts)
        gini = float((2 * np.arange(1, n + 1) - n - 1).dot(sorted_counts) / (n * sorted_counts.sum()))
    return {
        "selector": selector,
        "requested_support_size": int(requested_size),
        "actual_support_size": int(len(idx)),
        "support_profiles": int(selected_rows["n_profiles"].sum()),
        "support_target_coverage": int((target_counts > 0).sum()),
        "support_target_entropy": entropy,
        "profile_count_gini": gini,
    }


def drug_overlap_count(support_rows: pd.DataFrame, query_rows: pd.DataFrame, idx: np.ndarray) -> int:
    if len(idx) == 0 or len(query_rows) == 0:
        return 0
    support_drugs = set(support_rows.iloc[idx]["drug"].astype(str))
    query_drugs = set(query_rows["drug"].astype(str))
    return len(support_drugs.intersection(query_drugs))


class QueryGatedSupportAdapter(nn.Module):
    def __init__(self, target_anchor: torch.Tensor, hidden_dim: int, dropout: float, max_delta_norm: float) -> None:
        super().__init__()
        dim = int(target_anchor.shape[1])
        self.dim = dim
        self.max_delta_norm = float(max_delta_norm)
        self.register_buffer("target_anchor", target_anchor.detach().clone())
        in_dim = dim * 4
        self.support_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.key_proj = nn.Linear(in_dim, dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.out_norm = nn.LayerNorm(dim)

    def support_features(self, support_q: torch.Tensor, support_y: torch.Tensor) -> torch.Tensor:
        pos_count = torch.clamp(support_y.sum(dim=1, keepdim=True), min=1.0)
        pos_proto = F.normalize((support_y @ self.target_anchor) / pos_count, dim=1)
        return torch.cat([support_q, pos_proto, support_q * pos_proto, pos_proto - support_q], dim=1)

    def forward(
        self,
        query_q: torch.Tensor,
        support_q: torch.Tensor,
        support_y: torch.Tensor,
        support_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.support_features(support_q, support_y)
        row_delta = self.support_mlp(features)
        support_key = F.normalize(self.key_proj(features), dim=1)
        attn_logits = query_q @ support_key.T / math.sqrt(float(self.dim))
        attn_logits = attn_logits + torch.log(torch.clamp(support_w, min=1.0e-6))[None, :]
        attn = torch.softmax(attn_logits, dim=1)
        delta = attn @ row_delta
        delta = self.out_norm(delta)
        if self.max_delta_norm > 0:
            norm = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
            shrink = torch.clamp(
                torch.tensor(self.max_delta_norm, dtype=delta.dtype, device=delta.device)
                / torch.clamp(norm, min=1.0e-6),
                max=1.0,
            )
            delta = delta * shrink
        context = attn @ support_key
        gate_in = torch.cat([query_q, context, query_q * context, context - query_q], dim=1)
        gate = torch.sigmoid(self.gate_mlp(gate_in)).squeeze(-1)
        return delta, gate


class CellLevelSupportAdapter(nn.Module):
    def __init__(self, target_anchor: torch.Tensor, hidden_dim: int, dropout: float, max_delta_norm: float) -> None:
        super().__init__()
        dim = int(target_anchor.shape[1])
        self.dim = dim
        self.max_delta_norm = float(max_delta_norm)
        self.register_buffer("target_anchor", target_anchor.detach().clone())
        in_dim = dim * 4
        self.row_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.attn_mlp = nn.Sequential(
            nn.Linear(in_dim, max(16, hidden_dim // 2)),
            nn.GELU(),
            nn.Linear(max(16, hidden_dim // 2), 1),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, support_q: torch.Tensor, support_y: torch.Tensor, support_w: torch.Tensor) -> torch.Tensor:
        pos_count = torch.clamp(support_y.sum(dim=1, keepdim=True), min=1.0)
        pos_proto = F.normalize((support_y @ self.target_anchor) / pos_count, dim=1)
        features = torch.cat([support_q, pos_proto, support_q * pos_proto, pos_proto - support_q], dim=1)
        row_delta = self.row_mlp(features)
        attn_logits = self.attn_mlp(features).squeeze(-1)
        attn_logits = attn_logits + torch.log(torch.clamp(support_w, min=1.0e-6))
        attn = torch.softmax(attn_logits, dim=0)
        delta = torch.sum(attn[:, None] * row_delta, dim=0)
        delta = self.out_norm(delta[None, :]).squeeze(0)
        if self.max_delta_norm > 0:
            norm = torch.linalg.vector_norm(delta)
            shrink = torch.clamp(
                torch.tensor(self.max_delta_norm, dtype=delta.dtype, device=delta.device)
                / torch.clamp(norm, min=1.0e-6),
                max=1.0,
            )
            delta = delta * shrink
        return delta


def score_with_query_delta(model: nn.Module, query_q: torch.Tensor, delta: torch.Tensor, gate: torch.Tensor, scale: float) -> torch.Tensor:
    adapted_q = F.normalize(query_q + float(scale) * gate[:, None] * delta, dim=1)
    target = current_target_tensor(model)
    score = (adapted_q * model.diag[None, :]) @ target.T
    score = score * torch.exp(torch.clamp(model.score_scale, -2.0, 2.0))
    return score + model.target_bias[None, :]


def score_with_cell_delta(model: nn.Module, query_q: torch.Tensor, delta: torch.Tensor, scale: float) -> torch.Tensor:
    adapted_q = F.normalize(query_q + float(scale) * delta[None, :], dim=1)
    target = current_target_tensor(model)
    score = (adapted_q * model.diag[None, :]) @ target.T
    score = score * torch.exp(torch.clamp(model.score_scale, -2.0, 2.0))
    return score + model.target_bias[None, :]


def retrieval_loss_from_logits(r243, logits: torch.Tensor, y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return r243.retrieval_loss_from_logits(logits, y, weights)


def sample_episode(item: dict, cell: str, support_size: int, query_size: int, selector: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    rows = item["rows"]
    all_idx = np.asarray(item["by_cell"][cell], dtype=np.int64)
    cell_rows = rows.iloc[all_idx].reset_index(drop=True)
    drugs = cell_rows["drug"].astype(str).unique()
    if len(drugs) < 2:
        raise RuntimeError(f"Cell {cell} has too few drugs for an episode")
    qn = min(query_size, max(1, len(drugs) // 5))
    query_drugs = set(map(str, rng.choice(drugs, size=qn, replace=False)))
    drug_values = cell_rows["drug"].astype(str).to_numpy()
    query_local_all = np.where(np.isin(drug_values, list(query_drugs)))[0]
    support_local_all = np.where(~np.isin(drug_values, list(query_drugs)))[0]
    if len(query_local_all) > query_size:
        query_local_all = rng.choice(query_local_all, size=query_size, replace=False)
    support_rows = cell_rows.iloc[support_local_all].reset_index(drop=True)
    support_y = item["y"][all_idx[support_local_all]]
    support_local = select_support_indices(support_rows, support_y, support_size, selector, int(rng.integers(0, 2**31 - 1)))
    return all_idx[support_local_all[support_local]], all_idx[query_local_all]


def eligible_cells(item: dict, min_support: int, query_size: int) -> list[str]:
    rows = item["rows"]
    cells = []
    for cell, idx in item["by_cell"].items():
        drugs = rows.iloc[np.asarray(idx, dtype=np.int64)]["drug"].astype(str).nunique()
        if drugs >= max(2, min_support + min(query_size, 4)):
            cells.append(str(cell))
    return cells


def train_query_gated_adapter(context: dict[str, Any], device: torch.device, smoke: bool = False) -> tuple[QueryGatedSupportAdapter, dict[str, Any], pd.DataFrame]:
    report_step('Preparing query-gated support initializer')
    r243 = context["r243"]
    r208 = context["r208"]
    model = context["model"]
    tables = context["tables"]
    cfg = dict(ADAPTER_CFG)
    if smoke:
        cfg["variant"] = cfg["variant"] + "_smoke"
        cfg["max_epochs"] = 1
        cfg["min_epochs"] = 1
        cfg["patience"] = 1
        cfg["tasks_per_epoch"] = 8
        cfg["support_size"] = 32
    adapter = QueryGatedSupportAdapter(
        model.target_anchor.detach(),
        hidden_dim=int(cfg["hidden_dim"]),
        dropout=float(cfg["dropout"]),
        max_delta_norm=float(cfg["max_delta_norm"]),
    ).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    train_item = tables["adapter_train"]
    cells = eligible_cells(train_item, min_support=8, query_size=int(cfg["query_size"]))
    if not cells:
        raise RuntimeError("No eligible adapter-train cells for Round247")
    rng = np.random.default_rng(SEED)
    best = None
    best_state = None
    stale = 0
    history = []
    report_progress(0, int(cfg["max_epochs"]), label='query gated adapter epochs')
    report_metrics(cfg, label="Training settings")
    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        adapter.train()
        losses = []
        gate_means = []
        delta_norms = []
        for _ in range(int(cfg["tasks_per_epoch"])):
            report_progress(_, int(cfg["tasks_per_epoch"]), label=f"Epoch {epoch} training episodes")
            cell = str(rng.choice(cells))
            support_idx, query_idx = sample_episode(train_item, cell, int(cfg["support_size"]), int(cfg["query_size"]), "diverse_label", rng)
            if len(support_idx) == 0 or len(query_idx) == 0:
                continue
            support_q = tensor_slice(train_item, "q", support_idx, device)
            support_y = tensor_slice(train_item, "y", support_idx, device)
            support_w = tensor_slice(train_item, "weights", support_idx, device)
            query_q = tensor_slice(train_item, "q", query_idx, device)
            query_y = tensor_slice(train_item, "y", query_idx, device)
            query_w = tensor_slice(train_item, "weights", query_idx, device)
            with torch.no_grad():
                base_logits = r243.score_from_q(model, query_q, delta=None, scale=0.0)
            delta, gate = adapter(query_q, support_q, support_y, support_w)
            logits = score_with_query_delta(model, query_q, delta, gate, float(cfg["shift_scale"]))
            loss = retrieval_loss_from_logits(r243, logits, query_y, query_w)
            loss = loss + float(cfg["delta_l2"]) * torch.square(delta).mean()
            loss = loss + float(cfg["gate_l2"]) * torch.square(gate).mean()
            pos_mask = query_y > 0
            if pos_mask.any():
                base_pos = base_logits.masked_fill(~pos_mask, -1.0e9).max(dim=1).values
                adapt_pos = logits.masked_fill(~pos_mask, -1.0e9).max(dim=1).values
                loss = loss + float(cfg["base_margin_weight"]) * F.relu(base_pos - adapt_pos).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            gate_means.append(float(gate.mean().detach().cpu()))
            delta_norms.append(float(torch.linalg.vector_norm(delta, dim=1).mean().detach().cpu()))
        report_progress(int(cfg["tasks_per_epoch"]), int(cfg["tasks_per_epoch"]), label=f"Epoch {epoch} training episodes")
        report_step(f"Epoch {epoch}: evaluating existing training and validation metrics")
        rec = evaluate_selection(context, adapter, device, cfg, epoch, float(np.mean(losses)), float(np.mean(gate_means)), float(np.mean(delta_norms)))
        history.append(rec)
        report_progress(epoch, int(cfg["max_epochs"]), label='query gated adapter epochs')
        report_metrics(
            {key: rec[key] for key in ("train_loss", "pseudo_val_Hit@10", "pseudo_val_MeanAP")},
            label=f"Epoch {epoch}/{int(cfg['max_epochs'])}",
        )
        if epoch == 1 or epoch % 5 == 0:
            log(
                f"{cfg['variant']} epoch={epoch} pseudo_hit10={rec['pseudo_val_Hit@10']:.4f} "
                f"loss={rec['train_loss']:.4f} gate={rec['gate_mean']:.3f}"
            )
        if is_better(rec, best):
            best = rec
            best_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch >= int(cfg["min_epochs"]) and stale >= int(cfg["patience"]):
            log(f"{cfg['variant']} early stop at epoch={epoch}; best_epoch={best['epoch'] if best else 'NA'}")
            break
    if best is None or best_state is None:
        raise RuntimeError("No Round247 adapter selected")
    report_metrics(
        {"epochs_run": len(history), "best_epoch": best["epoch"],
         "pseudo_val_Hit@10": best["pseudo_val_Hit@10"],
         "pseudo_val_MeanAP": best["pseudo_val_MeanAP"]},
        label="Selected support initializer validation metrics",
    )
    adapter.load_state_dict(best_state, strict=True)
    ckpt_dir = CHECKPOINTS / cfg["variant"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "adapter_checkpoint.pth"
    torch.save(
        {
            "stage": "round247_query_gated_adapter",
            "adapter_state_dict": best_state,
            "cfg": cfg,
            "selection": best,
            "round246_checkpoint": context["round246_selected"]["checkpoint"],
        },
        ckpt_path,
    )
    selected = {
        "variant": cfg["variant"],
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": context["r243"].sha256_file(ckpt_path),
        "selection": best,
        "cfg": cfg,
    }
    (SOURCE / f"selected_{cfg['variant']}.json").write_text(json.dumps(json_ready(selected), indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(SOURCE / f"training_history_{cfg['variant']}.tsv", sep="\t", index=False)
    return adapter, selected, pd.DataFrame(history)


def evaluate_selection(context: dict[str, Any], adapter: QueryGatedSupportAdapter, device: torch.device, cfg: dict[str, Any], epoch: int, train_loss: float, gate_mean: float, delta_norm: float) -> dict[str, Any]:
    metrics, _cell, _per, _audit = evaluate_query_gated(
        context,
        adapter,
        device,
        split_prefix="pseudo",
        support_key="pseudo_support",
        query_key="pseudo_val",
        unit="drug-mean",
        support_size=int(cfg["support_size"]),
        selector="diverse_label",
        variant=f"{cfg['variant']}_pseudo_val_adapt",
        keep_per_unit=False,
    )
    row = metrics.iloc[0].to_dict()
    return {
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "gate_mean": float(gate_mean),
        "delta_norm": float(delta_norm),
        "pseudo_val_Hit@10": float(row.get("Hit@10", float("nan"))),
        "pseudo_val_MeanAP": float(row.get("MeanAP", float("nan"))),
        "pseudo_val_NDCG@10": float(row.get("NDCG@10", float("nan"))),
        "pseudo_val_AUROC": float(row.get("AUROC", float("nan"))),
    }


def is_better(candidate: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    if incumbent is None:
        return True
    for key in ["pseudo_val_Hit@10", "pseudo_val_MeanAP", "pseudo_val_NDCG@10", "pseudo_val_AUROC"]:
        left = float(candidate.get(key, float("nan")))
        right = float(incumbent.get(key, float("nan")))
        if math.isnan(left) and math.isnan(right):
            continue
        if math.isnan(left):
            return False
        if math.isnan(right):
            return True
        if left > right + 1.0e-12:
            return True
        if left < right - 1.0e-12:
            return False
    return False


def evaluate_no_adapt(context: dict[str, Any], query_key: str, unit: str, variant: str, keep_per_unit: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    r243 = context["r243"]
    r208 = context["r208"]
    model = context["model"]
    item = context["tables"][query_key]
    cell_rows = []
    per_parts = []
    for _cell_index, (cell, qidx_raw) in enumerate(item["by_cell"].items()):
        report_progress(_cell_index, len(item["by_cell"]), label=f"{variant}: cells inspected")
        qidx = np.asarray(qidx_raw, dtype=np.int64)
        if len(qidx) == 0:
            continue
        qrows = item["rows"].iloc[qidx].reset_index(drop=True)
        q = torch.tensor(item["q"][qidx], dtype=torch.float32, device=next(model.parameters()).device)
        scores = r243.score_from_q(model, q, delta=None, scale=0.0).detach().cpu().numpy()
        summary, per = r208.compute_metrics(scores, item["y"][qidx], qrows, "Round247", variant, unit)
        summary.update({"cell_line": cell, "adapt_support_size": 0, "selector": "none", "gate_mean": float("nan"), "delta_norm": float("nan")})
        cell_rows.append(summary)
        if keep_per_unit:
            per["adapt_support_size"] = 0
            per["selector"] = "none"
            per_parts.append(per)
    report_progress(len(item["by_cell"]), len(item["by_cell"]), label=f"{variant}: cells inspected")
    cell_df = pd.DataFrame(cell_rows)
    metric = macro_from_cells(cell_df, variant, item.get("split_name", query_key), unit, "Round247_no_adapt")
    return pd.DataFrame([metric]), cell_df, pd.concat(per_parts, ignore_index=True) if per_parts else pd.DataFrame()


def evaluate_query_gated(
    context: dict[str, Any],
    adapter: QueryGatedSupportAdapter,
    device: torch.device,
    split_prefix: str,
    support_key: str,
    query_key: str,
    unit: str,
    support_size: int,
    selector: str,
    variant: str,
    keep_per_unit: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    r243 = context["r243"]
    r208 = context["r208"]
    model = context["model"]
    support_item = context["tables"][support_key]
    query_item = context["tables"][query_key]
    adapter.eval()
    cell_rows = []
    per_parts = []
    audit_rows = []
    with torch.no_grad():
        for _cell_index, (cell, qidx_raw) in enumerate(query_item["by_cell"].items()):
            report_progress(_cell_index, len(query_item["by_cell"]), label=f"{variant}: cells inspected")
            qidx = np.asarray(qidx_raw, dtype=np.int64)
            support_all = np.asarray(support_item["by_cell"].get(cell, np.asarray([], dtype=np.int64)), dtype=np.int64)
            if len(qidx) == 0 or len(support_all) == 0:
                continue
            support_rows = support_item["rows"].iloc[support_all].reset_index(drop=True)
            support_y_np = support_item["y"][support_all]
            local = select_support_indices(support_rows, support_y_np, support_size, selector, seed=stable_seed(SEED, cell, selector, support_size))
            support_idx = support_all[local]
            audit = support_audit(support_rows, support_y_np, local, selector, support_size)
            qrows = query_item["rows"].iloc[qidx].reset_index(drop=True)
            overlap = drug_overlap_count(support_rows, qrows, local)
            if overlap:
                raise RuntimeError(f"Support/query drug overlap for {cell}/{query_key}: {overlap}")
            audit.update({"cell_line": cell, "split": query_key, "evaluation_unit": unit, "support_query_drug_overlap": overlap})
            audit_rows.append(audit)
            if len(support_idx) == 0:
                continue
            support_q = tensor_slice(support_item, "q", support_idx, device)
            support_y = tensor_slice(support_item, "y", support_idx, device)
            support_w = tensor_slice(support_item, "weights", support_idx, device)
            q = tensor_slice(query_item, "q", qidx, device)
            delta, gate = adapter(q, support_q, support_y, support_w)
            scores = score_with_query_delta(model, q, delta, gate, float(ADAPTER_CFG["shift_scale"])).detach().cpu().numpy()
            summary, per = r208.compute_metrics(scores, query_item["y"][qidx], qrows, "Round247", variant, unit)
            summary.update(
                {
                    "cell_line": cell,
                    "adapt_support_size": int(len(support_idx)),
                    "requested_support_size": int(support_size),
                    "selector": selector,
                    "gate_mean": float(gate.mean().detach().cpu()),
                    "delta_norm": float(torch.linalg.vector_norm(delta, dim=1).mean().detach().cpu()),
                }
            )
            cell_rows.append(summary)
            if keep_per_unit:
                per["adapt_support_size"] = int(len(support_idx))
                per["requested_support_size"] = int(support_size)
                per["selector"] = selector
                per["gate_mean"] = float(gate.mean().detach().cpu())
                per_parts.append(per)
        report_progress(len(query_item["by_cell"]), len(query_item["by_cell"]), label=f"{variant}: cells inspected")
    cell_df = pd.DataFrame(cell_rows)
    metric = macro_from_cells(cell_df, variant, query_key, unit, "Round247_query_gated")
    metric["selector"] = selector
    metric["requested_support_size"] = int(support_size)
    per_df = pd.concat(per_parts, ignore_index=True) if per_parts else pd.DataFrame()
    audit_df = pd.DataFrame(audit_rows)
    return pd.DataFrame([metric]), cell_df, per_df, audit_df


def evaluate_round246_cell_adapter(
    context: dict[str, Any],
    device: torch.device,
    query_key: str,
    unit: str,
    variant: str,
    keep_per_unit: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    r208 = context["r208"]
    model = context["model"]
    adapter = context["round246_cell_adapter"]
    support_item = context["tables"]["official_support"]
    query_item = context["tables"][query_key]
    cfg = context["round246_ckpt"]["joint_cfg"]
    support_size = int(cfg["support_size"])
    shift_scale = float(cfg["shift_scale"])
    adapter.eval()
    cell_rows = []
    per_parts = []
    audit_rows = []
    with torch.no_grad():
        for _cell_index, (cell, qidx_raw) in enumerate(query_item["by_cell"].items()):
            report_progress(_cell_index, len(query_item["by_cell"]), label=f"{variant}: cells inspected")
            qidx = np.asarray(qidx_raw, dtype=np.int64)
            support_all = np.asarray(support_item["by_cell"].get(cell, np.asarray([], dtype=np.int64)), dtype=np.int64)
            if len(qidx) == 0 or len(support_all) == 0:
                continue
            support_rows = support_item["rows"].iloc[support_all].reset_index(drop=True)
            support_y_np = support_item["y"][support_all]
            local = select_support_indices(support_rows, support_y_np, support_size, "top_count", seed=stable_seed(SEED, "round246", cell, support_size))
            support_idx = support_all[local]
            qrows = query_item["rows"].iloc[qidx].reset_index(drop=True)
            overlap = drug_overlap_count(support_rows, qrows, local)
            if overlap:
                raise RuntimeError(f"Round246 support/query drug overlap for {cell}/{query_key}: {overlap}")
            audit = support_audit(support_rows, support_y_np, local, "top_count", support_size)
            audit.update(
                {
                    "cell_line": cell,
                    "split": query_key,
                    "evaluation_unit": unit,
                    "support_query_drug_overlap": overlap,
                    "baseline_adapter": "round246_cell_level_support_adapter",
                }
            )
            audit_rows.append(audit)
            support_q = tensor_slice(support_item, "q", support_idx, device)
            support_y = tensor_slice(support_item, "y", support_idx, device)
            support_w = tensor_slice(support_item, "weights", support_idx, device)
            q = tensor_slice(query_item, "q", qidx, device)
            delta = adapter(support_q, support_y, support_w)
            scores = score_with_cell_delta(model, q, delta, shift_scale).detach().cpu().numpy()
            summary, per = r208.compute_metrics(scores, query_item["y"][qidx], qrows, "Round247", variant, unit)
            summary.update(
                {
                    "cell_line": cell,
                    "adapt_support_size": int(len(support_idx)),
                    "requested_support_size": int(support_size),
                    "selector": "top_count",
                    "gate_mean": float("nan"),
                    "delta_norm": float(torch.linalg.vector_norm(delta).detach().cpu()),
                }
            )
            cell_rows.append(summary)
            if keep_per_unit:
                per["adapt_support_size"] = int(len(support_idx))
                per["requested_support_size"] = int(support_size)
                per["selector"] = "top_count"
                per["gate_mean"] = float("nan")
                per_parts.append(per)
        report_progress(len(query_item["by_cell"]), len(query_item["by_cell"]), label=f"{variant}: cells inspected")
    cell_df = pd.DataFrame(cell_rows)
    metric = macro_from_cells(cell_df, variant, query_key, unit, "Round246_cell_level_adapter")
    metric["selector"] = "top_count"
    metric["requested_support_size"] = int(support_size)
    return (
        pd.DataFrame([metric]),
        cell_df,
        pd.concat(per_parts, ignore_index=True) if per_parts else pd.DataFrame(),
        pd.DataFrame(audit_rows),
    )


def macro_from_cells(cell_df: pd.DataFrame, variant: str, split_name: str, unit: str, source: str) -> dict[str, Any]:
    out = {
        "source": source,
        "variant": variant,
        "split": split_name,
        "evaluation_family": "seen_668_no_adapt" if "no_adapt" in variant else "seen_668_support_adapt",
        "evaluation_unit": unit,
        "candidate_space": 668,
        "n_cells": int(cell_df["cell_line"].nunique()) if len(cell_df) else 0,
        "n_groups": int(cell_df["n_groups"].sum()) if "n_groups" in cell_df and len(cell_df) else 0,
        "n_drugs": int(cell_df["n_drugs"].sum()) if "n_drugs" in cell_df and len(cell_df) else 0,
        "n_profiles": int(cell_df["n_profiles"].sum()) if "n_profiles" in cell_df and len(cell_df) else 0,
    }
    for col in ["adapt_support_size", "requested_support_size", "gate_mean", "delta_norm"]:
        if col in cell_df:
            out[col + "_mean"] = float(cell_df[col].mean()) if len(cell_df) else float("nan")
    for metric in metric_cols():
        out[metric] = float(cell_df[metric].mean()) if metric in cell_df and len(cell_df) else float("nan")
    return out


def paired_rescue_loss(no_per: pd.DataFrame, adapt_per: pd.DataFrame, variant: str) -> pd.DataFrame:
    if no_per.empty or adapt_per.empty:
        return pd.DataFrame()
    key_cols = ["split", "evaluation_unit", "cell_line", "drug"]
    left = no_per.copy()
    right = adapt_per.copy()
    for df in [left, right]:
        df["_occ"] = df.groupby(key_cols).cumcount()
    key_cols2 = key_cols + ["_occ"]
    merged = left[key_cols2 + ["Hit@10", "Hit@50"]].merge(
        right[key_cols2 + ["Hit@10", "Hit@50"]],
        on=key_cols2,
        suffixes=("_no", "_adapt"),
    )
    rows = []
    for cell, g in merged.groupby("cell_line"):
        row = {"variant": variant, "cell_line": cell, "n_units": int(len(g))}
        for metric in ["Hit@10", "Hit@50"]:
            no = g[f"{metric}_no"].astype(float) > 0
            ad = g[f"{metric}_adapt"].astype(float) > 0
            rescued = int((~no & ad).sum())
            lost = int((no & ~ad).sum())
            row[f"{metric}_rescued"] = rescued
            row[f"{metric}_lost"] = lost
            row[f"{metric}_net_rescued"] = rescued - lost
            row[f"{metric}_win_rate"] = float((ad.astype(int) > no.astype(int)).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def add_sample_bins(cell_df: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    support_map = audit.set_index("cell_line")["support_groups"].astype(int).to_dict()
    out = cell_df.copy()
    out["support_groups_total"] = out["cell_line"].map(support_map).fillna(0).astype(int)
    def bin_one(v: int) -> str:
        if v <= 64:
            return "small_<=64_support_groups"
        if v <= 256:
            return "medium_65_256_support_groups"
        return "large_>256_support_groups"
    out["sample_bin"] = out["support_groups_total"].map(bin_one)
    return out


def summarize_by_bin(cell_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["variant", "split", "evaluation_unit", "sample_bin", "selector", "requested_support_size"]
    for keys, g in cell_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_cells"] = int(g["cell_line"].nunique())
        row["n_groups"] = int(g["n_groups"].sum()) if "n_groups" in g else 0
        row["n_profiles"] = int(g["n_profiles"].sum()) if "n_profiles" in g else 0
        for m in metric_cols():
            row[m] = float(g[m].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def run_eval(args: argparse.Namespace, device: torch.device) -> None:
    context = load_context(device)
    r243 = context["r243"]
    tables = context["tables"]
    audit = pd.read_csv(SOURCE / "cell_split_audit.tsv", sep="\t")
    adapter = None
    selected = None
    history = pd.DataFrame()
    if args.phase in {"train", "all", "smoke"}:
        adapter, selected, history = train_query_gated_adapter(context, device, smoke=(args.phase == "smoke" or args.smoke))
    else:
        selected_path = SOURCE / f"selected_{ADAPTER_CFG['variant']}.json"
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        ckpt = torch.load(selected["checkpoint"], map_location="cpu", weights_only=False)
        adapter = QueryGatedSupportAdapter(
            context["model"].target_anchor.detach(),
            hidden_dim=int(ckpt["cfg"]["hidden_dim"]),
            dropout=float(ckpt["cfg"]["dropout"]),
            max_delta_norm=float(ckpt["cfg"]["max_delta_norm"]),
        ).to(device)
        adapter.load_state_dict(ckpt["adapter_state_dict"], strict=True)
        adapter.eval()

    metrics = []
    cells = []
    per_units = []
    support_audits = []
    # no-adapt baseline
    for query_key, unit in [("official_test", "drug-mean"), ("official_test_profile", "true-profile-specific")]:
        m, c, p = evaluate_no_adapt(context, query_key, unit, f"round246_base_{query_key}_no_adapt", keep_per_unit=True)
        c["selector"] = "none"
        c["requested_support_size"] = 0
        metrics.append(m)
        cells.append(c)
        per_units.append(p)
    # Round246 old cell-level support adapter baseline
    for query_key, unit in [("official_test", "drug-mean"), ("official_test_profile", "true-profile-specific")]:
        m, c, p, a = evaluate_round246_cell_adapter(
            context,
            device,
            query_key,
            unit,
            f"round246_cell_adapter_{query_key}_support_adapt",
            keep_per_unit=True,
        )
        metrics.append(m)
        cells.append(c)
        per_units.append(p)
        support_audits.append(a)
    # query-gated curves
    eval_sizes = SUPPORT_SIZES if not args.smoke else [0, 32, -1]
    eval_selectors = SELECTORS if not args.smoke else ["top_count", "diverse_label"]
    for support_size in eval_sizes:
        if support_size == 0:
            continue
        for selector in eval_selectors:
            for query_key, unit in [("official_test", "drug-mean"), ("official_test_profile", "true-profile-specific")]:
                variant = f"{ADAPTER_CFG['variant']}_{query_key}_{selector}_s{support_size if support_size >= 0 else 'all'}"
                m, c, p, a = evaluate_query_gated(
                    context,
                    adapter,
                    device,
                    split_prefix="official",
                    support_key="official_support",
                    query_key=query_key,
                    unit=unit,
                    support_size=int(support_size),
                    selector=selector,
                    variant=variant,
                    keep_per_unit=True,
                )
                metrics.append(m)
                cells.append(c)
                per_units.append(p)
                support_audits.append(a)
    metric_df = pd.concat(metrics, ignore_index=True, sort=False)
    cell_df = pd.concat(cells, ignore_index=True, sort=False)
    per_df = pd.concat(per_units, ignore_index=True, sort=False)
    support_df = pd.concat(support_audits, ignore_index=True, sort=False) if support_audits else pd.DataFrame()
    cell_df = add_sample_bins(cell_df, audit)
    metric_df.to_csv(SOURCE / "metric_summary.tsv", sep="\t", index=False)
    cell_df.to_csv(SOURCE / "per_cell_metrics.tsv", sep="\t", index=False)
    per_df.to_csv(SOURCE / "per_unit_metrics.tsv", sep="\t", index=False)
    support_df.to_csv(SOURCE / "support_selection_audit.tsv", sep="\t", index=False)
    summarize_by_bin(cell_df).to_csv(SOURCE / "metric_summary_by_sample_bin.tsv", sep="\t", index=False)
    # paired rescue/loss for the protocol-relevant diverse/all and diverse/128 rows.
    no_drug = per_df[(per_df["variant"].eq("round246_base_official_test_no_adapt")) & (per_df["evaluation_unit"].eq("drug-mean"))]
    rescue_parts = []
    for variant in sorted(per_df["variant"].dropna().unique()):
        if "official_test_" not in variant or "official_test_profile" in variant:
            continue
        if "diverse_label_s128" not in variant and "diverse_label_sall" not in variant and "round246_cell_adapter" not in variant:
            continue
        ad = per_df[per_df["variant"].eq(variant)]
        rescue_parts.append(paired_rescue_loss(no_drug, ad, variant))
    if rescue_parts:
        pd.concat(rescue_parts, ignore_index=True, sort=False).to_csv(SOURCE / "paired_rescue_loss.tsv", sep="\t", index=False)
    write_method_lock(context, selected)
    write_audit(context, selected)
    write_report()


def write_method_lock(context: dict[str, Any], selected: dict[str, Any] | None) -> None:
    payload = {
        "round": "round247_support_curve_query_gated_adapter",
        "created_at": "2026-05-20",
        "seed": SEED,
        "scope": "Round246 frozen base plus query-conditioned gated support adapter; 668 seen-target space only",
        "round246_checkpoint": context["round246_selected"]["checkpoint"],
        "round246_checkpoint_sha256": context["round246_selected"]["checkpoint_sha256"],
        "kg_checkpoint": str(context["kg_path"]),
        "kg_checkpoint_sha256": context["kg_checkpoint_sha256"],
        "adapter_cfg": ADAPTER_CFG,
        "support_sizes": SUPPORT_SIZES,
        "support_selectors": SELECTORS,
        "data_audit": context["data_audit"],
        "selected_adapter": selected,
    }
    (ROUND_ROOT / "method_lock.json").write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")
    (SOURCE / "method_lock.json").write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


def write_audit(context: dict[str, Any], selected: dict[str, Any] | None) -> None:
    data_audit = context["data_audit"]
    lines = [
        "# Leakage and denominator audit",
        "",
        "- Scope: 668-label closed-set official unseen cell adaptation.",
        "- Base model is frozen Round246; KG and base predictor are not retrained.",
        "- Official unseen cells remain final evaluation cells.",
        "- Support/query drugs are disjoint by protocol.",
        "- Query-gated adapter uses only official_support for each target cell and never sees official_test labels during adaptation.",
        "- Round246 old cell-level support adapter is reported as a fixed baseline, not as the main Round247 method.",
        "",
        "## Leakage assertions",
        "",
    ]
    for key, value in data_audit["leakage_assertions"].items():
        lines.append(f"- `{key}`: `{value}`")
    if selected:
        lines.extend(["", "## Selected Adapter", "", f"- `{selected['variant']}`", f"- checkpoint: `{selected['checkpoint']}`"])
    (QA / "leakage_and_denominator_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report() -> None:
    metric_path = SOURCE / "metric_summary.tsv"
    if not metric_path.exists():
        return
    metrics = pd.read_csv(metric_path, sep="\t")
    bins = pd.read_csv(SOURCE / "metric_summary_by_sample_bin.tsv", sep="\t") if (SOURCE / "metric_summary_by_sample_bin.tsv").exists() else pd.DataFrame()
    main = metrics[
        (metrics["split"].eq("official_test"))
        & (metrics["evaluation_unit"].eq("drug-mean"))
        & (
            metrics["variant"].eq("round246_base_official_test_no_adapt")
            | metrics["variant"].eq("round246_cell_adapter_official_test_support_adapt")
            | metrics["variant"].str.contains("diverse_label_s128", na=False)
            | metrics["variant"].str.contains("diverse_label_sall", na=False)
        )
    ].copy()
    keep = ["variant", "selector", "requested_support_size", "n_cells", "n_groups", "n_profiles", "Hit@10", "MeanAP", "NDCG@10", "AUROC", "Hit@50"]
    main_text = main[[c for c in keep if c in main.columns]].to_string(index=False)
    bin_text = "No bin summary."
    if len(bins):
        show_bins = bins[
            (bins["evaluation_unit"].eq("drug-mean"))
            & (
                bins["variant"].eq("round246_base_official_test_no_adapt")
                | bins["variant"].eq("round246_cell_adapter_official_test_support_adapt")
                | bins["variant"].str.contains("diverse_label_s128", na=False)
                | bins["variant"].str.contains("diverse_label_sall", na=False)
            )
        ].copy()
        bin_text = show_bins[[c for c in ["variant", "sample_bin", "selector", "requested_support_size", "n_cells", "Hit@10", "MeanAP", "NDCG@10", "AUROC", "Hit@50"] if c in show_bins.columns]].to_string(index=False)
    text = f"""# Support-conditioned query adapter

## Scope

Round247 freezes the Round246 base predictor and trains only a query-conditioned gated support adapter. It also evaluates support-size and support-selector curves on the fixed official unseen-cell denominator.

## Main table

```text
{main_text}
```

## Sample-bin table

```text
{bin_text}
```

## Evaluation protocol

- The candidate set contains the 668 training targets.
- Official held-out cell lines are used for evaluation.
- Support/query drug split remains disjoint.
- Adaptation effects are measured relative to direct prediction on the same evaluation units.
"""
    (ROUND_ROOT / "training_summary.md").write_text(text, encoding="utf-8")
    (SOURCE / "training_summary.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["smoke", "train", "eval", "all"], default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    ensure_dirs()
    set_seed(SEED)
    args = parse_args()
    require_torch()
    device = torch.device(resolve_device(args.device))
    run_eval(args, device)


if __name__ == "__main__":
    main()
