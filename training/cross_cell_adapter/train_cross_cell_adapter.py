#!/usr/bin/env python3
"""Train the episodic query-gated adapter using held-out validation cell lines."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ModuleNotFoundError:
    torch = None
    F = None

    class _MissingNN:
        class Module:
            pass

    nn = _MissingNN()


ROUND_ROOT = Path(__file__).resolve().parents[1]
METAR_ROOT = ROUND_ROOT.parent
SOURCE = ROUND_ROOT / "source_data"
QA = ROUND_ROOT / "qa"
CHECKPOINTS = ROUND_ROOT / "checkpoints"
LOGS = ROUND_ROOT / "logs"

ROUND246_ROOT = METAR_ROOT / "claude_round246_joint_seen_target_train_pc3_a375_mcf7_20260519"
ROUND250_ROOT = METAR_ROOT / "claude_round250_epmeta_stable_tuning_20260520"
ROUND247_ROOT = METAR_ROOT / "claude_round247_support_curve_query_gated_adapter_20260520"
ROUND251_ROOT = METAR_ROOT / "claude_round251_epmeta_query_gated_adapter_20260520"
ROUND250_SCRIPT = ROUND250_ROOT / "scripts" / "run_round250_epmeta_stable_tuning.py"
ROUND247_SCRIPT = ROUND247_ROOT / "scripts" / "run_round247_support_curve_query_gated_adapter.py"

SEED = 20260520
FORCED_TRAIN_CELLS = ["PC3", "A375", "MCF7"]
SUPPORT_SIZES = [0, 8, 16, 32, 64, 128, -1]
SUPPORT_SELECTORS = ["top_count", "random", "diverse_label"]
METRIC_COLS = [f"Hit@{k}" for k in [1, 5, 10, 20, 30, 50]] + ["AUROC", "NDCG@10", "MeanAP"]

EPIQGATE_DEFAULT = {
    "variant": "r252_epiqgate_h128_sc0p15_mn2p5_lam1p0_s32_q32",
    "hidden_dim": 128,
    "dropout": 0.10,
    "max_delta_norm": 2.5,
    "shift_scale": 0.15,
    "lambda_episode": 1.0,
    "base_anchor_weight": 0.10,
    "delta_l2": 1.0e-3,
    "gate_l2": 1.0e-3,
    "support_size": 32,
    "query_size": 32,
    "tasks_per_epoch": 240,
    "max_epochs": 36,
    "min_epochs": 8,
    "patience": 8,
    "lr": 1.5e-4,
    "weight_decay": 1.0e-5,
    "train_selector": "diverse_label",
    "eval_selector": "diverse_label",
    "eval_support_size": 32,
}

HYBRID_DEFAULT = {
    "variant": "r252_qgate_mamlhybrid_query_bias_ilr0p0015_s32",
    "inner_kind": "query_bias",
    "inner_lr": 1.5e-3,
    "inner_steps": 1,
    "inner_parameter_policy": "q-precomputed hybrid updates only scorer-side parameters used after q-gated adaptation",
}


def require_torch() -> None:
    if torch is None or F is None:
        raise ModuleNotFoundError("torch is required for Round252")


def ensure_dirs() -> None:
    for path in [SOURCE, QA, CHECKPOINTS, LOGS]:
        path.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    print(f"[round252] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def stable_seed(*parts: Any) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:12], 16) % (2**31 - 1)


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


def load_round250_module():
    module = import_from_path("round251_round250", ROUND250_SCRIPT)
    module.ROUND_ROOT = ROUND_ROOT
    module.SOURCE = SOURCE
    module.QA = QA
    module.CHECKPOINTS = CHECKPOINTS
    module.LOGS = LOGS
    return module


def load_round247_module():
    module = import_from_path("round251_round247", ROUND247_SCRIPT)
    module.ROUND_ROOT = ROUND_ROOT
    module.SOURCE = SOURCE
    module.QA = QA
    module.CHECKPOINTS = CHECKPOINTS
    module.LOGS = LOGS
    return module


def fmt_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def support_size_value(size: int, available: int) -> int:
    if size < 0:
        return int(available)
    return int(min(max(size, 0), available))


def metric_macro_from_cells(cell_df: pd.DataFrame, variant: str, split_name: str, unit: str, source: str, family: str) -> dict[str, Any]:
    out = {
        "source": source,
        "variant": variant,
        "split": split_name,
        "evaluation_family": family,
        "evaluation_unit": unit,
        "candidate_space": 668,
        "n_cells": int(cell_df["cell_line"].nunique()) if len(cell_df) else 0,
        "n_groups": int(cell_df["n_groups"].sum()) if len(cell_df) else 0,
        "n_drugs": int(cell_df["n_drugs"].sum()) if len(cell_df) else 0,
        "n_profiles": int(cell_df["n_profiles"].sum()) if len(cell_df) else 0,
    }
    for extra in ["adapt_support_size", "used_support_rows", "inner_lr", "inner_steps"]:
        if extra in cell_df and len(cell_df):
            out[extra] = float(cell_df[extra].iloc[0]) if extra not in {"adapt_support_size", "inner_steps"} else int(cell_df[extra].iloc[0])
    for extra in ["gate_mean", "delta_norm"]:
        if extra in cell_df and len(cell_df):
            out[extra + "_mean"] = float(cell_df[extra].mean())
    for metric in METRIC_COLS:
        out[metric] = float(cell_df[metric].mean()) if len(cell_df) and metric in cell_df else float("nan")
    return out


def load_context(r250, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    r243, r211, r208, labels, kg_path, kg_ckpt, tables, data_audit = r250.load_context(args, device)
    r243.write_round243_audit_tables(tables, data_audit)
    support_ckpt = r250.resolve_current_checkpoint(args.support_baseline_variant, args.support_checkpoint)
    support_model, support_adapter, support_payload = r250.load_current_support_baseline(r243, r211, kg_ckpt, labels, support_ckpt, device)
    for key in [
        "adapter_train",
        "meta_train",
        "pseudo_support",
        "pseudo_val",
        "official_support",
        "official_test",
        "official_test_profile",
        "seen_support",
        "seen_test",
        "seen_test_profile",
    ]:
        r243.precompute_q(support_model, tables[key], device)
    data_audit["round252_policy"] = {
        "scope": "Round252 pseudo-validation sweep of Round247-style episodic query-gated meta-adapter in 668 seen-target closed-set cross-cell protocol",
        "candidate_space": 668,
        "kg_checkpoint": str(kg_path),
        "kg_checkpoint_sha256": sha256_file(kg_path),
        "kg_retrained": False,
        "round246_base_checkpoint": str(support_ckpt),
        "round246_base_checkpoint_sha256": sha256_file(support_ckpt),
        "forced_train_cells": FORCED_TRAIN_CELLS,
    }
    return {
        "r250": r250,
        "r243": r243,
        "r211": r211,
        "r208": r208,
        "labels": labels,
        "kg_path": kg_path,
        "kg_ckpt": kg_ckpt,
        "tables": tables,
        "data_audit": data_audit,
        "support_model": support_model,
        "support_adapter": support_adapter,
        "support_payload": support_payload,
        "support_checkpoint": support_ckpt,
    }


def resolve_round247_adapter_checkpoint(user_path: str | None = None) -> Path | None:
    if user_path:
        path = Path(user_path).expanduser()
        if path.exists():
            return path.resolve()
        raise FileNotFoundError(path)
    rel = Path("checkpoints") / "r247_query_gated_h128_sc0p15_mn2p5" / "adapter_checkpoint.pth"
    candidates = [
        ROUND247_ROOT / rel,
        Path("/NFS2_home/NFS2_home_1/lwn/claude/manuscript/write") / ROUND247_ROOT.name / rel,
        Path("/home/lwn/claude/manuscript/write") / ROUND247_ROOT.name / rel,
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def resolve_round251_adapter_checkpoint(user_path: str | None = None) -> Path | None:
    if user_path:
        path = Path(user_path).expanduser()
        if path.exists():
            return path.resolve()
        raise FileNotFoundError(path)
    rel = Path("checkpoints") / "r251_epiqgate_lm1_lr0p00015_s32_q32_seed20260520" / "adapter_checkpoint.pth"
    candidates = [
        ROUND251_ROOT / rel,
        Path("/NFS2_home/NFS2_home_1/lwn/claude/manuscript/write") / ROUND251_ROOT.name / rel,
        Path("/home/lwn/claude/manuscript/write") / ROUND251_ROOT.name / rel,
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def new_query_adapter(r247, model: nn.Module, cfg: dict[str, Any], device: torch.device) -> nn.Module:
    adapter = r247.QueryGatedSupportAdapter(
        model.target_anchor.detach(),
        hidden_dim=int(cfg["hidden_dim"]),
        dropout=float(cfg["dropout"]),
        max_delta_norm=float(cfg["max_delta_norm"]),
    ).to(device)
    return adapter


def maybe_init_from_round247(adapter: nn.Module, checkpoint: Path | None) -> str | None:
    if checkpoint is None:
        return None
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    adapter.load_state_dict(payload["adapter_state_dict"], strict=True)
    return sha256_file(checkpoint)


def maybe_init_from_checkpoint(adapter: nn.Module, checkpoint: Path | None) -> str | None:
    if checkpoint is None:
        return None
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    adapter.load_state_dict(payload["adapter_state_dict"], strict=True)
    return sha256_file(checkpoint)


def score_with_query_delta(r247, model: nn.Module, query_q: torch.Tensor, delta: torch.Tensor, gate: torch.Tensor, scale: float) -> torch.Tensor:
    return r247.score_with_query_delta(model, query_q, delta, gate, scale)


def select_support_indices(r247, rows: pd.DataFrame, y: np.ndarray, size: int, selector: str, seed: int) -> np.ndarray:
    return r247.select_support_indices(rows, y, int(size), str(selector), int(seed))


def support_audit(r247, rows: pd.DataFrame, y: np.ndarray, idx: np.ndarray, selector: str, requested_size: int) -> dict[str, Any]:
    return r247.support_audit(rows, y, idx, selector, requested_size)


def drug_overlap_count(r247, support_rows: pd.DataFrame, query_rows: pd.DataFrame, idx: np.ndarray) -> int:
    return int(r247.drug_overlap_count(support_rows, query_rows, idx))


def sample_episode(r247, item: dict, cell: str, support_size: int, query_size: int, selector: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    return r247.sample_episode(item, cell, support_size, query_size, selector, rng)


def eligible_cells(r247, item: dict, support_size: int, query_size: int) -> list[str]:
    return r247.eligible_cells(item, min_support=max(2, min(support_size, 8)), query_size=query_size)


def better_selection(candidate: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    if incumbent is None:
        return True
    for key in ["pseudo_val_Hit@10", "pseudo_val_MeanAP", "pseudo_val_NDCG@10", "pseudo_val_AUROC", "meta_val_noadapt_Hit@10"]:
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


def noadapt_summary(r250, r208, model: nn.Module, item: dict, device: torch.device, variant: str, unit: str) -> dict[str, Any]:
    return r250.noadapt_summary(r208, model, item, device, variant, unit)


def selection_record(context: dict[str, Any], adapter: nn.Module, r247, device: torch.device, cfg: dict[str, Any], epoch: int, stats: dict[str, float]) -> dict[str, Any]:
    metrics, _cells, _per, _audit = evaluate_epiqgate(
        context,
        r247,
        adapter,
        device,
        support_key="pseudo_support",
        query_key="pseudo_val",
        split_name="pseudo_val",
        unit="drug-mean",
        split_role="pseudo_unseen_validation",
        support_size=int(cfg["eval_support_size"]),
        selector=str(cfg["eval_selector"]),
        variant=f"{cfg['variant']}_pseudo_val_epiqgate_s{cfg['eval_support_size']}",
        shift_scale=float(cfg["shift_scale"]),
        keep_per_unit=False,
    )
    row = metrics.iloc[0].to_dict()
    meta = noadapt_summary(
        context["r250"],
        context["r208"],
        context["support_model"],
        context["tables"]["meta_train"],
        device,
        f"{cfg['variant']}_meta_train_no_adapt_anchor",
        "drug-mean",
    )
    return {
        "variant": cfg["variant"],
        "epoch": int(epoch),
        "train_loss": float(stats.get("train_loss", float("nan"))),
        "episode_loss": float(stats.get("episode_loss", float("nan"))),
        "base_anchor_loss": float(stats.get("base_anchor_loss", float("nan"))),
        "gate_mean": float(stats.get("gate_mean", float("nan"))),
        "delta_norm": float(stats.get("delta_norm", float("nan"))),
        "pseudo_val_Hit@10": float(row.get("Hit@10", float("nan"))),
        "pseudo_val_MeanAP": float(row.get("MeanAP", float("nan"))),
        "pseudo_val_NDCG@10": float(row.get("NDCG@10", float("nan"))),
        "pseudo_val_AUROC": float(row.get("AUROC", float("nan"))),
        "meta_val_noadapt_Hit@10": float(meta.get("Hit@10", float("nan"))),
        "meta_val_noadapt_MeanAP": float(meta.get("MeanAP", float("nan"))),
    }


def train_epiqgate(context: dict[str, Any], r247, args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, pd.DataFrame, dict[str, Any]]:
    cfg = deepcopy(EPIQGATE_DEFAULT)
    cfg["hidden_dim"] = int(args.adapter_hidden_dim)
    cfg["dropout"] = float(args.adapter_dropout)
    cfg["max_delta_norm"] = float(args.max_delta_norm)
    cfg["shift_scale"] = float(args.shift_scale)
    cfg["lambda_episode"] = float(args.lambda_episode)
    cfg["base_anchor_weight"] = float(args.base_anchor_weight)
    cfg["delta_l2"] = float(args.delta_l2)
    cfg["gate_l2"] = float(args.gate_l2)
    cfg["support_size"] = int(args.train_support_size)
    cfg["query_size"] = int(args.train_query_size)
    cfg["tasks_per_epoch"] = int(args.tasks_per_epoch)
    cfg["max_epochs"] = int(args.max_epochs)
    cfg["min_epochs"] = int(args.min_epochs)
    cfg["patience"] = int(args.patience)
    cfg["lr"] = float(args.adapter_lr)
    cfg["weight_decay"] = float(args.adapter_weight_decay)
    cfg["train_selector"] = str(args.train_selector)
    cfg["eval_selector"] = str(args.eval_selector)
    cfg["eval_support_size"] = int(args.eval_support_size)
    cfg["variant"] = (
        "r252_epiqgate"
        f"_lm{fmt_float(cfg['lambda_episode'])}"
        f"_lr{fmt_float(cfg['lr'])}"
        f"_sc{fmt_float(cfg['shift_scale'])}"
        f"_mn{fmt_float(cfg['max_delta_norm'])}"
        f"_ba{fmt_float(cfg['base_anchor_weight'])}"
        f"_s{cfg['support_size']}_q{cfg['query_size']}"
        f"_{cfg['train_selector']}"
        f"_seed{int(args.run_seed)}"
    )
    if args.smoke:
        cfg["variant"] += "_smoke"
        cfg["tasks_per_epoch"] = min(int(cfg["tasks_per_epoch"]), 4)
        cfg["max_epochs"] = 1
        cfg["min_epochs"] = 1
        cfg["patience"] = 1
    adapter = new_query_adapter(r247, context["support_model"], cfg, device)
    init_source = str(args.init_adapter)
    if init_source == "round251":
        init_ckpt = resolve_round251_adapter_checkpoint(args.round251_adapter_checkpoint)
        init_sha = maybe_init_from_checkpoint(adapter, init_ckpt)
    elif init_source == "none":
        init_ckpt = None
        init_sha = None
    else:
        init_ckpt = resolve_round247_adapter_checkpoint(args.round247_adapter_checkpoint)
        init_sha = maybe_init_from_round247(adapter, init_ckpt)
    opt = torch.optim.AdamW(adapter.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    train_item = context["tables"]["adapter_train"]
    cells = eligible_cells(r247, train_item, int(cfg["support_size"]), int(cfg["query_size"]))
    if not cells:
        raise RuntimeError("No eligible cells for EpiQGate training")
    rng = np.random.default_rng(int(args.run_seed))
    best = None
    best_state = None
    history: list[dict[str, Any]] = []
    stale = 0
    model = context["support_model"]
    r243 = context["r243"]
    for param in model.parameters():
        param.requires_grad_(False)
    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        adapter.train()
        losses = []
        episode_losses = []
        anchor_losses = []
        gates = []
        deltas = []
        for _ in range(int(cfg["tasks_per_epoch"])):
            cell = str(rng.choice(cells))
            support_idx, query_idx = sample_episode(
                r247,
                train_item,
                cell,
                int(cfg["support_size"]),
                int(cfg["query_size"]),
                str(cfg["train_selector"]),
                rng,
            )
            if len(support_idx) == 0 or len(query_idx) == 0:
                continue
            support_q = r247.tensor_slice(train_item, "q", support_idx, device)
            support_y = r247.tensor_slice(train_item, "y", support_idx, device)
            support_w = r247.tensor_slice(train_item, "weights", support_idx, device)
            query_q = r247.tensor_slice(train_item, "q", query_idx, device)
            query_y = r247.tensor_slice(train_item, "y", query_idx, device)
            query_w = r247.tensor_slice(train_item, "weights", query_idx, device)
            with torch.no_grad():
                base_logits = r243.score_from_q(model, query_q, delta=None, scale=0.0)
            delta, gate = adapter(query_q, support_q, support_y, support_w)
            logits = score_with_query_delta(r247, model, query_q, delta, gate, float(cfg["shift_scale"]))
            episode_loss = r247.retrieval_loss_from_logits(r243, logits, query_y, query_w)
            base_anchor_loss = torch.tensor(0.0, dtype=episode_loss.dtype, device=device)
            pos_mask = query_y > 0
            if pos_mask.any():
                base_pos = base_logits.masked_fill(~pos_mask, -1.0e9).max(dim=1).values
                adapt_pos = logits.masked_fill(~pos_mask, -1.0e9).max(dim=1).values
                base_anchor_loss = F.relu(base_pos - adapt_pos).mean()
            loss = (
                float(cfg["lambda_episode"]) * episode_loss
                + float(cfg["base_anchor_weight"]) * base_anchor_loss
                + float(cfg["delta_l2"]) * torch.square(delta).mean()
                + float(cfg["gate_l2"]) * torch.square(gate).mean()
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            episode_losses.append(float(episode_loss.detach().cpu()))
            anchor_losses.append(float(base_anchor_loss.detach().cpu()))
            gates.append(float(gate.mean().detach().cpu()))
            deltas.append(float(torch.linalg.vector_norm(delta, dim=1).mean().detach().cpu()))
        stats = {
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "episode_loss": float(np.mean(episode_losses)) if episode_losses else float("nan"),
            "base_anchor_loss": float(np.mean(anchor_losses)) if anchor_losses else float("nan"),
            "gate_mean": float(np.mean(gates)) if gates else float("nan"),
            "delta_norm": float(np.mean(deltas)) if deltas else float("nan"),
        }
        rec = selection_record(context, adapter, r247, device, cfg, epoch, stats)
        history.append(rec)
        if epoch == 1 or epoch % 5 == 0:
            log(
                f"{cfg['variant']} epoch={epoch} pseudo_hit10={rec['pseudo_val_Hit@10']:.4f} "
                f"loss={rec['train_loss']:.4f} gate={rec['gate_mean']:.3f}"
            )
        if better_selection(rec, best):
            best = rec
            best_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch >= int(cfg["min_epochs"]) and stale >= int(cfg["patience"]):
            log(f"{cfg['variant']} early stop at epoch={epoch}; best_epoch={best['epoch'] if best else 'NA'}")
            break
    if best is None or best_state is None:
        raise RuntimeError("No EpiQGate checkpoint selected")
    adapter.load_state_dict(best_state, strict=True)
    ckpt_dir = CHECKPOINTS / cfg["variant"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "adapter_checkpoint.pth"
    torch.save(
        {
            "stage": "round252_epiqgate_meta_adapter_sweep",
            "adapter_state_dict": best_state,
            "cfg": cfg,
            "selection": best,
            "round246_base_checkpoint": str(context["support_checkpoint"]),
            "init_adapter": init_source,
            "init_checkpoint": str(init_ckpt) if init_ckpt is not None else None,
            "init_sha256": init_sha,
        },
        ckpt_path,
    )
    selected = {
        "method_family": "epiqgate",
        "variant": cfg["variant"],
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "selection": best,
        "cfg": cfg,
        "init_adapter": init_source,
        "init_checkpoint": str(init_ckpt) if init_ckpt is not None else None,
        "init_sha256": init_sha,
        "round247_init_checkpoint": str(init_ckpt) if init_source == "round247" and init_ckpt is not None else None,
        "round247_init_sha256": init_sha if init_source == "round247" else None,
    }
    (SOURCE / f"selected_{cfg['variant']}.json").write_text(json.dumps(json_ready(selected), indent=2), encoding="utf-8")
    return adapter, pd.DataFrame(history), selected


def evaluate_epiqgate(
    context: dict[str, Any],
    r247,
    adapter: nn.Module,
    device: torch.device,
    support_key: str,
    query_key: str,
    split_name: str,
    unit: str,
    split_role: str,
    support_size: int,
    selector: str,
    variant: str,
    shift_scale: float,
    keep_per_unit: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    r208 = context["r208"]
    model = context["support_model"]
    support_item = context["tables"][support_key]
    query_item = context["tables"][query_key]
    adapter.eval()
    cell_rows = []
    per_parts = []
    audit_rows = []
    with torch.no_grad():
        for cell, qidx_raw in query_item["by_cell"].items():
            qidx = np.asarray(qidx_raw, dtype=np.int64)
            support_all = np.asarray(support_item["by_cell"].get(cell, np.asarray([], dtype=np.int64)), dtype=np.int64)
            if len(qidx) == 0 or len(support_all) == 0:
                continue
            support_rows = support_item["rows"].iloc[support_all].reset_index(drop=True)
            support_y_np = support_item["y"][support_all]
            local = select_support_indices(r247, support_rows, support_y_np, support_size, selector, stable_seed(SEED, "r251", cell, selector, support_size))
            support_idx = support_all[local]
            qrows = query_item["rows"].iloc[qidx].reset_index(drop=True)
            overlap = drug_overlap_count(r247, support_rows, qrows, local)
            if overlap:
                raise RuntimeError(f"Support/query drug overlap for {cell}/{query_key}: {overlap}")
            audit = support_audit(r247, support_rows, support_y_np, local, selector, support_size)
            audit.update(
                {
                    "method_family": "epiqgate",
                    "variant": variant,
                    "cell_line": cell,
                    "split_role": split_role,
                    "split": split_name,
                    "evaluation_unit": unit,
                    "support_query_drug_overlap": overlap,
                    "support_groups": int(len(support_all)),
                }
            )
            audit_rows.append(audit)
            if len(support_idx) == 0:
                continue
            support_q = r247.tensor_slice(support_item, "q", support_idx, device)
            support_y = r247.tensor_slice(support_item, "y", support_idx, device)
            support_w = r247.tensor_slice(support_item, "weights", support_idx, device)
            query_q = r247.tensor_slice(query_item, "q", qidx, device)
            delta, gate = adapter(query_q, support_q, support_y, support_w)
            scores = score_with_query_delta(r247, model, query_q, delta, gate, float(shift_scale)).detach().cpu().numpy()
            summary, per = r208.compute_metrics(scores, query_item["y"][qidx], qrows, "Round252_EpiQGate", variant, unit)
            summary.update(
                {
                    "cell_line": cell,
                    "adapt_support_size": int(support_size),
                    "used_support_rows": int(len(support_idx)),
                    "requested_support_size": int(support_size),
                    "selector": selector,
                    "gate_mean": float(gate.mean().detach().cpu()),
                    "delta_norm": float(torch.linalg.vector_norm(delta, dim=1).mean().detach().cpu()),
                }
            )
            cell_rows.append(summary)
            if keep_per_unit:
                per["method_family"] = "epiqgate"
                per["split_role"] = split_role
                per["variant"] = variant
                per["adapt_support_size"] = int(support_size)
                per["used_support_rows"] = int(len(support_idx))
                per["requested_support_size"] = int(support_size)
                per["selector"] = selector
                per["gate_mean"] = float(gate.mean().detach().cpu())
                per_parts.append(per)
    cells = context["r250"].add_cell_count_columns(pd.DataFrame(cell_rows), support_item)
    cells["method_family"] = "epiqgate"
    cells["split_role"] = split_role
    cells["variant_core"] = variant.rsplit("_", 1)[0] if "_s" in variant else variant
    cells["selector"] = selector
    metric = metric_macro_from_cells(cells, variant, split_name, unit, "Round252_EpiQGate", "seen_668_support_adapt")
    metric["method_family"] = "epiqgate"
    metric["split_role"] = split_role
    metric["adapt_support_size"] = int(support_size)
    metric["selector"] = selector
    return (
        pd.DataFrame([metric]),
        cells,
        pd.concat(per_parts, ignore_index=True, sort=False) if per_parts else pd.DataFrame(),
        pd.DataFrame(audit_rows),
    )


def pget(model: nn.Module, params: dict[str, torch.Tensor] | None, name: str) -> torch.Tensor:
    if params is not None and name in params:
        return params[name]
    current: Any = model
    for part in name.split("."):
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    return current


def functional_logits_from_q(model: nn.Module, query_q: torch.Tensor, params: dict[str, torch.Tensor] | None) -> torch.Tensor:
    target = F.normalize(model.target_anchor + pget(model, params, "target_delta"), dim=1)
    score = (query_q * pget(model, params, "diag")[None, :]) @ target.T
    score = score * torch.exp(torch.clamp(pget(model, params, "score_scale"), -2.0, 2.0))
    return score + pget(model, params, "target_bias")[None, :]


def final_query_layer_indices(model: nn.Module) -> tuple[int, int]:
    linear = [idx for idx, layer in enumerate(model.query_encoder) if isinstance(layer, nn.Linear)]
    norm = [idx for idx, layer in enumerate(model.query_encoder) if isinstance(layer, nn.LayerNorm)]
    if not linear or not norm:
        raise RuntimeError("Cannot infer query-tail layers")
    return linear[-1], norm[-1]


def inner_parameter_names(model: nn.Module, kind: str) -> list[str]:
    # Hybrid evaluation works from precomputed q. Query encoder tail parameters
    # are therefore not in the differentiable path; keep the inner loop on the
    # readout-side parameters that still affect q-gated scores.
    base = ["diag", "score_scale"]
    if kind == "query_tail":
        return base
    if kind == "query_bias":
        return base + ["target_bias"]
    raise ValueError(f"Unknown inner kind: {kind}")


def hybrid_adapt_params(
    r243,
    r247,
    model: nn.Module,
    adapter: nn.Module,
    support_item: dict,
    support_idx: np.ndarray,
    device: torch.device,
    inner_names: list[str],
    inner_lr: float,
    inner_steps: int,
) -> dict[str, torch.Tensor]:
    inner_set = set(inner_names)
    params = {
        name: (p.detach().clone().requires_grad_(True) if name in inner_set else p.detach())
        for name, p in model.named_parameters()
    }
    support_q = r247.tensor_slice(support_item, "q", support_idx, device)
    support_y = r247.tensor_slice(support_item, "y", support_idx, device)
    support_w = r247.tensor_slice(support_item, "weights", support_idx, device)
    adapter.eval()
    for _ in range(int(inner_steps)):
        with torch.no_grad():
            delta, gate = adapter(support_q, support_q, support_y, support_w)
            adapted_q = F.normalize(support_q + float(EPIQGATE_DEFAULT["shift_scale"]) * gate[:, None] * delta, dim=1)
        logits = functional_logits_from_q(model, adapted_q, params)
        loss = r247.retrieval_loss_from_logits(r243, logits, support_y, support_w)
        grads = torch.autograd.grad(loss, [params[name] for name in inner_names], create_graph=False, allow_unused=False)
        updated = dict(params)
        for name, grad in zip(inner_names, grads):
            updated[name] = params[name] - float(inner_lr) * grad
        params = updated
    return params


def evaluate_hybrid(
    context: dict[str, Any],
    r247,
    adapter: nn.Module,
    device: torch.device,
    support_key: str,
    query_key: str,
    split_name: str,
    unit: str,
    split_role: str,
    support_size: int,
    selector: str,
    cfg: dict[str, Any],
    keep_per_unit: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    r208 = context["r208"]
    r243 = context["r243"]
    model = context["support_model"]
    support_item = context["tables"][support_key]
    query_item = context["tables"][query_key]
    inner_names = inner_parameter_names(model, str(cfg["inner_kind"]))
    variant = f"{cfg['variant']}_{split_role}_{selector}_s{support_size}"
    cell_rows = []
    per_parts = []
    audit_rows = []
    adapter.eval()
    model.eval()
    for cell, qidx_raw in query_item["by_cell"].items():
        qidx = np.asarray(qidx_raw, dtype=np.int64)
        support_all = np.asarray(support_item["by_cell"].get(cell, np.asarray([], dtype=np.int64)), dtype=np.int64)
        if len(qidx) == 0 or len(support_all) == 0:
            continue
        support_rows = support_item["rows"].iloc[support_all].reset_index(drop=True)
        support_y_np = support_item["y"][support_all]
        local = select_support_indices(r247, support_rows, support_y_np, support_size, selector, stable_seed(SEED, "hybrid", cell, selector, support_size))
        support_idx = support_all[local]
        qrows = query_item["rows"].iloc[qidx].reset_index(drop=True)
        overlap = drug_overlap_count(r247, support_rows, qrows, local)
        if overlap:
            raise RuntimeError(f"Hybrid support/query drug overlap for {cell}/{query_key}: {overlap}")
        audit = support_audit(r247, support_rows, support_y_np, local, selector, support_size)
        audit.update(
            {
                "method_family": "qgate_maml_hybrid",
                "variant": variant,
                "cell_line": cell,
                "split_role": split_role,
                "split": split_name,
                "evaluation_unit": unit,
                "support_query_drug_overlap": overlap,
                "support_groups": int(len(support_all)),
            }
        )
        audit_rows.append(audit)
        if len(support_idx) == 0:
            continue
        params = hybrid_adapt_params(
            r243,
            r247,
            model,
            adapter,
            support_item,
            support_idx,
            device,
            inner_names,
            float(cfg["inner_lr"]),
            int(cfg["inner_steps"]),
        )
        query_q = r247.tensor_slice(query_item, "q", qidx, device)
        support_q = r247.tensor_slice(support_item, "q", support_idx, device)
        support_y = r247.tensor_slice(support_item, "y", support_idx, device)
        support_w = r247.tensor_slice(support_item, "weights", support_idx, device)
        with torch.no_grad():
            delta, gate = adapter(query_q, support_q, support_y, support_w)
            adapted_q = F.normalize(query_q + float(EPIQGATE_DEFAULT["shift_scale"]) * gate[:, None] * delta, dim=1)
            scores = functional_logits_from_q(model, adapted_q, params).detach().cpu().numpy()
        summary, per = r208.compute_metrics(scores, query_item["y"][qidx], qrows, "Round251_QGateMAMLHybrid", variant, unit)
        summary.update(
            {
                "cell_line": cell,
                "adapt_support_size": int(support_size),
                "used_support_rows": int(len(support_idx)),
                "requested_support_size": int(support_size),
                "selector": selector,
                "inner_lr": float(cfg["inner_lr"]),
                "inner_steps": int(cfg["inner_steps"]),
                "gate_mean": float(gate.mean().detach().cpu()),
                "delta_norm": float(torch.linalg.vector_norm(delta, dim=1).mean().detach().cpu()),
            }
        )
        cell_rows.append(summary)
        if keep_per_unit:
            per["method_family"] = "qgate_maml_hybrid"
            per["split_role"] = split_role
            per["variant"] = variant
            per["adapt_support_size"] = int(support_size)
            per["used_support_rows"] = int(len(support_idx))
            per["requested_support_size"] = int(support_size)
            per["selector"] = selector
            per_parts.append(per)
    cells = context["r250"].add_cell_count_columns(pd.DataFrame(cell_rows), support_item)
    cells["method_family"] = "qgate_maml_hybrid"
    cells["split_role"] = split_role
    cells["variant_core"] = cfg["variant"]
    cells["selector"] = selector
    metric = metric_macro_from_cells(cells, variant, split_name, unit, "Round251_QGateMAMLHybrid", "seen_668_maml_hybrid")
    metric["method_family"] = "qgate_maml_hybrid"
    metric["split_role"] = split_role
    metric["adapt_support_size"] = int(support_size)
    metric["selector"] = selector
    return (
        pd.DataFrame([metric]),
        cells,
        pd.concat(per_parts, ignore_index=True, sort=False) if per_parts else pd.DataFrame(),
        pd.DataFrame(audit_rows),
    )


def evaluate_epiqgate_all(
    context: dict[str, Any],
    r247,
    adapter: nn.Module,
    device: torch.device,
    selected: dict[str, Any],
    support_sizes: Iterable[int],
    selectors: Iterable[str],
    keep_per_unit: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = []
    cells_all = []
    per_all = []
    audits = []
    splits = [
        ("pseudo_support", "pseudo_val", "pseudo_val", "drug-mean", "pseudo_unseen_validation"),
        ("official_support", "official_test", "official_test", "drug-mean", "official_unseen"),
        ("official_support", "official_test_profile", "official_test_profile", "true-profile-specific", "official_unseen"),
        ("seen_support", "seen_test", "seen_test", "drug-mean", "seen_diagnostic"),
        ("seen_support", "seen_test_profile", "seen_test_profile", "true-profile-specific", "seen_diagnostic"),
    ]
    for support_size in support_sizes:
        if int(support_size) <= 0:
            continue
        size_tag = "all" if int(support_size) < 0 else str(int(support_size))
        for selector in selectors:
            for support_key, query_key, split_name, unit, split_role in splits:
                variant = f"{selected['variant']}_{split_role}_{selector}_s{size_tag}"
                m, c, p, a = evaluate_epiqgate(
                    context,
                    r247,
                    adapter,
                    device,
                    support_key,
                    query_key,
                    split_name,
                    unit,
                    split_role,
                    int(support_size),
                    str(selector),
                    variant,
                    float(selected.get("cfg", {}).get("shift_scale", EPIQGATE_DEFAULT["shift_scale"])),
                    keep_per_unit and split_role == "official_unseen",
                )
                metrics.append(m)
                cells_all.append(c)
                if len(p):
                    per_all.append(p)
                if len(a):
                    audits.append(a)
    return (
        pd.concat(metrics, ignore_index=True, sort=False) if metrics else pd.DataFrame(),
        pd.concat(cells_all, ignore_index=True, sort=False) if cells_all else pd.DataFrame(),
        pd.concat(per_all, ignore_index=True, sort=False) if per_all else pd.DataFrame(),
        pd.concat(audits, ignore_index=True, sort=False) if audits else pd.DataFrame(),
    )


def evaluate_hybrid_all(
    context: dict[str, Any],
    r247,
    adapter: nn.Module,
    device: torch.device,
    support_sizes: Iterable[int],
    selectors: Iterable[str],
    keep_per_unit: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = deepcopy(HYBRID_DEFAULT)
    metrics = []
    cells_all = []
    per_all = []
    audits = []
    splits = [
        ("pseudo_support", "pseudo_val", "pseudo_val", "drug-mean", "pseudo_unseen_validation"),
        ("official_support", "official_test", "official_test", "drug-mean", "official_unseen"),
        ("official_support", "official_test_profile", "official_test_profile", "true-profile-specific", "official_unseen"),
        ("seen_support", "seen_test", "seen_test", "drug-mean", "seen_diagnostic"),
        ("seen_support", "seen_test_profile", "seen_test_profile", "true-profile-specific", "seen_diagnostic"),
    ]
    for support_size in support_sizes:
        if int(support_size) <= 0:
            continue
        for selector in selectors:
            for support_key, query_key, split_name, unit, split_role in splits:
                m, c, p, a = evaluate_hybrid(
                    context,
                    r247,
                    adapter,
                    device,
                    support_key,
                    query_key,
                    split_name,
                    unit,
                    split_role,
                    int(support_size),
                    str(selector),
                    cfg,
                    keep_per_unit and split_role == "official_unseen",
                )
                metrics.append(m)
                cells_all.append(c)
                if len(p):
                    per_all.append(p)
                if len(a):
                    audits.append(a)
    return (
        pd.concat(metrics, ignore_index=True, sort=False) if metrics else pd.DataFrame(),
        pd.concat(cells_all, ignore_index=True, sort=False) if cells_all else pd.DataFrame(),
        pd.concat(per_all, ignore_index=True, sort=False) if per_all else pd.DataFrame(),
        pd.concat(audits, ignore_index=True, sort=False) if audits else pd.DataFrame(),
    )


def paired_rescue_loss(per_df: pd.DataFrame) -> pd.DataFrame:
    if per_df.empty:
        return pd.DataFrame()
    base = per_df[(per_df["method_family"].eq("no_adapt")) & (per_df["split_role"].eq("official_unseen"))].copy()
    if base.empty:
        return pd.DataFrame()
    rows = []
    key_cols = ["split", "evaluation_unit", "cell_line", "drug"]
    base["_occ"] = base.groupby(key_cols).cumcount()
    base_keys = key_cols + ["_occ"]
    for variant, meth in per_df[per_df["method_family"].ne("no_adapt")].groupby("variant"):
        if "official_unseen" not in set(meth["split_role"].astype(str)):
            continue
        right = meth.copy()
        right["_occ"] = right.groupby(key_cols).cumcount()
        merged = base[base_keys + ["Hit@10", "Hit@50"]].merge(
            right[base_keys + ["Hit@10", "Hit@50", "method_family", "adapt_support_size", "selector"]],
            on=base_keys,
            suffixes=("_base", "_method"),
            how="inner",
        )
        for cell, grp in merged.groupby("cell_line"):
            rec = {
                "variant": variant,
                "method_family": str(grp["method_family"].iloc[0]),
                "cell_line": cell,
                "adapt_support_size": int(grp["adapt_support_size"].iloc[0]),
                "selector": str(grp["selector"].iloc[0]),
                "n_units": int(len(grp)),
            }
            for metric in ["Hit@10", "Hit@50"]:
                no = grp[f"{metric}_base"].astype(float) > 0
                ad = grp[f"{metric}_method"].astype(float) > 0
                rec[f"{metric}_rescued"] = int((~no & ad).sum())
                rec[f"{metric}_lost"] = int((no & ~ad).sum())
                rec[f"{metric}_net_rescued"] = rec[f"{metric}_rescued"] - rec[f"{metric}_lost"]
                rec[f"{metric}_win_rate"] = float((ad.astype(int) > no.astype(int)).mean())
            rows.append(rec)
    return pd.DataFrame(rows)


def paired_delta_vs_noadapt(cell_metrics: pd.DataFrame) -> pd.DataFrame:
    if cell_metrics.empty:
        return pd.DataFrame()
    base = cell_metrics[cell_metrics["method_family"].eq("no_adapt")].copy()
    meth = cell_metrics[cell_metrics["method_family"].ne("no_adapt")].copy()
    keys = ["split_role", "split", "evaluation_unit", "cell_line"]
    merged = meth.merge(
        base[keys + METRIC_COLS],
        on=keys,
        suffixes=("_method", "_baseline"),
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()
    rows = []
    group_cols = [
        "split_role",
        "split",
        "evaluation_unit",
        "method_family",
        "variant",
        "selector",
        "adapt_support_size",
        "sample_bin",
    ]
    for group_keys, grp in merged.groupby(group_cols, dropna=False):
        rec = dict(zip(group_cols, group_keys))
        rec["baseline_family"] = "no_adapt"
        rec["n_cells"] = int(grp["cell_line"].nunique())
        for metric in METRIC_COLS:
            delta = grp[f"{metric}_method"] - grp[f"{metric}_baseline"]
            rec[f"{metric}_mean_delta"] = float(delta.mean())
            rec[f"{metric}_win_cells"] = int((delta > 0).sum())
            rec[f"{metric}_tie_or_win_cells"] = int((delta >= 0).sum())
            rec[f"{metric}_loss_cells"] = int((delta < 0).sum())
        rows.append(rec)
    return pd.DataFrame(rows)


def write_audit(data_audit: dict[str, Any]) -> None:
    lines = [
        "# Round252 leakage and denominator audit",
        "",
        "- Scope: 668-label closed-set Round247-style episodic query-gated meta-learning.",
        "- Round240 full-space target-side is out of scope.",
        "- KG checkpoint is reused; KG is not retrained.",
        "- Round246 whole-model base is reused.",
        "- Official unseen cells are final evaluation only.",
        "- Pseudo-unseen cells are used for model selection only.",
        "- Support/query drugs must be disjoint.",
        "",
        "## Leakage assertions",
        "",
    ]
    for key, value in data_audit.get("leakage_assertions", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Pre-registered metric bins",
            "",
            "- `small_<=64_support_groups`: Hit@10 and MeanAP.",
            "- `medium_65_256_support_groups`: Hit@10 primary; NDCG@10/AUROC secondary.",
            "- `large_>256_support_groups`: MeanAP/NDCG@10/AUROC and paired rescue/loss; Hit@10 is diagnostic.",
        ]
    )
    (QA / "leakage_and_denominator_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (SOURCE / "leakage_and_denominator_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_method_lock(payload: dict[str, Any]) -> None:
    base = {
        "round": "round252_epiqgate_pseudoval_sweep",
        "created_at": "2026-05-20",
        "seed": SEED,
        "scope": "Round247-style episodic query-gated meta-adapter plus conservative QGate-MAML hybrid; 668 seen-target closed-set only",
        "support_sizes": SUPPORT_SIZES,
        "support_selectors": SUPPORT_SELECTORS,
        "selection": "pseudo-unseen EpiQGate support-adapt Hit@10, then MeanAP, NDCG@10, AUROC, and meta-train no-adapt Hit@10 anchor",
        "sample_bins": {
            "small_<=64_support_groups": "support groups <= 64; Hit@10 and MeanAP",
            "medium_65_256_support_groups": "65 <= support groups <= 256; Hit@10 primary",
            "large_>256_support_groups": "support groups > 256; MeanAP/NDCG@10/AUROC and rescue/loss emphasized",
        },
        "artifacts": payload,
    }
    text = json.dumps(json_ready(base), indent=2)
    (ROUND_ROOT / "method_lock.json").write_text(text, encoding="utf-8")
    (SOURCE / "method_lock.json").write_text(text, encoding="utf-8")


def write_report(metrics: pd.DataFrame, by_bin: pd.DataFrame, deltas: pd.DataFrame, rescue: pd.DataFrame) -> None:
    official = metrics[
        (metrics["split"].eq("official_test"))
        & (metrics["evaluation_unit"].eq("drug-mean"))
        & (metrics["adapt_support_size"].isin([0, 32, 64, 128, -1]))
    ].copy()
    official_text = official[
        [
            "source",
            "variant",
            "method_family",
            "selector",
            "evaluation_family",
            "adapt_support_size",
            "n_cells",
            "Hit@10",
            "MeanAP",
            "NDCG@10",
            "AUROC",
            "Hit@20",
            "Hit@50",
        ]
    ].to_csv(sep="\t", index=False).strip() if len(official) else "No official metrics."
    bin_text = by_bin[
        (by_bin["split_role"].eq("official_unseen"))
        & (by_bin["evaluation_unit"].eq("drug-mean"))
        & (by_bin["adapt_support_size"].isin([0, 32, 64, 128, -1]))
    ].to_csv(sep="\t", index=False).strip() if len(by_bin) else "No sample-bin metrics."
    delta_text = deltas[
        (deltas["split_role"].eq("official_unseen"))
        & (deltas["evaluation_unit"].eq("drug-mean"))
    ].to_csv(sep="\t", index=False).strip() if len(deltas) else "No paired deltas."
    rescue_text = rescue.to_csv(sep="\t", index=False).strip() if len(rescue) else "No rescue/loss metrics."
    lines = [
        "# Round252 optimized Round247-style episodic query-gated meta-learning",
        "",
        "## Main official unseen metrics",
        "",
        "```tsv",
        official_text,
        "```",
        "",
        "## Official unseen metrics by support-count bin",
        "",
        "```tsv",
        bin_text,
        "```",
        "",
        "## Paired cell-level deltas",
        "",
        "```tsv",
        delta_text,
        "```",
        "",
        "## Paired rescue/loss",
        "",
        "```tsv",
        rescue_text,
        "```",
        "",
        "## Interpretation guardrails",
        "",
        "- Official unseen cells are final evidence only.",
        "- Main headline remains drug-mean; profile-level is diagnostic.",
        "- Support-bin metrics are pre-registered and must be compared against no-adapt on the same denominator.",
        "- QGate-MAML hybrid is a comparison branch; EpiQGate is the primary Round247-style meta-learning branch.",
    ]
    (ROUND_ROOT / "round252_result_report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (SOURCE / "round252_result_report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_round252_optimization_report(summary: pd.DataFrame, metrics: pd.DataFrame, by_bin: pd.DataFrame, deltas: pd.DataFrame) -> None:
    lines = [
        "# Round252 EpiQGate pseudo-unseen sweep report",
        "",
        "本轮只优化 Round251 EpiQGate 主方法。配置选择只使用 pseudo-unseen validation；official unseen 只在选择后作为最终评估。",
        "",
        "## Pseudo-unseen sweep ranking",
        "",
        "```tsv",
    ]
    rank_cols = [
        "sweep_name",
        "variant",
        "init_adapter",
        "epoch",
        "pseudo_val_Hit@10",
        "pseudo_val_MeanAP",
        "pseudo_val_NDCG@10",
        "pseudo_val_AUROC",
        "cfg_shift_scale",
        "cfg_max_delta_norm",
        "cfg_base_anchor_weight",
        "cfg_support_size",
        "cfg_query_size",
        "cfg_lr",
    ]
    available = [c for c in rank_cols if c in summary.columns]
    ranked = summary.sort_values(["pseudo_val_Hit@10", "pseudo_val_MeanAP", "pseudo_val_NDCG@10"], ascending=False)
    lines.append(ranked[available].to_csv(sep="\t", index=False).strip())
    lines.extend(["```", "", "## Selected official unseen metric", "", "```tsv"])
    official = metrics[(metrics["split"].eq("official_test")) & (metrics["evaluation_unit"].eq("drug-mean"))]
    show_cols = [
        "source",
        "variant",
        "method_family",
        "selector",
        "adapt_support_size",
        "n_cells",
        "Hit@10",
        "MeanAP",
        "NDCG@10",
        "AUROC",
        "Hit@20",
        "Hit@50",
    ]
    lines.append(official[[c for c in show_cols if c in official.columns]].to_csv(sep="\t", index=False).strip())
    lines.extend(["```", "", "## Official unseen by sample bin", "", "```tsv"])
    bin_rows = by_bin[(by_bin["split"].eq("official_test")) & (by_bin["evaluation_unit"].eq("drug-mean"))]
    bin_cols = [
        "method_family",
        "variant_core",
        "adapt_support_size",
        "sample_bin",
        "n_cells",
        "n_groups",
        "Hit@10",
        "MeanAP",
        "NDCG@10",
        "AUROC",
        "Hit@50",
    ]
    lines.append(bin_rows[[c for c in bin_cols if c in bin_rows.columns]].to_csv(sep="\t", index=False).strip())
    lines.extend(["```", "", "## Paired delta vs no-adapt", "", "```tsv"])
    delta_rows = deltas[(deltas["split"].eq("official_test")) & (deltas["evaluation_unit"].eq("drug-mean"))]
    delta_cols = [
        "method_family",
        "variant",
        "selector",
        "adapt_support_size",
        "sample_bin",
        "n_cells",
        "Hit@10_mean_delta",
        "Hit@10_win_cells",
        "Hit@10_tie_or_win_cells",
        "Hit@10_loss_cells",
        "MeanAP_mean_delta",
        "NDCG@10_mean_delta",
        "AUROC_mean_delta",
    ]
    lines.append(delta_rows[[c for c in delta_cols if c in delta_rows.columns]].to_csv(sep="\t", index=False).strip())
    lines.append("```")
    text = "\n".join(lines) + "\n"
    (ROUND_ROOT / "round252_optimization_report_zh.md").write_text(text, encoding="utf-8")
    (SOURCE / "round252_optimization_report_zh.md").write_text(text, encoding="utf-8")


def run_training(args: argparse.Namespace, device: torch.device) -> None:
    r250 = load_round250_module()
    r247 = load_round247_module()
    context = load_context(r250, args, device)
    write_audit(context["data_audit"])
    (SOURCE / "cell_splits.json").write_text(json.dumps(json_ready(context["data_audit"]), indent=2), encoding="utf-8")

    adapter, history, selected = train_epiqgate(context, r247, args, device)
    history.to_csv(SOURCE / "training_history.tsv", sep="\t", index=False)

    report_sizes = [int(x) for x in args.report_support_sizes]
    selectors = list(args.report_selectors)
    all_metrics = []
    all_cells = []
    all_per = []
    all_audits = []

    support_metrics, support_cells, support_per = r250.evaluate_support_baseline(
        context["r243"],
        context["r208"],
        context["support_model"],
        context["support_adapter"],
        context["tables"],
        device,
        str(context["support_payload"]["joint_cfg"]["variant"]),
        float(context["support_payload"]["joint_cfg"]["shift_scale"]),
        report_sizes,
        keep_per_unit=True,
    )
    all_metrics.append(support_metrics)
    all_cells.append(support_cells)
    if len(support_per):
        all_per.append(support_per)

    r247_ckpt = resolve_round247_adapter_checkpoint(args.round247_adapter_checkpoint)
    if r247_ckpt is not None:
        qgate_adapter, qgate_payload = r250.load_query_gated_baseline(r247, context["support_model"], r247_ckpt, device)
        qgate_metrics, qgate_cells, qgate_per = r250.evaluate_query_gated_baseline(
            r247,
            {"r247": r247, "r243": context["r243"], "r208": context["r208"], "model": context["support_model"], "tables": context["tables"]},
            qgate_adapter,
            device,
            support_sizes=report_sizes,
            selectors=selectors,
            keep_per_unit=True,
        )
        all_metrics.append(qgate_metrics)
        all_cells.append(qgate_cells)
        if len(qgate_per):
            all_per.append(qgate_per)

    epi_metrics, epi_cells, epi_per, epi_audit = evaluate_epiqgate_all(
        context,
        r247,
        adapter,
        device,
        selected,
        report_sizes,
        selectors,
        keep_per_unit=True,
    )
    all_metrics.append(epi_metrics)
    all_cells.append(epi_cells)
    if len(epi_per):
        all_per.append(epi_per)
    if len(epi_audit):
        all_audits.append(epi_audit)

    if not args.skip_hybrid:
        hybrid_sizes = [x for x in report_sizes if x in {32, 64}]
        hybrid_metrics, hybrid_cells, hybrid_per, hybrid_audit = evaluate_hybrid_all(
            context,
            r247,
            adapter,
            device,
            hybrid_sizes,
            selectors=["diverse_label"],
            keep_per_unit=True,
        )
        all_metrics.append(hybrid_metrics)
        all_cells.append(hybrid_cells)
        if len(hybrid_per):
            all_per.append(hybrid_per)
        if len(hybrid_audit):
            all_audits.append(hybrid_audit)

    metrics_all = pd.concat(all_metrics, ignore_index=True, sort=False)
    cells_all = pd.concat(all_cells, ignore_index=True, sort=False)
    per_all = pd.concat(all_per, ignore_index=True, sort=False) if all_per else pd.DataFrame()
    audit_all = pd.concat(all_audits, ignore_index=True, sort=False) if all_audits else pd.DataFrame()
    by_bin = r250.summarize_by_sample_bin(cells_all)
    deltas = paired_delta_vs_noadapt(cells_all)
    rescue = paired_rescue_loss(per_all)

    metrics_all.to_csv(SOURCE / "metric_summary.tsv", sep="\t", index=False)
    cells_all.to_csv(SOURCE / "per_cell_metrics.tsv", sep="\t", index=False)
    by_bin.to_csv(SOURCE / "metric_summary_by_sample_bin.tsv", sep="\t", index=False)
    deltas.to_csv(SOURCE / "paired_delta_by_sample_bin.tsv", sep="\t", index=False)
    if len(per_all):
        per_all.to_csv(SOURCE / "per_unit_metrics.tsv", sep="\t", index=False)
    if len(audit_all):
        audit_all.to_csv(SOURCE / "support_selection_audit.tsv", sep="\t", index=False)
    r247_ckpt = resolve_round247_adapter_checkpoint(args.round247_adapter_checkpoint)
    write_method_lock(
        {
            "kg_checkpoint": str(context["kg_path"]),
            "kg_checkpoint_sha256": sha256_file(context["kg_path"]),
            "data_audit": context["data_audit"],
            "selected_epiqgate": selected,
            "round246_checkpoint": str(context["support_checkpoint"]),
            "round246_checkpoint_sha256": sha256_file(context["support_checkpoint"]),
            "round247_adapter_checkpoint": str(r247_ckpt) if r247_ckpt is not None else None,
            "round247_adapter_sha256": sha256_file(r247_ckpt) if r247_ckpt is not None else None,
            "hybrid_cfg": None if args.skip_hybrid else HYBRID_DEFAULT,
        }
    )
    write_report(metrics_all, by_bin, deltas, rescue)
    log(f"done; outputs written to {ROUND_ROOT}")


def run_audit(args: argparse.Namespace, device: torch.device) -> None:
    r250 = load_round250_module()
    context = load_context(r250, args, device)
    write_audit(context["data_audit"])
    (SOURCE / "cell_splits.json").write_text(json.dumps(json_ready(context["data_audit"]), indent=2), encoding="utf-8")
    write_method_lock({"data_audit": context["data_audit"], "status": "audit_only"})


def apply_sweep_config(args: argparse.Namespace, cfg: dict[str, Any]) -> argparse.Namespace:
    child = argparse.Namespace(**vars(args))
    for key, value in cfg.items():
        setattr(child, key, value)
    return child


def sweep_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    # Deliberately small and conservative: all choices are selected only by
    # pseudo-unseen validation, not by official unseen performance.
    base = {
        "init_adapter": "round247",
        "adapter_hidden_dim": 128,
        "adapter_dropout": 0.10,
        "lambda_episode": 1.0,
        "adapter_lr": 1.5e-4,
        "adapter_weight_decay": 1.0e-5,
        "max_delta_norm": 2.5,
        "shift_scale": 0.15,
        "base_anchor_weight": 0.10,
        "delta_l2": 1.0e-3,
        "gate_l2": 1.0e-3,
        "train_support_size": 32,
        "train_query_size": 32,
        "eval_support_size": 64,
        "train_selector": "diverse_label",
        "eval_selector": "diverse_label",
        "tasks_per_epoch": int(args.tasks_per_epoch),
        "max_epochs": int(args.max_epochs),
        "min_epochs": int(args.min_epochs),
        "patience": int(args.patience),
        "skip_hybrid": True,
    }
    variants = [
        ("base_eval64", {}),
        ("lr1e-4", {"adapter_lr": 1.0e-4}),
        ("lr8e-5", {"adapter_lr": 8.0e-5}),
        ("shift012", {"shift_scale": 0.12}),
        ("shift018", {"shift_scale": 0.18}),
        ("anchor02", {"base_anchor_weight": 0.20}),
        ("anchor005", {"base_anchor_weight": 0.05}),
        ("s64", {"train_support_size": 64, "eval_support_size": 64}),
        ("s128", {"train_support_size": 128, "eval_support_size": 128}),
        ("q48", {"train_query_size": 48}),
        ("drop005", {"adapter_dropout": 0.05}),
        ("init251_lr5e-5", {"init_adapter": "round251", "adapter_lr": 5.0e-5, "max_epochs": min(int(args.max_epochs), 18), "patience": min(int(args.patience), 5)}),
    ]
    out = []
    for idx, (name, patch) in enumerate(variants):
        rec = dict(base)
        rec.update(patch)
        rec["sweep_name"] = name
        rec["run_seed"] = int(args.run_seed) + idx
        out.append(rec)
    return out


def selected_to_row(selected: dict[str, Any]) -> dict[str, Any]:
    rec = dict(selected.get("selection", {}))
    cfg = selected.get("cfg", {})
    rec.update(
        {
            "variant": selected.get("variant"),
            "checkpoint": selected.get("checkpoint"),
            "checkpoint_sha256": selected.get("checkpoint_sha256"),
            "init_adapter": selected.get("init_adapter"),
            "init_checkpoint": selected.get("init_checkpoint"),
        }
    )
    for key in [
        "hidden_dim",
        "dropout",
        "max_delta_norm",
        "shift_scale",
        "lambda_episode",
        "base_anchor_weight",
        "delta_l2",
        "gate_l2",
        "support_size",
        "query_size",
        "eval_support_size",
        "lr",
        "weight_decay",
        "train_selector",
        "eval_selector",
        "tasks_per_epoch",
        "max_epochs",
        "patience",
    ]:
        rec[f"cfg_{key}"] = cfg.get(key)
    return rec


def pick_best_sweep(summary: pd.DataFrame) -> pd.Series:
    keys = ["pseudo_val_Hit@10", "pseudo_val_MeanAP", "pseudo_val_NDCG@10", "pseudo_val_AUROC", "meta_val_noadapt_Hit@10"]
    return summary.sort_values(keys, ascending=[False] * len(keys)).iloc[0]


def run_sweep(args: argparse.Namespace, device: torch.device) -> None:
    r250 = load_round250_module()
    r247 = load_round247_module()
    context = load_context(r250, args, device)
    write_audit(context["data_audit"])
    (SOURCE / "cell_splits.json").write_text(json.dumps(json_ready(context["data_audit"]), indent=2), encoding="utf-8")
    rows = []
    histories = []
    selected_by_variant: dict[str, dict[str, Any]] = {}
    for cfg in sweep_configs(args):
        log(f"sweep start {cfg['sweep_name']} {cfg}")
        child = apply_sweep_config(args, cfg)
        adapter, history, selected = train_epiqgate(context, r247, child, device)
        row = selected_to_row(selected)
        row["sweep_name"] = cfg["sweep_name"]
        rows.append(row)
        history["sweep_name"] = cfg["sweep_name"]
        histories.append(history)
        selected_by_variant[str(selected["variant"])] = {"adapter": adapter, "selected": selected}
        pd.DataFrame(rows).to_csv(SOURCE / "sweep_summary.tsv", sep="\t", index=False)
        pd.concat(histories, ignore_index=True, sort=False).to_csv(SOURCE / "sweep_training_history.tsv", sep="\t", index=False)
        log(
            f"sweep done {cfg['sweep_name']} variant={selected['variant']} "
            f"epoch={row.get('epoch')} pseudo_hit10={row.get('pseudo_val_Hit@10'):.4f}"
        )

    summary = pd.DataFrame(rows)
    best_row = pick_best_sweep(summary)
    best_variant = str(best_row["variant"])
    best = selected_by_variant[best_variant]
    log(f"sweep selected {best_variant} by pseudo-unseen Hit@10={best_row['pseudo_val_Hit@10']:.4f}")

    report_sizes = [int(x) for x in args.report_support_sizes]
    selectors = list(args.report_selectors)
    all_metrics = []
    all_cells = []
    all_per = []
    all_audits = []
    support_metrics, support_cells, support_per = r250.evaluate_support_baseline(
        context["r243"],
        context["r208"],
        context["support_model"],
        context["support_adapter"],
        context["tables"],
        device,
        str(context["support_payload"]["joint_cfg"]["variant"]),
        float(context["support_payload"]["joint_cfg"]["shift_scale"]),
        report_sizes,
        keep_per_unit=True,
    )
    all_metrics.append(support_metrics)
    all_cells.append(support_cells)
    if len(support_per):
        all_per.append(support_per)

    epi_metrics, epi_cells, epi_per, epi_audit = evaluate_epiqgate_all(
        context,
        r247,
        best["adapter"],
        device,
        best["selected"],
        report_sizes,
        selectors,
        keep_per_unit=True,
    )
    all_metrics.append(epi_metrics)
    all_cells.append(epi_cells)
    if len(epi_per):
        all_per.append(epi_per)
    if len(epi_audit):
        all_audits.append(epi_audit)

    metrics_all = pd.concat(all_metrics, ignore_index=True, sort=False)
    cells_all = pd.concat(all_cells, ignore_index=True, sort=False)
    per_all = pd.concat(all_per, ignore_index=True, sort=False) if all_per else pd.DataFrame()
    audit_all = pd.concat(all_audits, ignore_index=True, sort=False) if all_audits else pd.DataFrame()
    by_bin = r250.summarize_by_sample_bin(cells_all)
    deltas = paired_delta_vs_noadapt(cells_all)
    rescue = paired_rescue_loss(per_all)
    metrics_all.to_csv(SOURCE / "metric_summary.tsv", sep="\t", index=False)
    cells_all.to_csv(SOURCE / "per_cell_metrics.tsv", sep="\t", index=False)
    by_bin.to_csv(SOURCE / "metric_summary_by_sample_bin.tsv", sep="\t", index=False)
    deltas.to_csv(SOURCE / "paired_delta_by_sample_bin.tsv", sep="\t", index=False)
    if len(per_all):
        per_all.to_csv(SOURCE / "per_unit_metrics.tsv", sep="\t", index=False)
    if len(audit_all):
        audit_all.to_csv(SOURCE / "support_selection_audit.tsv", sep="\t", index=False)
    write_method_lock(
        {
            "kg_checkpoint": str(context["kg_path"]),
            "kg_checkpoint_sha256": sha256_file(context["kg_path"]),
            "data_audit": context["data_audit"],
            "selected_epiqgate": best["selected"],
            "sweep_summary": summary.to_dict(orient="records"),
            "selection_policy": "pseudo-unseen validation only; official unseen evaluated after sweep selection",
            "round246_checkpoint": str(context["support_checkpoint"]),
            "round246_checkpoint_sha256": sha256_file(context["support_checkpoint"]),
            "round247_adapter_checkpoint": str(r247_ckpt) if r247_ckpt is not None else None,
            "round247_adapter_sha256": sha256_file(r247_ckpt) if r247_ckpt is not None else None,
        }
    )
    write_report(metrics_all, by_bin, deltas, rescue)
    write_round252_optimization_report(summary, metrics_all, by_bin, deltas)
    log(f"sweep done; outputs written to {ROUND_ROOT}")


def run_finalize_existing(args: argparse.Namespace, device: torch.device) -> None:
    r250 = load_round250_module()
    context = load_context(r250, args, device)
    summary_path = SOURCE / "sweep_summary.tsv"
    metrics_path = SOURCE / "metric_summary.tsv"
    by_bin_path = SOURCE / "metric_summary_by_sample_bin.tsv"
    deltas_path = SOURCE / "paired_delta_by_sample_bin.tsv"
    per_path = SOURCE / "per_unit_metrics.tsv"
    if not summary_path.exists() or not metrics_path.exists() or not by_bin_path.exists() or not deltas_path.exists():
        raise FileNotFoundError("Cannot finalize: sweep output tables are incomplete")
    summary = pd.read_csv(summary_path, sep="\t")
    metrics = pd.read_csv(metrics_path, sep="\t")
    by_bin = pd.read_csv(by_bin_path, sep="\t")
    deltas = pd.read_csv(deltas_path, sep="\t")
    per = pd.read_csv(per_path, sep="\t") if per_path.exists() else pd.DataFrame()
    rescue = paired_rescue_loss(per) if len(per) else pd.DataFrame()
    best_row = pick_best_sweep(summary)
    selected_json = SOURCE / f"selected_{best_row['variant']}.json"
    if not selected_json.exists():
        raise FileNotFoundError(selected_json)
    selected = json.loads(selected_json.read_text(encoding="utf-8"))
    r247_ckpt = resolve_round247_adapter_checkpoint(args.round247_adapter_checkpoint)
    write_method_lock(
        {
            "kg_checkpoint": str(context["kg_path"]),
            "kg_checkpoint_sha256": sha256_file(context["kg_path"]),
            "data_audit": context["data_audit"],
            "selected_epiqgate": selected,
            "sweep_summary": summary.to_dict(orient="records"),
            "selection_policy": "pseudo-unseen validation only; official unseen evaluated after sweep selection",
            "round246_checkpoint": str(context["support_checkpoint"]),
            "round246_checkpoint_sha256": sha256_file(context["support_checkpoint"]),
            "round247_adapter_checkpoint": str(r247_ckpt) if r247_ckpt is not None else None,
            "round247_adapter_sha256": sha256_file(r247_ckpt) if r247_ckpt is not None else None,
            "finalized_from_existing_tables": True,
        }
    )
    write_report(metrics, by_bin, deltas, rescue)
    write_round252_optimization_report(summary, metrics, by_bin, deltas)
    log(f"finalized existing Round252 outputs; selected {best_row['variant']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["audit_protocol", "smoke", "train", "sweep", "finalize_existing"], default="train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--kg-checkpoint", default=None)
    parser.add_argument("--pseudo-unseen-cells", default=",".join(["HA1E", "HEK293", "MCF10A", "MDAMB231", "PHH", "THP1"]))
    parser.add_argument("--official-unseen-cells", default="round215_heldout_panel")
    parser.add_argument("--seen-diagnostic-cells", default=",".join(FORCED_TRAIN_CELLS))
    parser.add_argument("--support-baseline-variant", default="r246_joint_lm0p5_sc0p15_mn2p5")
    parser.add_argument("--support-checkpoint", default=None)
    parser.add_argument("--round247-adapter-checkpoint", default=None)
    parser.add_argument("--round251-adapter-checkpoint", default=None)
    parser.add_argument("--init-adapter", choices=["round247", "round251", "none"], default="round247")
    parser.add_argument("--adapter-hidden-dim", type=int, default=128)
    parser.add_argument("--adapter-dropout", type=float, default=0.10)
    parser.add_argument("--max-delta-norm", type=float, default=2.5)
    parser.add_argument("--shift-scale", type=float, default=0.15)
    parser.add_argument("--base-anchor-weight", type=float, default=0.10)
    parser.add_argument("--delta-l2", type=float, default=1.0e-3)
    parser.add_argument("--gate-l2", type=float, default=1.0e-3)
    parser.add_argument("--train-selector", choices=SUPPORT_SELECTORS, default="diverse_label")
    parser.add_argument("--eval-selector", choices=SUPPORT_SELECTORS, default="diverse_label")
    parser.add_argument("--train-support-size", type=int, default=32)
    parser.add_argument("--train-query-size", type=int, default=32)
    parser.add_argument("--eval-support-size", type=int, default=32)
    parser.add_argument("--report-support-sizes", nargs="+", type=int, default=SUPPORT_SIZES)
    parser.add_argument("--report-selectors", nargs="+", default=SUPPORT_SELECTORS)
    parser.add_argument("--tasks-per-epoch", type=int, default=240)
    parser.add_argument("--max-epochs", type=int, default=36)
    parser.add_argument("--min-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--adapter-lr", type=float, default=1.5e-4)
    parser.add_argument("--adapter-weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--lambda-episode", type=float, default=1.0)
    parser.add_argument("--run-seed", type=int, default=SEED)
    parser.add_argument("--skip-hybrid", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    ensure_dirs()
    require_torch()
    args = parse_args()
    if args.phase == "smoke":
        args.smoke = True
        args.max_epochs = 1
        args.min_epochs = 1
        args.patience = 1
        args.tasks_per_epoch = min(args.tasks_per_epoch, 4)
        args.report_support_sizes = [0, 32]
        args.report_selectors = ["diverse_label"]
    set_seed(int(args.run_seed))
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    log(f"using device={device}")
    if args.phase == "audit_protocol":
        run_audit(args, device)
    elif args.phase == "sweep":
        run_sweep(args, device)
    elif args.phase == "finalize_existing":
        run_finalize_existing(args, device)
    else:
        run_training(args, device)


if __name__ == "__main__":
    main()
