"""Cell-as-task episodic meta-learning for support-conditioned target ranking."""

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
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from metar.runtime import report_metrics, report_progress, report_step, resolve_device

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

ROUND243_ROOT = METAR_ROOT / "claude_round243_leakage_safe_unseen_cell_support_adapter_20260519"
ROUND243_SCRIPT = ROUND243_ROOT / "scripts" / "run_round243_leakage_safe_unseen_cell_support_adapter.py"
ROUND246_ROOT = METAR_ROOT / "claude_round246_joint_seen_target_train_pc3_a375_mcf7_20260519"
ROUND246_SCRIPT = ROUND246_ROOT / "scripts" / "run_round246_joint_seen_target_train_pc3_a375_mcf7.py"
ROUND246_SELECTED = ROUND246_ROOT / "source_data" / "selected_r246_joint_lm0p5_sc0p15_mn2p5.json"
ROUND247_ROOT = METAR_ROOT / "claude_round247_support_curve_query_gated_adapter_20260520"
ROUND247_SCRIPT = ROUND247_ROOT / "scripts" / "run_round247_support_curve_query_gated_adapter.py"
ROUND247_SELECTED = ROUND247_ROOT / "source_data" / "selected_r247_query_gated_h128_sc0p15_mn2p5.json"

SEED = 20260520
BASE_REFERENCE = {
    "round246_current_joint": "r246_joint_lm0p5_sc0p15_mn2p5",
    "round247_query_gated": "r247_query_gated_h128_sc0p15_mn2p5",
}
FORCED_TRAIN_CELLS = ["PC3", "A375", "MCF7"]
DEFAULT_R250_PSEUDO_UNSEEN_VAL_CELLS = ["HA1E", "HEK293", "MCF10A", "MDAMB231", "PHH", "THP1"]

READOUT_CFG = {
    "readout_id": "r250_epmeta_stable_tuning",
    "hidden": 256,
    "extra_layers": 0,
    "dropout": 0.35,
    "lr": 5.0e-4,
    "weight_decay": 3.0e-4,
    "target_bias_l2": 5.0e-4,
    "target_delta_l2": 5.0e-3,
    "scorer_l2": 2.0e-4,
    "max_epochs": 60,
    "min_epochs": 12,
    "patience": 8,
    "batch_size": 256,
}

MAML_DEFAULT = {
    "variant": "r250_epmeta_query_bias_lm0p08_ilr0p0015_s32_q32",
    "inner_kind": "query_bias",
    "lambda_meta": 0.08,
    "inner_lr": 1.5e-3,
    "inner_steps": 1,
    "support_size": 32,
    "query_size": 32,
    "tasks_per_epoch": 300,
    "max_epochs": 36,
    "min_epochs": 10,
    "patience": 10,
    "meta_lr": 7.5e-5,
    "weight_decay": 3.0e-4,
    "batch_size": 256,
    "eval_support_size": 32,
}

HIT_KS = [1, 5, 10, 20, 30, 50]
METRIC_COLS = [f"Hit@{k}" for k in HIT_KS] + ["AUROC", "NDCG@10", "MeanAP"]


def ensure_dirs() -> None:
    for path in [SOURCE, QA, CHECKPOINTS, LOGS]:
        path.mkdir(parents=True, exist_ok=True)


def require_torch() -> None:
    if torch is None or F is None:
        raise ModuleNotFoundError("torch is required for Round250 training/evaluation")


def log(message: str) -> None:
    print(f"[round250] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def set_seed(seed: int = SEED) -> None:
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


def load_round243():
    module = import_from_path("round250_round243", ROUND243_SCRIPT)
    module.ROUND_ROOT = ROUND_ROOT
    module.SOURCE = SOURCE
    module.QA = QA
    module.CHECKPOINTS = CHECKPOINTS
    module.LOGS = LOGS
    return module


def load_round247_module():
    module = import_from_path("round250_round247", ROUND247_SCRIPT)
    module.ROUND_ROOT = ROUND_ROOT
    module.SOURCE = SOURCE
    module.QA = QA
    module.CHECKPOINTS = CHECKPOINTS
    module.LOGS = LOGS
    return module


def fmt_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def resolve_current_checkpoint(variant: str, user_path: str | None = None) -> Path:
    if user_path:
        path = Path(user_path).expanduser()
        if path.exists():
            return path.resolve()
        raise FileNotFoundError(path)
    rel = Path("checkpoints") / variant / "whole_model_checkpoint.pth"
    candidates = [
        ROUND246_ROOT / rel,
        Path("/NFS2_home/NFS2_home_1/lwn/claude/manuscript/write")
        / ROUND246_ROOT.name
        / rel,
        Path("/home/lwn/claude/manuscript/write") / ROUND246_ROOT.name / rel,
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError("Round246 current checkpoint not found: " + "\n".join(map(str, candidates)))


def resolve_query_gated_checkpoint(user_path: str | None = None) -> Path | None:
    if user_path:
        path = Path(user_path).expanduser()
        if path.exists():
            return path.resolve()
        raise FileNotFoundError(path)
    variant = BASE_REFERENCE["round247_query_gated"]
    rel = Path("checkpoints") / variant / "adapter_checkpoint.pth"
    candidates = [
        ROUND247_ROOT / rel,
        Path("/NFS2_home/NFS2_home_1/lwn/claude/manuscript/write") / ROUND247_ROOT.name / rel,
        Path("/home/lwn/claude/manuscript/write") / ROUND247_ROOT.name / rel,
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def load_context(args: argparse.Namespace, device: torch.device):
    report_step('Loading checkpoints and preparing bulk split tables')
    r243 = load_round243()
    r211 = r243.load_round211_module()
    r208 = r243.load_round208_module()
    kg_path = r243.resolve_kg_checkpoint(args.kg_checkpoint)
    kg_ckpt = torch.load(kg_path, map_location="cpu", weights_only=False)
    target_entities = [str(x) for x in kg_ckpt["target_entities"]]
    labels = [x.split(":", 1)[1] if ":" in x else x for x in target_entities]
    if len(labels) != 668 or len(set(labels)) != 668:
        raise RuntimeError(f"Expected 668 unique targets, observed {len(labels)} / {len(set(labels))}")
    cell_sets = r243.choose_round243_cells(
        r208,
        args.pseudo_unseen_cells,
        args.official_unseen_cells,
        args.seen_diagnostic_cells,
    )
    tables, data_audit = r243.build_round243_tables(r208, labels, cell_sets, device)
    for name, item in tables.items():
        if item["x"].shape[1] != 978:
            raise RuntimeError(f"{name} expected 978 input features, observed {item['x'].shape[1]}")
        if item["y"].shape[1] != 668:
            raise RuntimeError(f"{name} expected 668 targets, observed {item['y'].shape[1]}")
    missing_forced = [
        cell
        for cell in FORCED_TRAIN_CELLS
        if cell not in set(cell_sets["base_train_cells"]) or cell not in set(cell_sets["adapter_train_cells"])
    ]
    if missing_forced:
        raise RuntimeError(
            "Forced current training cells are not all in base_train/adapter_train: "
            + ", ".join(missing_forced)
        )
    data_audit["round250_policy"] = {
        "mode": "current Round246/Round247 baseline plus stable episodic cell-as-task meta-learning tuning",
        "candidate_space": 668,
        "kg_checkpoint": str(kg_path),
        "kg_checkpoint_sha256": sha256_file(kg_path),
        "kg_retrained": False,
        "forced_train_cells": FORCED_TRAIN_CELLS,
        "base_initializer": BASE_REFERENCE["round246_current_joint"],
        "query_gated_reference": BASE_REFERENCE["round247_query_gated"],
    }
    return r243, r211, r208, labels, kg_path, kg_ckpt, tables, data_audit


def pget(model: nn.Module, params: Optional[dict[str, torch.Tensor]], name: str) -> torch.Tensor:
    if params is not None and name in params:
        return params[name]
    current: Any = model
    for part in name.split("."):
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    return current


def functional_logits(
    model: nn.Module,
    x: torch.Tensor,
    params: Optional[dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    if params is None:
        return model(x)
    h = x
    for idx, layer in enumerate(model.query_encoder):
        prefix = f"query_encoder.{idx}"
        if isinstance(layer, nn.Linear):
            h = F.linear(h, pget(model, params, f"{prefix}.weight"), pget(model, params, f"{prefix}.bias"))
        elif isinstance(layer, nn.LayerNorm):
            h = F.layer_norm(
                h,
                layer.normalized_shape,
                pget(model, params, f"{prefix}.weight"),
                pget(model, params, f"{prefix}.bias"),
                layer.eps,
            )
        elif isinstance(layer, nn.GELU):
            h = F.gelu(h)
        elif isinstance(layer, nn.Dropout):
            h = F.dropout(h, p=layer.p, training=model.training)
        else:
            raise TypeError(f"Unsupported query_encoder layer {idx}: {type(layer)!r}")
    q = F.layer_norm(
        h,
        model.query_norm.normalized_shape,
        pget(model, params, "query_norm.weight"),
        pget(model, params, "query_norm.bias"),
        model.query_norm.eps,
    )
    q = F.normalize(q, dim=1)
    target = F.normalize(model.target_anchor + pget(model, params, "target_delta"), dim=1)
    score = (q * pget(model, params, "diag")[None, :]) @ target.T
    score = score * torch.exp(torch.clamp(pget(model, params, "score_scale"), -2.0, 2.0))
    return score + pget(model, params, "target_bias")[None, :]


def batch_tensors(item: dict, idx: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(item["x"][idx], dtype=torch.float32, device=device),
        torch.tensor(item["y"][idx], dtype=torch.float32, device=device),
        torch.tensor(item["weights"][idx], dtype=torch.float32, device=device),
    )


def loss_from_item(
    r243,
    model: nn.Module,
    item: dict,
    idx: np.ndarray,
    device: torch.device,
    params: Optional[dict[str, torch.Tensor]] = None,
    include_reg: bool = False,
) -> torch.Tensor:
    xb, yb, wb = batch_tensors(item, idx, device)
    loss = r243.retrieval_loss_from_logits(functional_logits(model, xb, params=params), yb, wb)
    if include_reg and params is None:
        loss = loss + model.regularization()
    return loss


def final_query_layer_indices(model: nn.Module) -> tuple[int, int]:
    linear = [idx for idx, layer in enumerate(model.query_encoder) if isinstance(layer, nn.Linear)]
    norm = [idx for idx, layer in enumerate(model.query_encoder) if isinstance(layer, nn.LayerNorm)]
    if not linear or not norm:
        raise RuntimeError("Cannot infer query-tail layers")
    return linear[-1], norm[-1]


def inner_parameter_names(model: nn.Module, kind: str) -> list[str]:
    final_linear, final_norm = final_query_layer_indices(model)
    base = [
        f"query_encoder.{final_linear}.weight",
        f"query_encoder.{final_linear}.bias",
        f"query_encoder.{final_norm}.weight",
        f"query_encoder.{final_norm}.bias",
        "query_norm.weight",
        "query_norm.bias",
        "diag",
        "score_scale",
    ]
    if kind == "query_tail":
        return base
    if kind == "query_bias":
        return base + ["target_bias"]
    raise ValueError(f"Unknown inner kind: {kind}")


def adapt_params(
    r243,
    model: nn.Module,
    support_item: dict,
    support_idx: np.ndarray,
    device: torch.device,
    inner_names: list[str],
    inner_lr: float,
    inner_steps: int,
) -> dict[str, torch.Tensor]:
    params = {name: p for name, p in model.named_parameters()}
    for _ in range(inner_steps):
        loss = loss_from_item(r243, model, support_item, support_idx, device, params=params, include_reg=False)
        grads = torch.autograd.grad(loss, [params[name] for name in inner_names], create_graph=False, allow_unused=False)
        updated = dict(params)
        for name, grad in zip(inner_names, grads):
            updated[name] = params[name] - float(inner_lr) * grad
        params = updated
    return params


def sample_drug_disjoint_episode(
    item: dict,
    cell: str,
    support_size: int,
    query_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    rows = item["rows"]
    all_idx = np.asarray(item["by_cell"][cell], dtype=np.int64)
    cell_rows = rows.iloc[all_idx].copy()
    drugs = cell_rows["drug"].astype(str).unique()
    if len(drugs) < support_size + query_size:
        raise RuntimeError(f"Cell {cell} has too few drugs for a disjoint episode")
    perm_drugs = rng.permutation(drugs)
    support_drugs = set(map(str, perm_drugs[:support_size]))
    query_drugs = set(map(str, perm_drugs[support_size : support_size + query_size]))
    drug_values = cell_rows["drug"].astype(str).to_numpy()
    support_idx = all_idx[np.isin(drug_values, list(support_drugs))]
    query_idx = all_idx[np.isin(drug_values, list(query_drugs))]
    if len(support_idx) > support_size:
        support_idx = rng.choice(support_idx, size=support_size, replace=False)
    if len(query_idx) > query_size:
        query_idx = rng.choice(query_idx, size=query_size, replace=False)
    overlap = set(rows.iloc[support_idx]["drug"].astype(str)).intersection(set(rows.iloc[query_idx]["drug"].astype(str)))
    if overlap:
        raise RuntimeError(f"Drug overlap inside MAML episode: {sorted(overlap)[:5]}")
    return np.asarray(support_idx, dtype=np.int64), np.asarray(query_idx, dtype=np.int64)


def eligible_cells(item: dict, support_size: int, query_size: int) -> list[str]:
    rows = item["rows"]
    out = []
    for cell, idx in item["by_cell"].items():
        n_drugs = rows.iloc[np.asarray(idx, dtype=np.int64)]["drug"].astype(str).nunique()
        if n_drugs >= support_size + query_size:
            out.append(str(cell))
    return out


def deterministic_support_indices(rows: pd.DataFrame, size: int) -> np.ndarray:
    order = np.lexsort(
        (
            rows["profile_id"].astype(str).to_numpy(),
            rows["drug"].astype(str).to_numpy(),
            rows["n_profiles"].to_numpy(dtype=np.int64) * -1,
        )
    )
    if size <= 0 or len(order) <= size:
        return order.astype(np.int64)
    return order[:size].astype(np.int64)


def predict_functional(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    params: Optional[dict[str, torch.Tensor]] = None,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    detached_params = {k: v.detach() for k, v in params.items()} if params is not None else None
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            report_progress(start, len(x), label='predict functional rows')
            xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            out.append(functional_logits(model, xb, params=detached_params).detach().cpu().numpy())
        report_progress(len(x), len(x), label='predict functional rows')
    return np.vstack(out) if out else np.zeros((0, model.n_targets), dtype=np.float32)


def noadapt_summary(r208, model: nn.Module, item: dict, device: torch.device, variant: str, unit: str) -> dict:
    scores = predict_functional(model, item["x"], device)
    summary, _ = r208.compute_metrics(scores, item["y"], item["rows"], "Round250_EpMeta", variant, unit)
    return summary


def better_candidate(candidate: dict[str, Any], incumbent: Optional[dict[str, Any]]) -> bool:
    if incumbent is None:
        return True
    for key in ["pseudo_val_Hit@10", "pseudo_val_MeanAP", "pseudo_val_NDCG@10", "meta_val_noadapt_Hit@10"]:
        left = float(candidate.get(key, float("nan")))
        right = float(incumbent.get(key, float("nan")))
        if math.isnan(left) and math.isnan(right):
            continue
        if math.isnan(left):
            return False
        if math.isnan(right):
            return True
        if abs(left - right) > 1.0e-12:
            return left > right
    return False


def sample_bin_from_support(n_support_groups: int) -> str:
    if n_support_groups <= 64:
        return "small_<=64_support_groups"
    if n_support_groups <= 256:
        return "medium_65_256_support_groups"
    return "large_>256_support_groups"


def add_cell_count_columns(cell_df: pd.DataFrame, support_item: dict) -> pd.DataFrame:
    out = cell_df.copy()
    support_counts = support_item["rows"].groupby("cell_line").agg(
        support_groups=("drug", "size"),
        support_drugs=("drug", "nunique"),
        support_profiles=("n_profiles", "sum"),
    )
    out["support_groups"] = out["cell_line"].map(support_counts["support_groups"]).fillna(0).astype(int)
    out["support_drugs"] = out["cell_line"].map(support_counts["support_drugs"]).fillna(0).astype(int)
    out["support_profiles"] = out["cell_line"].map(support_counts["support_profiles"]).fillna(0).astype(int)
    out["sample_bin"] = out["support_groups"].map(sample_bin_from_support)
    return out


def evaluate_maml_by_cell(
    r243,
    r208,
    model: nn.Module,
    support_item: dict,
    query_item: dict,
    device: torch.device,
    variant: str,
    support_size: int,
    inner_names: list[str],
    inner_lr: float,
    inner_steps: int,
    unit: str,
    keep_per_unit: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell_summaries = []
    per_parts = []
    model.eval()
    for _cell_index, (cell, qidx_raw) in enumerate(query_item["by_cell"].items()):
        report_progress(_cell_index, len(query_item["by_cell"]), label=f"{variant}: cells inspected")
        qidx = np.asarray(qidx_raw, dtype=np.int64)
        if len(qidx) == 0:
            continue
        params = None
        used_support = 0
        if support_size > 0:
            support_all = support_item["by_cell"].get(cell, np.asarray([], dtype=np.int64))
            if len(support_all) == 0:
                continue
            local = deterministic_support_indices(support_item["rows"].iloc[support_all], support_size)
            support_idx = np.asarray(support_all, dtype=np.int64)[local]
            used_support = int(len(support_idx))
            params = adapt_params(r243, model, support_item, support_idx, device, inner_names, inner_lr, inner_steps)
        qrows = query_item["rows"].iloc[qidx].reset_index(drop=True)
        scores = predict_functional(model, query_item["x"][qidx], device, params=params)
        summary, per = r208.compute_metrics(scores, query_item["y"][qidx], qrows, "Round250_EpMeta", variant, unit)
        summary.update(
            {
                "cell_line": cell,
                "adapt_support_size": int(support_size),
                "used_support_rows": used_support,
                "inner_lr": float(inner_lr),
                "inner_steps": int(inner_steps),
            }
        )
        cell_summaries.append(summary)
        if keep_per_unit:
            per["adapt_support_size"] = int(support_size)
            per["used_support_rows"] = used_support
            per_parts.append(per)
    report_progress(len(query_item["by_cell"]), len(query_item["by_cell"]), label=f"{variant}: cells inspected")
    cells = add_cell_count_columns(pd.DataFrame(cell_summaries), support_item)
    per_all = pd.concat(per_parts, ignore_index=True) if per_parts else pd.DataFrame()
    return cells, per_all


def metric_macro_from_cells(cell_df: pd.DataFrame, variant: str, split_name: str, unit: str, source: str, family: str) -> dict:
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
    for extra in ["adapt_support_size", "inner_lr", "inner_steps"]:
        if extra in cell_df and len(cell_df):
            out[extra] = float(cell_df[extra].iloc[0]) if extra != "adapt_support_size" else int(cell_df[extra].iloc[0])
    for metric in METRIC_COLS:
        out[metric] = float(cell_df[metric].mean()) if len(cell_df) else float("nan")
    return out


def selection_record(
    r243,
    r208,
    model: nn.Module,
    tables: dict[str, dict],
    device: torch.device,
    cfg: dict[str, Any],
    epoch: int,
    train_loss: float,
    inner_names: list[str],
) -> dict[str, Any]:
    pseudo_cells, _ = evaluate_maml_by_cell(
        r243,
        r208,
        model,
        tables["pseudo_support"],
        tables["pseudo_val"],
        device,
        f"{cfg['variant']}_pseudo_val_maml_adapt{cfg['eval_support_size']}",
        int(cfg["eval_support_size"]),
        inner_names,
        float(cfg["inner_lr"]),
        int(cfg["inner_steps"]),
        "drug-mean",
        False,
    )
    pseudo = metric_macro_from_cells(
        pseudo_cells,
        f"{cfg['variant']}_pseudo_val_maml_adapt{cfg['eval_support_size']}",
        "pseudo_val",
        "drug-mean",
                "Round250_EpMeta_pseudo_validation",
        "seen_668_maml_adapt",
    )
    meta = noadapt_summary(r208, model, tables["meta_val"], device, f"{cfg['variant']}_meta_val_no_adapt", "drug-mean")
    return {
        "variant": cfg["variant"],
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "pseudo_val_Hit@10": float(pseudo["Hit@10"]),
        "pseudo_val_MeanAP": float(pseudo["MeanAP"]),
        "pseudo_val_NDCG@10": float(pseudo["NDCG@10"]),
        "pseudo_val_AUROC": float(pseudo["AUROC"]),
        "meta_val_noadapt_Hit@10": float(meta["Hit@10"]),
        "meta_val_noadapt_MeanAP": float(meta["MeanAP"]),
        "inner_kind": cfg["inner_kind"],
        "inner_lr": float(cfg["inner_lr"]),
        "inner_steps": int(cfg["inner_steps"]),
        "lambda_meta": float(cfg["lambda_meta"]),
    }


def train_maml_variant(
    r243,
    r211,
    r208,
    kg_ckpt: dict,
    labels: list[str],
    tables: dict[str, dict],
    device: torch.device,
    cfg: dict[str, Any],
    initializer_checkpoint: Path,
    smoke: bool,
) -> tuple[nn.Module, pd.DataFrame, dict[str, Any]]:
    report_step('Preparing episodic meta-learning')
    run_seed = int(cfg.get("run_seed", SEED))
    set_seed(run_seed)
    init_payload = torch.load(initializer_checkpoint, map_location="cpu", weights_only=False)
    readout_cfg = deepcopy(init_payload.get("readout_cfg", READOUT_CFG))
    readout_cfg["readout_id"] = cfg["variant"]
    if smoke:
        cfg = deepcopy(cfg)
        cfg["tasks_per_epoch"] = 4
        cfg["max_epochs"] = 1
        cfg["min_epochs"] = 1
        cfg["patience"] = 1
    target_init, target_audit = r211.target_embeddings_from_ckpt(kg_ckpt, labels, cfg["variant"])
    model = r211.DiagReadout(target_init, readout_cfg).to(device)
    model.load_state_dict(init_payload["readout_state_dict"], strict=True)
    inner_names = inner_parameter_names(model, str(cfg["inner_kind"]))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["meta_lr"]), weight_decay=float(cfg["weight_decay"]))

    train = tables["meta_train"]
    n_train = int(len(train["rows"]))
    rng = np.random.default_rng(run_seed)
    cells = eligible_cells(train, int(cfg["support_size"]), int(cfg["query_size"]))
    if not cells:
        raise RuntimeError("No eligible cells for MAML training")
    best: Optional[dict[str, Any]] = None
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    report_progress(0, int(cfg["max_epochs"]), label='maml variant epochs')
    report_metrics(cfg, label="Training settings")
    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        model.train()
        losses = []
        for _ in range(int(cfg["tasks_per_epoch"])):
            report_progress(_, int(cfg["tasks_per_epoch"]), label=f"Epoch {epoch} training episodes")
            base_idx = rng.integers(0, n_train, size=int(cfg["batch_size"]))
            base_loss = loss_from_item(r243, model, train, base_idx, device, params=None, include_reg=False)
            cell = str(rng.choice(cells))
            support_idx, query_idx = sample_drug_disjoint_episode(
                train,
                cell,
                int(cfg["support_size"]),
                int(cfg["query_size"]),
                rng,
            )
            adapted = adapt_params(
                r243,
                model,
                train,
                support_idx,
                device,
                inner_names,
                float(cfg["inner_lr"]),
                int(cfg["inner_steps"]),
            )
            episode_loss = loss_from_item(r243, model, train, query_idx, device, params=adapted, include_reg=False)
            loss = base_loss + float(cfg["lambda_meta"]) * episode_loss + model.regularization()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        report_progress(int(cfg["tasks_per_epoch"]), int(cfg["tasks_per_epoch"]), label=f"Epoch {epoch} training episodes")
        report_step(f"Epoch {epoch}: evaluating existing training and validation metrics")
        rec = selection_record(
            r243,
            r208,
            model,
            tables,
            device,
            cfg,
            epoch,
            float(np.mean(losses)),
            inner_names,
        )
        history.append(rec)
        report_progress(epoch, int(cfg["max_epochs"]), label='maml variant epochs')
        report_metrics(
            {key: rec[key] for key in ("train_loss", "pseudo_val_Hit@10", "pseudo_val_MeanAP")},
            label=f"Epoch {epoch}/{int(cfg['max_epochs'])}",
        )
        if epoch == 1 or epoch % 5 == 0:
            log(
                f"{cfg['variant']} epoch={epoch} "
                f"pseudo_hit10={rec['pseudo_val_Hit@10']:.4f} "
                f"meta_val_hit10={rec['meta_val_noadapt_Hit@10']:.4f} "
                f"loss={rec['train_loss']:.4f}"
            )
        if better_candidate(rec, best):
            best = rec
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch >= int(cfg["min_epochs"]) and stale >= int(cfg["patience"]):
            log(f"{cfg['variant']} early stop epoch={epoch}; best_epoch={best_epoch}")
            break
    if best is None or best_state is None:
        raise RuntimeError(f"No MAML checkpoint selected for {cfg['variant']}")
    report_metrics(
        {"epochs_run": len(history), "best_epoch": best_epoch,
         "pseudo_val_Hit@10": best["pseudo_val_Hit@10"],
         "pseudo_val_MeanAP": best["pseudo_val_MeanAP"]},
        label="Selected meta-learning validation metrics",
    )
    model.load_state_dict(best_state)
    ckpt_dir = CHECKPOINTS / cfg["variant"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "whole_model_checkpoint.pth"
    torch.save(
        {
            "stage": "round250_epmeta_stable_tuning",
            "readout_state_dict": best_state,
            "readout_cfg": readout_cfg,
            "maml_cfg": cfg,
            "best_epoch": best_epoch,
            "selection": best,
            "inner_parameter_names": inner_names,
            "target_audit": target_audit,
            "initializer_checkpoint": str(initializer_checkpoint),
            "initializer_checkpoint_sha256": sha256_file(initializer_checkpoint),
            "initializer_variant": init_payload.get("joint_cfg", {}).get("variant"),
            "checkpoint_policy": "whole model readout state selected from one episodic meta-training epoch",
            "run_seed": run_seed,
        },
        ckpt_path,
    )
    selected = {
        "variant": cfg["variant"],
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "best_epoch": best_epoch,
        "selection": best,
        "readout_cfg": readout_cfg,
        "maml_cfg": cfg,
        "inner_parameter_names": inner_names,
        "target_audit": target_audit,
        "initializer_checkpoint": str(initializer_checkpoint),
        "initializer_checkpoint_sha256": sha256_file(initializer_checkpoint),
        "initializer_variant": init_payload.get("joint_cfg", {}).get("variant"),
        "run_seed": run_seed,
    }
    (SOURCE / f"selected_{cfg['variant']}.json").write_text(json.dumps(json_ready(selected), indent=2), encoding="utf-8")
    return model, pd.DataFrame(history), selected


def load_maml_model(r211, kg_ckpt: dict, labels: list[str], selected: dict[str, Any], device: torch.device) -> nn.Module:
    payload = torch.load(selected["checkpoint"], map_location="cpu", weights_only=False)
    readout_cfg = deepcopy(payload["readout_cfg"])
    target_init, _ = r211.target_embeddings_from_ckpt(kg_ckpt, labels, readout_cfg["readout_id"])
    model = r211.DiagReadout(target_init, readout_cfg).to(device)
    model.load_state_dict(payload["readout_state_dict"], strict=True)
    model.eval()
    return model


def load_current_support_baseline(r243, r211, kg_ckpt: dict, labels: list[str], checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    readout_cfg = deepcopy(payload["readout_cfg"])
    target_init, _ = r211.target_embeddings_from_ckpt(kg_ckpt, labels, readout_cfg["readout_id"])
    model = r211.DiagReadout(target_init, readout_cfg).to(device)
    adapter = r243.SupportDeltaEncoder(
        model.target_anchor.detach(),
        hidden_dim=int(payload["joint_cfg"]["support_hidden_dim"]),
        dropout=float(payload["joint_cfg"]["support_dropout"]),
        max_delta_norm=float(payload["joint_cfg"]["max_delta_norm"]),
    ).to(device)
    model.load_state_dict(payload["readout_state_dict"], strict=True)
    adapter.load_state_dict(payload["support_adapter_state_dict"], strict=True)
    model.eval()
    adapter.eval()
    return model, adapter, payload


def load_query_gated_baseline(r247, model: nn.Module, checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["cfg"]
    adapter = r247.QueryGatedSupportAdapter(
        model.target_anchor.detach(),
        hidden_dim=int(cfg["hidden_dim"]),
        dropout=float(cfg["dropout"]),
        max_delta_norm=float(cfg["max_delta_norm"]),
    ).to(device)
    adapter.load_state_dict(payload["adapter_state_dict"], strict=True)
    adapter.eval()
    return adapter, payload


def resolve_support_checkpoint(variant: str, user_path: str | None = None) -> Path:
    return resolve_current_checkpoint(variant, user_path)


def load_support_baseline(r243, r211, kg_ckpt: dict, labels: list[str], checkpoint: Path, device: torch.device):
    return load_current_support_baseline(r243, r211, kg_ckpt, labels, checkpoint, device)


def evaluate_maml_model(
    r243,
    r208,
    model: nn.Module,
    tables: dict[str, dict],
    device: torch.device,
    selected: dict[str, Any],
    support_sizes: Iterable[int],
    keep_per_unit: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = selected["maml_cfg"]
    inner_names = selected["inner_parameter_names"]
    metrics = []
    cells_all = []
    per_all = []
    splits = [
        ("pseudo_support", "pseudo_val", "pseudo_val", "drug-mean", "pseudo_unseen_validation", "Round250_EpMeta_pseudo_validation"),
        ("official_support", "official_test", "official_test", "drug-mean", "official_unseen", "Round250_EpMeta_official_unseen"),
        ("official_support", "official_test_profile", "official_test_profile", "true-profile-specific", "official_unseen", "Round250_EpMeta_official_unseen"),
        ("seen_support", "seen_test", "seen_test", "drug-mean", "seen_diagnostic", "Round250_EpMeta_seen_diagnostic"),
        ("seen_support", "seen_test_profile", "seen_test_profile", "true-profile-specific", "seen_diagnostic", "Round250_EpMeta_seen_diagnostic"),
    ]
    support_sizes = tuple(support_sizes)
    _eval_total = len(support_sizes) * len(splits)
    _eval_completed = 0
    report_progress(0, _eval_total, label='MAML evaluation blocks')
    for support_size in support_sizes:
        for support_key, query_key, split_name, unit, split_role, source in splits:
            report_step(f"Evaluating support={support_size}, split={split_name}")
            variant = f"{selected['variant']}_{split_role}_maml_adapt{support_size}"
            cells, per = evaluate_maml_by_cell(
                r243,
                r208,
                model,
                tables[support_key],
                tables[query_key],
                device,
                variant,
                int(support_size),
                inner_names,
                float(cfg["inner_lr"]),
                int(cfg["inner_steps"]),
                unit,
                keep_per_unit and split_role == "official_unseen",
            )
            cells["method_family"] = "maml"
            cells["split_role"] = split_role
            cells["variant_core"] = selected["variant"]
            cells_all.append(cells)
            metrics.append(
                metric_macro_from_cells(cells, variant, split_name, unit, source, "seen_668_maml_adapt" if support_size else "seen_668_no_adapt")
            )
            if len(per):
                per["method_family"] = "maml"
                per["split_role"] = split_role
                per["variant"] = variant
                per_all.append(per)
            _eval_completed += 1
            report_progress(_eval_completed, _eval_total, label='MAML evaluation blocks')
    return pd.DataFrame(metrics), pd.concat(cells_all, ignore_index=True), pd.concat(per_all, ignore_index=True) if per_all else pd.DataFrame()


def evaluate_support_baseline(
    r243,
    r208,
    model: nn.Module,
    adapter: nn.Module,
    tables: dict[str, dict],
    device: torch.device,
    variant_core: str,
    shift_scale: float,
    support_sizes: Iterable[int],
    keep_per_unit: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for key in [
        "pseudo_support",
        "pseudo_val",
        "official_support",
        "official_test",
        "official_test_profile",
        "seen_support",
        "seen_test",
        "seen_test_profile",
    ]:
        r243.precompute_q(model, tables[key], device)
    metrics = []
    cells_all = []
    per_all = []
    splits = [
        ("pseudo_support", "pseudo_val", "pseudo_val", "drug-mean", "pseudo_unseen_validation", "Round246_support_adapt_pseudo_validation"),
        ("official_support", "official_test", "official_test", "drug-mean", "official_unseen", "Round246_support_adapt_official_unseen"),
        ("official_support", "official_test_profile", "official_test_profile", "true-profile-specific", "official_unseen", "Round246_support_adapt_official_unseen"),
        ("seen_support", "seen_test", "seen_test", "drug-mean", "seen_diagnostic", "Round246_support_adapt_seen_diagnostic"),
        ("seen_support", "seen_test_profile", "seen_test_profile", "true-profile-specific", "seen_diagnostic", "Round246_support_adapt_seen_diagnostic"),
    ]
    support_sizes = tuple(support_sizes)
    _eval_total = len(support_sizes) * len(splits)
    _eval_completed = 0
    report_progress(0, _eval_total, label='Support baseline evaluation blocks')
    for support_size in support_sizes:
        for support_key, query_key, split_name, unit, split_role, source in splits:
            report_step(f"Evaluating support={support_size}, split={split_name}")
            enc = None if int(support_size) == 0 else adapter
            family = "seen_668_no_adapt" if enc is None else "seen_668_support_adapt"
            variant = (
                f"{variant_core}_{split_role}_no_adapt"
                if enc is None
                else f"{variant_core}_{split_role}_support_adapt{support_size}"
            )
            cells, per = r243.evaluate_by_cell(
                r208,
                model,
                enc,
                tables[support_key],
                tables[query_key],
                device,
                variant,
                int(support_size),
                float(shift_scale),
                unit,
                keep_per_unit and split_role == "official_unseen",
            )
            cells = add_cell_count_columns(cells, tables[support_key])
            cells["method_family"] = "support_adapt" if enc is not None else "no_adapt"
            cells["split_role"] = split_role
            cells["variant_core"] = variant_core
            cells_all.append(cells)
            metrics.append(metric_macro_from_cells(cells, variant, split_name, unit, source, family))
            if len(per):
                per["method_family"] = cells["method_family"].iloc[0] if len(cells) else "support_adapt"
                per["split_role"] = split_role
                per["variant"] = variant
                per_all.append(per)
            _eval_completed += 1
            report_progress(_eval_completed, _eval_total, label='Support baseline evaluation blocks')
    return pd.DataFrame(metrics), pd.concat(cells_all, ignore_index=True), pd.concat(per_all, ignore_index=True) if per_all else pd.DataFrame()


def evaluate_query_gated_baseline(
    r247,
    context: dict[str, Any],
    adapter: nn.Module,
    device: torch.device,
    support_sizes: Iterable[int],
    selectors: Iterable[str],
    keep_per_unit: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = []
    cells_all = []
    per_all = []
    splits = [
        ("pseudo_support", "pseudo_val", "pseudo_val", "drug-mean", "pseudo_unseen_validation", "Round247_query_gated_pseudo_validation"),
        ("official_support", "official_test", "official_test", "drug-mean", "official_unseen", "Round247_query_gated_official_unseen"),
        ("official_support", "official_test_profile", "official_test_profile", "true-profile-specific", "official_unseen", "Round247_query_gated_official_unseen"),
        ("seen_support", "seen_test", "seen_test", "drug-mean", "seen_diagnostic", "Round247_query_gated_seen_diagnostic"),
        ("seen_support", "seen_test_profile", "seen_test_profile", "true-profile-specific", "seen_diagnostic", "Round247_query_gated_seen_diagnostic"),
    ]
    support_sizes = tuple(support_sizes)
    selectors = tuple(selectors)
    _eval_total = sum(int(size) > 0 for size in support_sizes) * len(splits) * len(selectors)
    _eval_completed = 0
    report_progress(0, _eval_total, label='Query-gated baseline evaluation blocks')
    for support_size in support_sizes:
        if int(support_size) <= 0:
            continue
        for selector in selectors:
            for support_key, query_key, split_name, unit, split_role, source in splits:
                report_step(f"Evaluating support={support_size}, selector={selector}, split={split_name}")
                variant = (
                    f"{BASE_REFERENCE['round247_query_gated']}_{split_role}"
                    f"_{selector}_s{support_size}"
                )
                metric, cells, per, _audit = r247.evaluate_query_gated(
                    context,
                    adapter,
                    device,
                    split_prefix=split_role,
                    support_key=support_key,
                    query_key=query_key,
                    unit=unit,
                    support_size=int(support_size),
                    selector=str(selector),
                    variant=variant,
                    keep_per_unit=keep_per_unit and split_role == "official_unseen",
                )
                cells = add_cell_count_columns(cells, context["tables"][support_key])
                cells["method_family"] = "query_gated"
                cells["split_role"] = split_role
                cells["variant_core"] = BASE_REFERENCE["round247_query_gated"]
                cells["adapt_support_size"] = int(support_size)
                cells["selector"] = str(selector)
                metric["source"] = source
                metric["method_family"] = "query_gated"
                metric["split_role"] = split_role
                metric["adapt_support_size"] = int(support_size)
                metric["selector"] = str(selector)
                metrics.append(metric)
                cells_all.append(cells)
                if len(per):
                    per["method_family"] = "query_gated"
                    per["split_role"] = split_role
                    per["variant"] = variant
                    per["selector"] = str(selector)
                    per_all.append(per)
                _eval_completed += 1
                report_progress(_eval_completed, _eval_total, label='Query-gated baseline evaluation blocks')
    return (
        pd.concat(metrics, ignore_index=True, sort=False) if metrics else pd.DataFrame(),
        pd.concat(cells_all, ignore_index=True, sort=False) if cells_all else pd.DataFrame(),
        pd.concat(per_all, ignore_index=True, sort=False) if per_all else pd.DataFrame(),
    )


def summarize_by_sample_bin(cell_metrics: pd.DataFrame) -> pd.DataFrame:
    if cell_metrics.empty:
        return pd.DataFrame()
    group_cols = [
        "method_family",
        "variant_core",
        "split_role",
        "split",
        "evaluation_unit",
        "adapt_support_size",
        "sample_bin",
    ]
    rows = []
    for keys, grp in cell_metrics.groupby(group_cols, dropna=False):
        rec = dict(zip(group_cols, keys))
        rec["n_cells"] = int(grp["cell_line"].nunique())
        rec["n_groups"] = int(grp["n_groups"].sum())
        rec["n_profiles"] = int(grp["n_profiles"].sum())
        rec["support_groups_min"] = int(grp["support_groups"].min())
        rec["support_groups_max"] = int(grp["support_groups"].max())
        for metric in METRIC_COLS:
            rec[metric] = float(grp[metric].mean())
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["split_role", "evaluation_unit", "sample_bin", "method_family", "adapt_support_size"])


def paired_delta_summary(cell_metrics: pd.DataFrame, baseline_family: str, method_family: str) -> pd.DataFrame:
    if cell_metrics.empty:
        return pd.DataFrame()
    base = cell_metrics[cell_metrics["method_family"].eq(baseline_family)].copy()
    meth = cell_metrics[cell_metrics["method_family"].eq(method_family)].copy()
    keys = ["split_role", "split", "evaluation_unit", "adapt_support_size", "cell_line"]
    merged = meth.merge(
        base[keys + METRIC_COLS],
        on=keys,
        suffixes=("_method", "_baseline"),
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()
    rows = []
    for group_keys, grp in merged.groupby(["split_role", "split", "evaluation_unit", "adapt_support_size", "sample_bin"], dropna=False):
        rec = dict(zip(["split_role", "split", "evaluation_unit", "adapt_support_size", "sample_bin"], group_keys))
        rec["method_family"] = method_family
        rec["baseline_family"] = baseline_family
        rec["n_cells"] = int(grp["cell_line"].nunique())
        for metric in METRIC_COLS:
            delta = grp[f"{metric}_method"] - grp[f"{metric}_baseline"]
            rec[f"{metric}_mean_delta"] = float(delta.mean())
            rec[f"{metric}_win_cells"] = int((delta > 0).sum())
            rec[f"{metric}_tie_or_win_cells"] = int((delta >= 0).sum())
        rows.append(rec)
    return pd.DataFrame(rows)


def maml_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs = []
    suffix = str(getattr(args, "variant_suffix", "") or "").strip()
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix
    for inner_kind in args.inner_kinds:
        cfg = deepcopy(MAML_DEFAULT)
        cfg["inner_kind"] = inner_kind
        cfg["lambda_meta"] = float(args.lambda_meta)
        cfg["inner_lr"] = float(args.inner_lr)
        cfg["inner_steps"] = int(args.inner_steps)
        cfg["support_size"] = int(args.train_support_size)
        cfg["query_size"] = int(args.train_query_size)
        cfg["tasks_per_epoch"] = int(args.tasks_per_epoch)
        cfg["max_epochs"] = int(args.max_epochs)
        cfg["min_epochs"] = int(args.min_epochs)
        cfg["patience"] = int(args.patience)
        cfg["eval_support_size"] = int(args.eval_support_size)
        cfg["meta_lr"] = float(args.meta_lr)
        cfg["weight_decay"] = float(args.weight_decay)
        cfg["batch_size"] = int(args.batch_size)
        cfg["run_seed"] = int(args.run_seed)
        cfg["variant"] = (
            f"r250_epmeta_{inner_kind}"
            f"_lm{fmt_float(cfg['lambda_meta'])}"
            f"_ilr{fmt_float(cfg['inner_lr'])}"
            f"_mlr{fmt_float(cfg['meta_lr'])}"
            f"_wd{fmt_float(cfg['weight_decay'])}"
            f"_s{cfg['support_size']}_q{cfg['query_size']}"
            f"{suffix}"
        )
        configs.append(cfg)
    return configs


def write_audit(data_audit: dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# Round250 leakage and denominator audit",
        "",
        "- Scope: 668-label closed-set cross-cell episodic meta-learning versus current baselines.",
        "- Round240 full-space target-side is out of scope.",
        "- KG checkpoint is reused; KG is not retrained.",
        "- Official unseen cells are final evaluation only.",
        "- Pseudo-unseen cells are used for meta-learning model selection.",
        "- Episodic support/query training uses drug-disjoint support/query sets.",
        "- Inner-loop updates are limited to query/readout-side parameters; target anchors and target_delta are not updated in the inner loop.",
        "",
        "## Leakage assertions",
        "",
    ]
    for key, value in data_audit["leakage_assertions"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Sample-count bins",
            "",
            "- `small_<=64_support_groups`",
            "- `medium_65_256_support_groups`",
            "- `large_>256_support_groups`",
            "",
            "## Config",
            "",
            "```json",
            json.dumps(json_ready(vars(args)), indent=2),
            "```",
        ]
    )
    (QA / "leakage_and_denominator_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_method_lock(payload: dict[str, Any]) -> None:
    base = {
        "round": "round250_epmeta_stable_tuning",
        "created_at": "2026-05-20",
        "seed": SEED,
        "scope": "668-label closed-set current-model stable episodic meta-learning tuning versus current baselines; no Round240 full-space target-side",
        "selection": "Meta-learning selected by pseudo-unseen support-adapt equivalent Hit@10, then MeanAP, NDCG@10, meta-val no-adapt Hit@10",
        "sample_bins": {
            "small_<=64_support_groups": "support groups <= 64",
            "medium_65_256_support_groups": "65 <= support groups <= 256",
            "large_>256_support_groups": "support groups > 256",
        },
        "artifacts": payload,
    }
    text = json.dumps(json_ready(base), indent=2)
    (ROUND_ROOT / "method_lock.json").write_text(text, encoding="utf-8")
    (SOURCE / "method_lock.json").write_text(text, encoding="utf-8")


def write_report(metrics: pd.DataFrame, by_bin: pd.DataFrame, deltas: pd.DataFrame) -> None:
    official = metrics[
        (metrics["split"].eq("official_test"))
        & (metrics["evaluation_unit"].eq("drug-mean"))
        & (metrics["adapt_support_size"].isin([0, 32]))
    ].copy()
    official_text = official[
        ["source", "variant", "evaluation_family", "adapt_support_size", "n_cells", "Hit@10", "MeanAP", "NDCG@10", "AUROC", "Hit@20", "Hit@50"]
    ].to_csv(sep="\t", index=False).strip() if len(official) else "No official metrics."
    by_bin_text = by_bin[
        (by_bin["split_role"].eq("official_unseen"))
        & (by_bin["evaluation_unit"].eq("drug-mean"))
        & (by_bin["adapt_support_size"].isin([0, 32]))
    ].to_csv(sep="\t", index=False).strip() if len(by_bin) else "No sample-bin metrics."
    delta_text = deltas[
        (deltas["split_role"].eq("official_unseen"))
        & (deltas["evaluation_unit"].eq("drug-mean"))
        & (deltas["adapt_support_size"].eq(32))
    ].to_csv(sep="\t", index=False).strip() if len(deltas) else "No paired deltas."
    lines = [
        "# Round250 stable episodic meta-learning tuning comparison",
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
        by_bin_text,
        "```",
        "",
        "## Paired deltas",
        "",
        "```tsv",
        delta_text,
        "```",
        "",
        "## Interpretation guardrails",
        "",
        "- Use official unseen cells as final evidence only.",
        "- Treat support-size bins as descriptive stratification, not a new model-selection rule.",
        "- If gains concentrate at Hit@50 / TopK-style metrics, describe candidate-list enrichment rather than uniform target recovery.",
    ]
    (ROUND_ROOT / "round250_result_report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (SOURCE / "round250_result_report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(args: argparse.Namespace, device: torch.device) -> None:
    r243, _r211, _r208, _labels, _kg_path, _kg_ckpt, tables, data_audit = load_context(args, device)
    r243.write_round243_audit_tables(tables, data_audit)
    (SOURCE / "round250_protocol_audit.json").write_text(json.dumps(json_ready(data_audit), indent=2), encoding="utf-8")
    write_audit(data_audit, args)
    write_method_lock({"data_audit": data_audit, "status": "audit_only"})


def run_training(args: argparse.Namespace, device: torch.device) -> None:
    r243, r211, r208, labels, kg_path, kg_ckpt, tables, data_audit = load_context(args, device)
    r243.write_round243_audit_tables(tables, data_audit)
    (SOURCE / "round250_protocol_audit.json").write_text(json.dumps(json_ready(data_audit), indent=2), encoding="utf-8")
    write_audit(data_audit, args)

    all_metrics = []
    all_cells = []
    all_per = []
    all_history = []
    selected_payloads = []
    current_initializer = resolve_current_checkpoint(BASE_REFERENCE["round246_current_joint"], None)
    for cfg in maml_configs(args):
        log(f"training {cfg['variant']} smoke={args.smoke}")
        model, history, selected = train_maml_variant(
            r243,
            r211,
            r208,
            kg_ckpt,
            labels,
            tables,
            device,
            cfg,
            initializer_checkpoint=current_initializer,
            smoke=bool(args.smoke),
        )
        history.to_csv(SOURCE / f"training_history_{cfg['variant']}.tsv", sep="	", index=False)
        all_history.append(history)
        selected_payloads.append(selected)
        metrics, cells, per = evaluate_maml_model(
            r243,
            r208,
            model,
            tables,
            device,
            selected,
            args.report_support_sizes,
            keep_per_unit=True,
        )
        metrics.to_csv(SOURCE / f"metric_summary_{cfg['variant']}.tsv", sep="	", index=False)
        cells.to_csv(SOURCE / f"per_cell_metrics_{cfg['variant']}.tsv", sep="	", index=False)
        all_metrics.append(metrics)
        all_cells.append(cells)
        if len(per):
            per.to_csv(SOURCE / f"per_unit_metrics_{cfg['variant']}.tsv", sep="	", index=False)
            all_per.append(per)

    support_ckpt = None
    support_payload = None
    query_gated_ckpt = None
    if not args.skip_reference_baselines:
        support_ckpt = resolve_current_checkpoint(args.support_baseline_variant, args.support_checkpoint)
        support_model, support_adapter, support_payload = load_current_support_baseline(r243, r211, kg_ckpt, labels, support_ckpt, device)
        support_metrics, support_cells, support_per = evaluate_support_baseline(
            r243,
            r208,
            support_model,
            support_adapter,
            tables,
            device,
            str(support_payload["joint_cfg"]["variant"]),
            float(support_payload["joint_cfg"]["shift_scale"]),
            args.report_support_sizes,
            keep_per_unit=True,
        )
        support_metrics.to_csv(SOURCE / f"metric_summary_support_baseline_{support_payload['joint_cfg']['variant']}.tsv", sep="	", index=False)
        support_cells.to_csv(SOURCE / f"per_cell_metrics_support_baseline_{support_payload['joint_cfg']['variant']}.tsv", sep="	", index=False)
        all_metrics.append(support_metrics)
        all_cells.append(support_cells)
        if len(support_per):
            support_per.to_csv(SOURCE / f"per_unit_metrics_support_baseline_{support_payload['joint_cfg']['variant']}.tsv", sep="	", index=False)
            all_per.append(support_per)

        query_gated_ckpt = resolve_query_gated_checkpoint(args.query_gated_checkpoint)
        if query_gated_ckpt is not None:
            r247 = load_round247_module()
            for key in [
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
            query_adapter, query_payload = load_query_gated_baseline(r247, support_model, query_gated_ckpt, device)
            query_metrics, query_cells, query_per = evaluate_query_gated_baseline(
                r247,
                {"r247": r247, "r243": r243, "r208": r208, "model": support_model, "tables": tables},
                query_adapter,
                device,
                support_sizes=args.report_support_sizes,
                selectors=["top_count", "random"],
                keep_per_unit=True,
            )
            query_metrics.to_csv(SOURCE / f"metric_summary_query_gated_{query_payload['cfg']['variant']}.tsv", sep="	", index=False)
            query_cells.to_csv(SOURCE / f"per_cell_metrics_query_gated_{query_payload['cfg']['variant']}.tsv", sep="	", index=False)
            if len(query_per):
                query_per.to_csv(SOURCE / f"per_unit_metrics_query_gated_{query_payload['cfg']['variant']}.tsv", sep="	", index=False)
                all_per.append(query_per)
            all_metrics.append(query_metrics)
            all_cells.append(query_cells)

    metrics_all = pd.concat(all_metrics, ignore_index=True, sort=False)
    cells_all = pd.concat(all_cells, ignore_index=True, sort=False)
    history_all = pd.concat(all_history, ignore_index=True, sort=False) if all_history else pd.DataFrame()
    per_all = pd.concat(all_per, ignore_index=True, sort=False) if all_per else pd.DataFrame()
    by_bin = summarize_by_sample_bin(cells_all)
    deltas = pd.concat(
        [
            paired_delta_summary(cells_all, "no_adapt", "maml"),
            paired_delta_summary(cells_all, "no_adapt", "support_adapt"),
            paired_delta_summary(cells_all, "support_adapt", "maml"),
        ],
        ignore_index=True,
        sort=False,
    )

    metrics_all.to_csv(SOURCE / "metric_summary.tsv", sep="	", index=False)
    cells_all.to_csv(SOURCE / "per_cell_metrics.tsv", sep="	", index=False)
    by_bin.to_csv(SOURCE / "metric_summary_by_sample_bin.tsv", sep="	", index=False)
    deltas.to_csv(SOURCE / "paired_delta_by_sample_bin.tsv", sep="	", index=False)
    if len(history_all):
        history_all.to_csv(SOURCE / "training_history.tsv", sep="	", index=False)
    if len(per_all):
        per_all.to_csv(SOURCE / "per_unit_metrics.tsv", sep="	", index=False)
    write_method_lock(
        {
            "kg_checkpoint": str(kg_path),
            "kg_checkpoint_sha256": sha256_file(kg_path),
            "data_audit": data_audit,
            "maml_selected": selected_payloads,
            "support_baseline_checkpoint": str(support_ckpt) if support_ckpt is not None else None,
            "support_baseline_sha256": sha256_file(support_ckpt) if support_ckpt is not None else None,
            "support_baseline_variant": support_payload["joint_cfg"]["variant"] if support_payload is not None else None,
            "query_gated_checkpoint": str(query_gated_ckpt) if query_gated_ckpt is not None else None,
            "skip_reference_baselines": bool(args.skip_reference_baselines),
        }
    )
    write_report(metrics_all, by_bin, deltas)
    log(f"done; outputs written to {ROUND_ROOT}")

def run_existing_eval(args: argparse.Namespace, device: torch.device) -> None:
    r243, r211, r208, labels, kg_path, kg_ckpt, tables, data_audit = load_context(args, device)
    all_metrics = []
    all_cells = []
    all_per = []
    selected_payloads = []
    for cfg in maml_configs(args):
        selected_path = SOURCE / f"selected_{cfg['variant']}.json"
        if not selected_path.exists():
            raise FileNotFoundError(selected_path)
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        model = load_maml_model(r211, kg_ckpt, labels, selected, device)
        selected_payloads.append(selected)
        metrics, cells, per = evaluate_maml_model(
            r243,
            r208,
            model,
            tables,
            device,
            selected,
            args.report_support_sizes,
            keep_per_unit=True,
        )
        all_metrics.append(metrics)
        all_cells.append(cells)
        if len(per):
            all_per.append(per)

    support_ckpt = resolve_current_checkpoint(args.support_baseline_variant, args.support_checkpoint)
    support_model, support_adapter, support_payload = load_current_support_baseline(r243, r211, kg_ckpt, labels, support_ckpt, device)
    support_metrics, support_cells, support_per = evaluate_support_baseline(
        r243,
        r208,
        support_model,
        support_adapter,
        tables,
        device,
        str(support_payload["joint_cfg"]["variant"]),
        float(support_payload["joint_cfg"]["shift_scale"]),
        args.report_support_sizes,
        keep_per_unit=True,
    )
    all_metrics.append(support_metrics)
    all_cells.append(support_cells)
    if len(support_per):
        all_per.append(support_per)

    query_gated_ckpt = resolve_query_gated_checkpoint(args.query_gated_checkpoint)
    if query_gated_ckpt is not None:
        r247 = load_round247_module()
        for key in [
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
        query_adapter, query_payload = load_query_gated_baseline(r247, support_model, query_gated_ckpt, device)
        query_metrics, query_cells, query_per = evaluate_query_gated_baseline(
            r247,
            {"r247": r247, "r243": r243, "r208": r208, "model": support_model, "tables": tables},
            query_adapter,
            device,
            support_sizes=args.report_support_sizes,
            selectors=["top_count", "random"],
            keep_per_unit=True,
        )
        all_metrics.append(query_metrics)
        all_cells.append(query_cells)
        if len(query_per):
            all_per.append(query_per)

    metrics_all = pd.concat(all_metrics, ignore_index=True, sort=False)
    cells_all = pd.concat(all_cells, ignore_index=True, sort=False)
    per_all = pd.concat(all_per, ignore_index=True, sort=False) if all_per else pd.DataFrame()
    by_bin = summarize_by_sample_bin(cells_all)
    deltas = pd.concat(
        [
            paired_delta_summary(cells_all, "no_adapt", "maml"),
            paired_delta_summary(cells_all, "no_adapt", "support_adapt"),
            paired_delta_summary(cells_all, "support_adapt", "maml"),
        ],
        ignore_index=True,
        sort=False,
    )

    metrics_all.to_csv(SOURCE / "metric_summary_existing_eval.tsv", sep="	", index=False)
    cells_all.to_csv(SOURCE / "per_cell_metrics_existing_eval.tsv", sep="	", index=False)
    by_bin.to_csv(SOURCE / "metric_summary_by_sample_bin_existing_eval.tsv", sep="	", index=False)
    deltas.to_csv(SOURCE / "paired_delta_by_sample_bin_existing_eval.tsv", sep="	", index=False)
    if len(per_all):
        per_all.to_csv(SOURCE / "per_unit_metrics_existing_eval.tsv", sep="	", index=False)

    seen_profile = by_bin[
        by_bin["split_role"].eq("seen_diagnostic")
        & by_bin["split"].eq("seen_test_profile")
        & by_bin["evaluation_unit"].eq("true-profile-specific")
    ].copy()
    seen_profile.to_csv(SOURCE / "seen_diagnostic_profile_by_sample_bin.tsv", sep="	", index=False)
    write_method_lock(
        {
            "kg_checkpoint": str(kg_path),
            "kg_checkpoint_sha256": sha256_file(kg_path),
            "data_audit": data_audit,
            "maml_selected": selected_payloads,
            "support_baseline_checkpoint": str(support_ckpt),
            "support_baseline_sha256": sha256_file(support_ckpt),
            "support_baseline_variant": support_payload["joint_cfg"]["variant"],
            "query_gated_checkpoint": str(query_gated_ckpt) if query_gated_ckpt is not None else None,
            "existing_eval_outputs": [
                "metric_summary_existing_eval.tsv",
                "metric_summary_by_sample_bin_existing_eval.tsv",
                "seen_diagnostic_profile_by_sample_bin.tsv",
            ],
        }
    )
    log(f"existing checkpoint evaluation done; outputs written to {SOURCE}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["audit_protocol", "smoke", "train", "eval_existing"], default="train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--kg-checkpoint", default=None)
    parser.add_argument("--pseudo-unseen-cells", default=",".join(DEFAULT_R250_PSEUDO_UNSEEN_VAL_CELLS))
    parser.add_argument("--official-unseen-cells", default="round215_heldout_panel")
    parser.add_argument("--seen-diagnostic-cells", default=",".join(FORCED_TRAIN_CELLS))
    parser.add_argument("--inner-kinds", nargs="+", default=["query_tail", "query_bias"])
    parser.add_argument("--lambda-meta", type=float, default=1.0)
    parser.add_argument("--inner-lr", type=float, default=4.0e-3)
    parser.add_argument("--inner-steps", type=int, default=1)
    parser.add_argument("--train-support-size", type=int, default=32)
    parser.add_argument("--train-query-size", type=int, default=32)
    parser.add_argument("--eval-support-size", type=int, default=32)
    parser.add_argument("--report-support-sizes", nargs="+", type=int, default=[0, 8, 16, 32, 64])
    parser.add_argument("--tasks-per-epoch", type=int, default=120)
    parser.add_argument("--max-epochs", type=int, default=22)
    parser.add_argument("--min-epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--meta-lr", type=float, default=MAML_DEFAULT["meta_lr"])
    parser.add_argument("--weight-decay", type=float, default=MAML_DEFAULT["weight_decay"])
    parser.add_argument("--batch-size", type=int, default=MAML_DEFAULT["batch_size"])
    parser.add_argument("--run-seed", type=int, default=SEED)
    parser.add_argument("--variant-suffix", default="")
    parser.add_argument("--skip-reference-baselines", action="store_true")
    parser.add_argument("--support-baseline-variant", default=BASE_REFERENCE["round246_current_joint"])
    parser.add_argument("--support-checkpoint", default=None)
    parser.add_argument("--query-gated-checkpoint", default=None)
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
    set_seed(int(args.run_seed))
    device = torch.device(resolve_device(args.device))
    log(f"using device={device}")
    if args.phase == "audit_protocol":
        run_audit(args, device)
    elif args.phase == "eval_existing":
        run_existing_eval(args, device)
    else:
        run_training(args, device)


if __name__ == "__main__":
    main()
