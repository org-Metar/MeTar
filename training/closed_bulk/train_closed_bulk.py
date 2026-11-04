#!/usr/bin/env python3
"""Joint training of the bulk response readout and support-conditioned adapter."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from metar.runtime import report_metrics, report_progress, report_step, resolve_device

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    F = None


ROUND_ROOT = Path(__file__).resolve().parents[1]
METAR_ROOT = ROUND_ROOT.parent
SOURCE = ROUND_ROOT / "source_data"
QA = ROUND_ROOT / "qa"
CHECKPOINTS = ROUND_ROOT / "checkpoints"
LOGS = ROUND_ROOT / "logs"

ROUND243_ROOT = METAR_ROOT / "claude_round243_leakage_safe_unseen_cell_support_adapter_20260519"
ROUND243_SCRIPT = ROUND243_ROOT / "scripts" / "run_round243_leakage_safe_unseen_cell_support_adapter.py"

SEED = 20260519
BASE_REFERENCE_HIT10 = 0.40967997424229485

READOUT_CFG = {
    "readout_id": "r246_joint_round211_h256_seen668",
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

JOINT_DEFAULT = {
    "variant": "r246_joint_lm0p5_sc0p15_mn2p5",
    "lambda_meta": 0.5,
    "shift_scale": 0.15,
    "max_delta_norm": 2.5,
    "delta_l2": 0.001,
    "support_hidden_dim": 128,
    "support_dropout": 0.1,
    "support_size": 32,
    "query_size": 32,
    "tasks_per_epoch": 240,
    "adapter_lr": 2.0e-4,
    "adapter_weight_decay": 1.0e-5,
}


def require_torch() -> None:
    if torch is None or F is None:
        raise ModuleNotFoundError("torch is required; run training on the remote PertKGE/MeTar environment.")


FORCED_TRAIN_CELLS = ["PC3", "A375", "MCF7"]
DEFAULT_R246_PSEUDO_UNSEEN_VAL_CELLS = ["HA1E", "HEK293", "MCF10A", "MDAMB231", "PHH", "THP1"]


def log(message: str) -> None:
    print(f"[round246] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


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


def load_round243():
    module = import_from_path("round246_round243", ROUND243_SCRIPT)
    module.ROUND_ROOT = ROUND_ROOT
    module.SOURCE = SOURCE
    module.QA = QA
    module.CHECKPOINTS = CHECKPOINTS
    module.LOGS = LOGS
    return module


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
        raise RuntimeError(f"Expected 668 unique target labels, observed {len(labels)} / {len(set(labels))}")
    cell_sets = r243.choose_round243_cells(
        r208,
        args.pseudo_unseen_cells,
        args.official_unseen_cells,
        args.seen_diagnostic_cells,
    )
    missing_forced = [
        cell
        for cell in FORCED_TRAIN_CELLS
        if cell not in set(cell_sets["base_train_cells"]) or cell not in set(cell_sets["adapter_train_cells"])
    ]
    if missing_forced:
        raise RuntimeError(
            "Forced training cells are not all in base_train/adapter_train: "
            + ", ".join(missing_forced)
        )
    tables, data_audit = r243.build_round243_tables(r208, labels, cell_sets, device)
    for item_name, item in tables.items():
        if item["x"].shape[1] != 978:
            raise RuntimeError(f"{item_name} expected 978 input features, observed {item['x'].shape[1]}")
        if item["y"].shape[1] != 668:
            raise RuntimeError(f"{item_name} expected 668 labels, observed {item['y'].shape[1]}")
    return r243, r211, r208, labels, kg_path, kg_ckpt, tables, data_audit


def fmt_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def variant_cfg(lambda_meta: float, shift_scale: float, max_delta_norm: float) -> dict[str, Any]:
    cfg = deepcopy(JOINT_DEFAULT)
    cfg["lambda_meta"] = float(lambda_meta)
    cfg["shift_scale"] = float(shift_scale)
    cfg["max_delta_norm"] = float(max_delta_norm)
    cfg["variant"] = (
        f"r246_joint_lm{fmt_float(lambda_meta)}"
        f"_sc{fmt_float(shift_scale)}_mn{fmt_float(max_delta_norm)}"
    )
    return cfg


def selected_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.phase == "sweep":
        cfgs = []
        for lm in args.sweep_lambda_meta:
            for sc in args.sweep_shift_scales:
                for mn in args.sweep_max_delta_norms:
                    cfgs.append(variant_cfg(lm, sc, mn))
        return cfgs
    if args.runs == "sweep":
        return selected_configs(argparse.Namespace(**{**vars(args), "phase": "sweep"}))
    return [variant_cfg(args.lambda_meta, args.shift_scale, args.max_delta_norm)]


def eligible_cells(item: dict, support_size: int, query_size: int) -> list[str]:
    rows = item["rows"]
    cells = []
    for cell, idx in item["by_cell"].items():
        drugs = rows.iloc[np.asarray(idx, dtype=np.int64)]["drug"].astype(str).nunique()
        if drugs >= support_size + query_size:
            cells.append(str(cell))
    return cells


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
        raise RuntimeError(f"Cell {cell} has too few drugs for drug-disjoint episode: {len(drugs)}")
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
    if set(rows.iloc[support_idx]["drug"].astype(str)).intersection(set(rows.iloc[query_idx]["drug"].astype(str))):
        raise RuntimeError("Drug leakage inside sampled episode")
    return np.asarray(support_idx, dtype=np.int64), np.asarray(query_idx, dtype=np.int64)


def tensor_slice(item: dict, key: str, idx: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(item[key][idx], dtype=torch.float32, device=device)


def evaluate_joint(
    r243,
    r208,
    model,
    adapter,
    tables: dict[str, dict],
    device: torch.device,
    cfg: dict[str, Any],
    keep_per_unit: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    per_parts = []

    official_no, official_no_per = r243.evaluate_by_cell(
        r208,
        model,
        None,
        tables["official_support"],
        tables["official_test"],
        device,
        f"{cfg['variant']}_official_unseen_no_adapt",
        int(cfg["support_size"]),
        float(cfg["shift_scale"]),
        "drug-mean",
        keep_per_unit,
    )
    metrics.append(
        r243.metric_macro_from_cells(
            official_no,
            f"{cfg['variant']}_official_unseen_no_adapt",
            "official_test",
            "drug-mean",
            "Round246_joint_official_unseen",
        )
    )
    if len(official_no_per):
        per_parts.append(official_no_per.assign(claim_scope="unseen_cell_main", split_role="official_unseen", variant=f"{cfg['variant']}_official_unseen_no_adapt"))

    official_ad, official_ad_per = r243.evaluate_by_cell(
        r208,
        model,
        adapter,
        tables["official_support"],
        tables["official_test"],
        device,
        f"{cfg['variant']}_official_unseen_support_adapt",
        int(cfg["support_size"]),
        float(cfg["shift_scale"]),
        "drug-mean",
        keep_per_unit,
    )
    metrics.append(
        r243.metric_macro_from_cells(
            official_ad,
            f"{cfg['variant']}_official_unseen_support_adapt",
            "official_test",
            "drug-mean",
            "Round246_joint_official_unseen",
        )
    )
    if len(official_ad_per):
        per_parts.append(official_ad_per.assign(claim_scope="unseen_cell_main", split_role="official_unseen", variant=f"{cfg['variant']}_official_unseen_support_adapt"))

    official_profile, official_profile_per = r243.evaluate_by_cell(
        r208,
        model,
        adapter,
        tables["official_support"],
        tables["official_test_profile"],
        device,
        f"{cfg['variant']}_official_unseen_support_adapt_profile",
        int(cfg["support_size"]),
        float(cfg["shift_scale"]),
        "true-profile-specific",
        keep_per_unit,
    )
    metrics.append(
        r243.metric_macro_from_cells(
            official_profile,
            f"{cfg['variant']}_official_unseen_support_adapt_profile",
            "official_test_profile",
            "true-profile-specific",
            "Round246_joint_official_unseen",
        )
    )
    if len(official_profile_per):
        per_parts.append(official_profile_per.assign(claim_scope="unseen_cell_main", split_role="official_unseen", variant=f"{cfg['variant']}_official_unseen_support_adapt_profile"))

    pseudo_ad, _ = r243.evaluate_by_cell(
        r208,
        model,
        adapter,
        tables["pseudo_support"],
        tables["pseudo_val"],
        device,
        f"{cfg['variant']}_pseudo_val_support_adapt",
        int(cfg["support_size"]),
        float(cfg["shift_scale"]),
        "drug-mean",
        False,
    )
    metrics.append(
        r243.metric_macro_from_cells(
            pseudo_ad,
            f"{cfg['variant']}_pseudo_val_support_adapt",
            "pseudo_val",
            "drug-mean",
            "Round246_joint_pseudo_validation",
        )
    )

    seen_no, _ = r243.evaluate_by_cell(
        r208,
        model,
        None,
        tables["seen_support"],
        tables["seen_test"],
        device,
        f"{cfg['variant']}_seen_diagnostic_no_adapt",
        int(cfg["support_size"]),
        float(cfg["shift_scale"]),
        "drug-mean",
        False,
    )
    metrics.append(
        r243.metric_macro_from_cells(
            seen_no,
            f"{cfg['variant']}_seen_diagnostic_no_adapt",
            "seen_test",
            "drug-mean",
            "Round246_joint_seen_diagnostic",
        )
    )

    seen_ad, _ = r243.evaluate_by_cell(
        r208,
        model,
        adapter,
        tables["seen_support"],
        tables["seen_test"],
        device,
        f"{cfg['variant']}_seen_diagnostic_support_adapt",
        int(cfg["support_size"]),
        float(cfg["shift_scale"]),
        "drug-mean",
        False,
    )
    metrics.append(
        r243.metric_macro_from_cells(
            seen_ad,
            f"{cfg['variant']}_seen_diagnostic_support_adapt",
            "seen_test",
            "drug-mean",
            "Round246_joint_seen_diagnostic",
        )
    )

    out = pd.DataFrame(metrics)
    out["stage"] = "round246_joint_seen_target_train_pc3_a375_mcf7"
    out["claim_scope"] = out["split"].map(
        {
            "official_test": "unseen_cell_main",
            "official_test_profile": "unseen_cell_main",
            "pseudo_val": "model_selection_diagnostic",
            "seen_test": "seen_cell_diagnostic",
        }
    ).fillna("diagnostic")
    out["lambda_meta"] = float(cfg["lambda_meta"])
    out["shift_scale"] = float(cfg["shift_scale"])
    out["max_delta_norm"] = float(cfg["max_delta_norm"])
    per = pd.concat(per_parts, ignore_index=True, sort=False) if per_parts else pd.DataFrame()
    return out, per


def selection_record(
    r243,
    r208,
    model,
    adapter,
    tables: dict[str, dict],
    device: torch.device,
    cfg: dict[str, Any],
    epoch: int,
    loss_parts: dict[str, float],
) -> dict[str, Any]:
    r243.precompute_q(model, tables["pseudo_support"], device)
    r243.precompute_q(model, tables["pseudo_val"], device)
    pseudo_cells, _ = r243.evaluate_by_cell(
        r208,
        model,
        adapter,
        tables["pseudo_support"],
        tables["pseudo_val"],
        device,
        f"{cfg['variant']}_pseudo_val_support_adapt",
        int(cfg["support_size"]),
        float(cfg["shift_scale"]),
        "drug-mean",
        False,
    )
    pseudo = r243.metric_macro_from_cells(
        pseudo_cells,
        f"{cfg['variant']}_pseudo_val_support_adapt",
        "pseudo_val",
        "drug-mean",
        "Round246_joint_pseudo_validation",
    )
    meta_val_scores = r243.predict_from_q_model(model, tables["meta_val"]["x"], device, batch_size=1024)
    meta_val, _ = r208.compute_metrics(
        meta_val_scores,
        tables["meta_val"]["y"],
        tables["meta_val"]["rows"],
        "Round246_joint",
        f"{cfg['variant']}_meta_val_no_adapt",
        "drug-mean",
    )
    return {
        "variant": cfg["variant"],
        "epoch": int(epoch),
        "train_loss": float(loss_parts["loss"]),
        "base_loss": float(loss_parts["base_loss"]),
        "episode_loss": float(loss_parts["episode_loss"]),
        "delta_norm": float(loss_parts["delta_norm"]),
        "pseudo_val_Hit@10": float(pseudo["Hit@10"]),
        "pseudo_val_MeanAP": float(pseudo["MeanAP"]),
        "pseudo_val_NDCG@10": float(pseudo["NDCG@10"]),
        "pseudo_val_AUROC": float(pseudo["AUROC"]),
        "meta_val_noadapt_Hit@10": float(meta_val["Hit@10"]),
        "meta_val_noadapt_MeanAP": float(meta_val["MeanAP"]),
        "lambda_meta": float(cfg["lambda_meta"]),
        "shift_scale": float(cfg["shift_scale"]),
        "max_delta_norm": float(cfg["max_delta_norm"]),
    }


def better_candidate(candidate: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    if incumbent is None:
        return True
    keys = ["pseudo_val_Hit@10", "pseudo_val_MeanAP", "pseudo_val_NDCG@10", "meta_val_noadapt_Hit@10"]
    for key in keys:
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


def checkpoint_reload_audit(
    r243,
    r211,
    target_init: np.ndarray,
    readout_cfg: dict[str, Any],
    cfg: dict[str, Any],
    ckpt_path: Path,
    model,
    adapter,
    tables: dict[str, dict],
    device: torch.device,
) -> dict[str, float]:
    report_step('Verifying saved joint checkpoint')
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    reloaded_model = r211.DiagReadout(target_init, readout_cfg).to(device)
    reloaded_adapter = r243.SupportDeltaEncoder(
        reloaded_model.target_anchor.detach(),
        hidden_dim=int(cfg["support_hidden_dim"]),
        dropout=float(cfg["support_dropout"]),
        max_delta_norm=float(cfg["max_delta_norm"]),
    ).to(device)
    reloaded_model.load_state_dict(payload["readout_state_dict"], strict=True)
    reloaded_adapter.load_state_dict(payload["support_adapter_state_dict"], strict=True)
    model.eval()
    adapter.eval()
    reloaded_model.eval()
    reloaded_adapter.eval()
    with torch.no_grad():
        x = torch.tensor(tables["meta_val"]["x"][: min(16, len(tables["meta_val"]["x"]))], dtype=torch.float32, device=device)
        noadapt_diff = torch.max(torch.abs(model(x) - reloaded_model(x))).detach().cpu().item() if len(x) else 0.0
        common_cells = sorted(set(tables["pseudo_support"]["by_cell"]).intersection(tables["pseudo_val"]["by_cell"]))
        adapt_diff = 0.0
        if common_cells:
            cell = common_cells[0]
            support_all = np.asarray(tables["pseudo_support"]["by_cell"][cell], dtype=np.int64)
            query_all = np.asarray(tables["pseudo_val"]["by_cell"][cell], dtype=np.int64)[:16]
            local = r243.deterministic_support_indices(tables["pseudo_support"]["rows"].iloc[support_all], int(cfg["support_size"]))
            support_idx = support_all[local]
            support_x = tensor_slice(tables["pseudo_support"], "x", support_idx, device)
            support_y = tensor_slice(tables["pseudo_support"], "y", support_idx, device)
            support_w = tensor_slice(tables["pseudo_support"], "weights", support_idx, device)
            query_x = tensor_slice(tables["pseudo_val"], "x", query_all, device)
            q1 = r243.encode_q(model, support_x)
            q2 = r243.encode_q(reloaded_model, support_x)
            d1 = adapter(q1, support_y, support_w)
            d2 = reloaded_adapter(q2, support_y, support_w)
            s1 = r243.score_from_q(model, r243.encode_q(model, query_x), delta=d1, scale=float(cfg["shift_scale"]))
            s2 = r243.score_from_q(reloaded_model, r243.encode_q(reloaded_model, query_x), delta=d2, scale=float(cfg["shift_scale"]))
            adapt_diff = torch.max(torch.abs(s1 - s2)).detach().cpu().item() if len(query_all) else 0.0
    return {
        "reload_noadapt_logits_max_abs_diff": float(noadapt_diff),
        "reload_support_adapt_logits_max_abs_diff": float(adapt_diff),
    }


def train_joint_variant(
    r243,
    r211,
    r208,
    kg_ckpt: dict[str, Any],
    labels: list[str],
    tables: dict[str, dict],
    device: torch.device,
    cfg: dict[str, Any],
    smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    report_step('Preparing joint bulk model and support adapter')
    readout_cfg = deepcopy(READOUT_CFG)
    readout_cfg["readout_id"] = cfg["variant"]
    if smoke:
        readout_cfg["max_epochs"] = 1
        readout_cfg["min_epochs"] = 1
        readout_cfg["patience"] = 1
        cfg = deepcopy(cfg)
        cfg["tasks_per_epoch"] = 4
    target_init, target_audit = r211.target_embeddings_from_ckpt(kg_ckpt, labels, cfg["variant"])
    model = r211.DiagReadout(target_init, readout_cfg).to(device)
    adapter = r243.SupportDeltaEncoder(
        model.target_anchor.detach(),
        hidden_dim=int(cfg["support_hidden_dim"]),
        dropout=float(cfg["support_dropout"]),
        max_delta_norm=float(cfg["max_delta_norm"]),
    ).to(device)
    opt = torch.optim.AdamW(
        [
            {"params": list(model.parameters()), "lr": float(readout_cfg["lr"]), "weight_decay": float(readout_cfg["weight_decay"])},
            {"params": list(adapter.parameters()), "lr": float(cfg["adapter_lr"]), "weight_decay": float(cfg["adapter_weight_decay"])},
        ]
    )

    train = tables["meta_train"]
    adapter_train = tables["adapter_train"]
    train_tensors = {
        "x": torch.tensor(train["x"], dtype=torch.float32, device=device),
        "y": torch.tensor(train["y"], dtype=torch.float32, device=device),
        "w": torch.tensor(train["weights"], dtype=torch.float32, device=device),
    }
    support_size = int(cfg["support_size"])
    query_size = int(cfg["query_size"])
    cells = eligible_cells(adapter_train, support_size, query_size)
    if not cells:
        raise RuntimeError("No eligible adapter-train cells for drug-disjoint joint episodes")

    rng = np.random.default_rng(SEED)
    best: dict[str, Any] | None = None
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    n_train = int(train_tensors["x"].shape[0])
    batch_size = int(readout_cfg["batch_size"])
    max_epochs = int(readout_cfg["max_epochs"])
    min_epochs = int(readout_cfg["min_epochs"])
    patience = int(readout_cfg["patience"])
    tasks_per_epoch = int(cfg["tasks_per_epoch"])

    report_progress(0, max_epochs, label='joint variant epochs')
    report_metrics(cfg, label="Training settings")
    for epoch in range(1, max_epochs + 1):
        model.train()
        adapter.train()
        losses = []
        base_losses = []
        episode_losses = []
        delta_norms = []
        for _ in range(tasks_per_epoch):
            report_progress(_, tasks_per_epoch, label=f"Epoch {epoch} training episodes")
            base_idx = torch.tensor(rng.integers(0, n_train, size=batch_size), dtype=torch.long, device=device)
            xb = train_tensors["x"][base_idx]
            yb = train_tensors["y"][base_idx]
            wb = train_tensors["w"][base_idx]
            base_logits = model(xb)
            base_loss = r243.retrieval_loss_from_logits(base_logits, yb, wb)

            cell = str(rng.choice(cells))
            support_idx, query_idx = sample_drug_disjoint_episode(adapter_train, cell, support_size, query_size, rng)
            support_x = tensor_slice(adapter_train, "x", support_idx, device)
            support_y = tensor_slice(adapter_train, "y", support_idx, device)
            support_w = tensor_slice(adapter_train, "weights", support_idx, device)
            query_x = tensor_slice(adapter_train, "x", query_idx, device)
            query_y = tensor_slice(adapter_train, "y", query_idx, device)
            query_w = tensor_slice(adapter_train, "weights", query_idx, device)

            support_q = r243.encode_q(model, support_x)
            delta = adapter(support_q, support_y, support_w)
            query_q = r243.encode_q(model, query_x)
            episode_logits = r243.score_from_q(model, query_q, delta=delta, scale=float(cfg["shift_scale"]))
            episode_loss = r243.retrieval_loss_from_logits(episode_logits, query_y, query_w)
            delta_l2 = float(cfg["delta_l2"]) * torch.square(delta).mean()
            loss = base_loss + float(cfg["lambda_meta"]) * episode_loss + model.regularization() + delta_l2

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(adapter.parameters()), 2.0)
            opt.step()

            losses.append(float(loss.detach().cpu()))
            base_losses.append(float(base_loss.detach().cpu()))
            episode_losses.append(float(episode_loss.detach().cpu()))
            delta_norms.append(float(torch.linalg.vector_norm(delta).detach().cpu()))
        report_progress(tasks_per_epoch, tasks_per_epoch, label=f"Epoch {epoch} training episodes")

        loss_parts = {
            "loss": float(np.mean(losses)),
            "base_loss": float(np.mean(base_losses)),
            "episode_loss": float(np.mean(episode_losses)),
            "delta_norm": float(np.mean(delta_norms)),
        }
        report_step(f"Epoch {epoch}: evaluating existing training and validation metrics")
        rec = selection_record(r243, r208, model, adapter, tables, device, cfg, epoch, loss_parts)
        history.append(rec)
        report_progress(epoch, max_epochs, label='joint variant epochs')
        report_metrics(
            {key: rec[key] for key in ("train_loss", "pseudo_val_Hit@10", "pseudo_val_MeanAP")},
            label=f"Epoch {epoch}/{max_epochs}",
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
            best_epoch = epoch
            best_state = {
                "readout": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "support_adapter": {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()},
            }
            stale = 0
        else:
            stale += 1
        if epoch >= min_epochs and stale >= patience:
            log(f"{cfg['variant']} early stop at epoch={epoch}; best_epoch={best_epoch}")
            break

    if best is None or best_state is None:
        raise RuntimeError(f"No checkpoint selected for {cfg['variant']}")
    report_metrics(
        {"epochs_run": len(history), "best_epoch": best_epoch,
         "pseudo_val_Hit@10": best["pseudo_val_Hit@10"],
         "pseudo_val_MeanAP": best["pseudo_val_MeanAP"]},
        label="Selected joint bulk validation metrics",
    )
    model.load_state_dict(best_state["readout"], strict=True)
    adapter.load_state_dict(best_state["support_adapter"], strict=True)

    ckpt_dir = CHECKPOINTS / cfg["variant"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "whole_model_checkpoint.pth"
    torch.save(
        {
            "stage": "round246_joint_seen_target_train_pc3_a375_mcf7",
            "readout_state_dict": best_state["readout"],
            "support_adapter_state_dict": best_state["support_adapter"],
            "best_epoch": best_epoch,
            "selection": best,
            "readout_cfg": readout_cfg,
            "joint_cfg": cfg,
            "target_audit": target_audit,
            "labels": labels,
            "kg_flow": kg_ckpt.get("flow", {}),
            "checkpoint_policy": "whole model: readout and support adapter from the same epoch",
        },
        ckpt_path,
    )
    reload_audit = checkpoint_reload_audit(
        r243,
        r211,
        target_init,
        readout_cfg,
        cfg,
        ckpt_path,
        model,
        adapter,
        tables,
        device,
    )

    report_step("Evaluating the selected joint bulk checkpoint")
    metrics, per = evaluate_joint(r243, r208, model, adapter, tables, device, cfg, keep_per_unit=True)
    metrics["checkpoint"] = str(ckpt_path)
    selected = {
        "variant": cfg["variant"],
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": r243.sha256_file(ckpt_path),
        "best_epoch": int(best_epoch),
        "selection": best,
        "readout_cfg": readout_cfg,
        "joint_cfg": cfg,
        "target_audit": target_audit,
        "reload_audit": reload_audit,
    }
    return metrics, per, pd.DataFrame(history), selected


def write_protocol_audit(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    require_torch()
    r243, _, r208, labels, kg_path, _, tables, data_audit = load_context(args, device)
    r243.write_round243_audit_tables(tables, data_audit)
    data_audit["round246_policy"] = {
        "mode": "joint_seen_target_crosscell_meta_train_pc3_a375_mcf7",
        "input_dim": 978,
        "candidate_space": 668,
        "kg_checkpoint": str(kg_path),
        "kg_checkpoint_sha256": r243.sha256_file(kg_path),
        "kg_retrained": False,
        "full_space_target_side": False,
        "whole_model_checkpoint_required": True,
        "forced_train_cells": FORCED_TRAIN_CELLS,
        "pseudo_unseen_cells": args.pseudo_unseen_cells,
        "official_unseen_cells": args.official_unseen_cells,
    }
    (SOURCE / "round246_protocol_audit.json").write_text(json.dumps(json_ready(data_audit), indent=2), encoding="utf-8")
    write_audit_md(data_audit)
    return data_audit


def write_audit_md(data_audit: dict[str, Any]) -> None:
    lines = [
        "# Round246 leakage and denominator audit",
        "",
        "- Scope: 668-label closed-set seen-target plus cross-cell support adaptation only.",
        "- Phase 2 full-space target side is out of scope.",
        "- Input dimensionality is 978 landmark features.",
        "- Candidate target space is 668 labels.",
        "- KG checkpoint is reused; KG is not retrained.",
        "- Support/query drug splits are disjoint by protocol and by episode sampler.",
        "- Official unseen cells are final evaluation only.",
        "- PC3, A375 and MCF7 are forced into base_train and adapter_train.",
        "",
        "## Leakage assertions",
        "",
    ]
    for key, value in data_audit["leakage_assertions"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Official unseen cells",
            "",
            "`" + ", ".join(data_audit["official_unseen_cells"]) + "`",
        ]
    )
    (QA / "leakage_and_denominator_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_method_lock(extra: dict[str, Any] | None = None) -> None:
    payload = {
        "round": "round246_joint_seen_target_train_pc3_a375_mcf7",
        "created_at": "2026-05-19",
        "seed": SEED,
        "scope": "668-label seen-target closed-set branch only; no Round240 full-space target side",
        "architecture": {
            "source": "Round211 target-side plus Round215-style support-conditioned query adapter",
            "input_features": 978,
            "query_encoder": "Linear(978,256)-LayerNorm-GELU-Dropout(0.35)-Linear(256,256)-LayerNorm-query_norm-L2",
            "candidate_space": 668,
            "target_side": "fixed Round211 KG anchor plus trainable target_delta/diag/target_bias/score_scale",
            "support_adapter": "row MLP + support-weighted attention over [support_q,pos_proto,product,difference]",
            "pos_proto_policy": "fixed KG target anchor, not target_delta-updated target tensor",
        },
        "training": {
            "single_stage_joint_optimizer": True,
            "loss": "base seen-target retrieval loss + lambda_meta * episodic support-adapt retrieval loss + readout regularization + delta_l2",
            "checkpoint_policy": "whole model only; readout and support adapter states from the same selected epoch",
            "selection": "pseudo-unseen support-adapt Hit@10, then MeanAP, NDCG@10, meta-val no-adapt Hit@10",
            "forced_train_cells": FORCED_TRAIN_CELLS,
        },
        "references": {
            "round243_two_step_official_unseen_support_adapt_Hit@10": BASE_REFERENCE_HIT10,
        },
        "paths": {
            "round_root": str(ROUND_ROOT),
            "round243_script": str(ROUND243_SCRIPT),
        },
    }
    if extra:
        payload["run_artifacts"] = json_ready(extra)
    text = json.dumps(json_ready(payload), indent=2)
    (ROUND_ROOT / "method_lock.json").write_text(text, encoding="utf-8")
    (SOURCE / "method_lock.json").write_text(text, encoding="utf-8")


def update_summaries() -> None:
    metrics = None
    metric_files = sorted(p for p in SOURCE.glob("metric_summary_r246_joint_*.tsv") if "_official_" not in p.name)
    if metric_files:
        metrics = pd.concat([pd.read_csv(p, sep="\t") for p in metric_files], ignore_index=True, sort=False)
        metrics.to_csv(SOURCE / "metric_summary.tsv", sep="\t", index=False)
        official = metrics[metrics["split"].isin(["official_test", "official_test_profile"])].copy()
        official.to_csv(SOURCE / "metric_summary_official_unseen_main.tsv", sep="\t", index=False)
        pseudo = metrics[metrics["split"].eq("pseudo_val")].copy()
        pseudo.to_csv(SOURCE / "metric_summary_pseudo_unseen_validation.tsv", sep="\t", index=False)
        seen = metrics[metrics["split"].eq("seen_test")].copy()
        seen.to_csv(SOURCE / "metric_summary_seen_cell_diagnostic.tsv", sep="\t", index=False)
    history_files = sorted(SOURCE.glob("training_history_r246_joint_*.tsv"))
    if history_files:
        pd.concat([pd.read_csv(p, sep="\t") for p in history_files], ignore_index=True, sort=False).to_csv(
            SOURCE / "training_history.tsv", sep="\t", index=False
        )
    selected_files = sorted(SOURCE.glob("selected_*.json"))
    selected_rows = []
    for path in selected_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "variant": payload["variant"],
            "checkpoint": payload["checkpoint"],
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "best_epoch": payload["best_epoch"],
            **{f"selection_{k}": v for k, v in payload["selection"].items() if isinstance(v, (int, float, str))},
            **{f"cfg_{k}": v for k, v in payload["joint_cfg"].items() if isinstance(v, (int, float, str, bool))},
        }
        selected_rows.append(row)
    if selected_rows:
        selection = pd.DataFrame(selected_rows)
        selection.to_csv(SOURCE / "joint_selection.tsv", sep="\t", index=False)
        if metrics is not None and len(metrics):
            sort_cols = [
                "selection_pseudo_val_Hit@10",
                "selection_pseudo_val_MeanAP",
                "selection_pseudo_val_NDCG@10",
                "selection_meta_val_noadapt_Hit@10",
            ]
            protocol_selected = selection.sort_values(sort_cols, ascending=False).iloc[0]
            official_support = metrics[
                (metrics["split"].eq("official_test"))
                & (metrics["evaluation_family"].eq("seen_668_support_adapt"))
                & (metrics["evaluation_unit"].eq("drug-mean"))
            ].copy()
            summary_rows = []
            picks = [("protocol_selected_by_pseudo_unseen", str(protocol_selected["variant"]))]
            if len(official_support):
                official_best = official_support.sort_values(["Hit@10", "MeanAP", "NDCG@10"], ascending=False).iloc[0]
                official_core = str(official_best["variant"]).replace("_official_unseen_support_adapt", "")
                picks.append(("official_unseen_oracle_best_diagnostic", official_core))
            for selection_type, core_variant in picks:
                official_rows = metrics[
                    (metrics["split"].eq("official_test"))
                    & (metrics["evaluation_unit"].eq("drug-mean"))
                    & (metrics["variant"].str.startswith(f"{core_variant}_official_unseen_"))
                ].copy()
                sel_rows = selection[selection["variant"].eq(core_variant)]
                sel = sel_rows.iloc[0].to_dict() if len(sel_rows) else {}
                for _, row in official_rows.iterrows():
                    summary_rows.append(
                        {
                            "selection_type": selection_type,
                            "variant_core": core_variant,
                            "official_variant": row["variant"],
                            "evaluation_family": row["evaluation_family"],
                            "selection_pseudo_val_Hit@10": sel.get("selection_pseudo_val_Hit@10"),
                            "selection_pseudo_val_MeanAP": sel.get("selection_pseudo_val_MeanAP"),
                            "selection_pseudo_val_NDCG@10": sel.get("selection_pseudo_val_NDCG@10"),
                            "selection_meta_val_noadapt_Hit@10": sel.get("selection_meta_val_noadapt_Hit@10"),
                            "Hit@10": row.get("Hit@10"),
                            "MeanAP": row.get("MeanAP"),
                            "NDCG@10": row.get("NDCG@10"),
                            "AUROC": row.get("AUROC"),
                            "Hit@20": row.get("Hit@20"),
                            "Hit@50": row.get("Hit@50"),
                            "lambda_meta": row.get("lambda_meta"),
                            "shift_scale": row.get("shift_scale"),
                            "max_delta_norm": row.get("max_delta_norm"),
                            "checkpoint": row.get("checkpoint"),
                        }
                    )
            if summary_rows:
                pd.DataFrame(summary_rows).to_csv(SOURCE / "protocol_selection_summary.tsv", sep="\t", index=False)


def write_report() -> None:
    update_summaries()
    metric_path = SOURCE / "metric_summary.tsv"
    if not metric_path.exists():
        table_text = "No metrics available yet."
        best_text = "No selected model yet."
    else:
        metrics = pd.read_csv(metric_path, sep="\t")
        selection_summary_path = SOURCE / "protocol_selection_summary.tsv"
        selection_summary = pd.read_csv(selection_summary_path, sep="\t") if selection_summary_path.exists() else pd.DataFrame()
        keep_cols = [
            "variant",
            "split",
            "evaluation_family",
            "evaluation_unit",
            "candidate_space",
            "Hit@10",
            "MeanAP",
            "NDCG@10",
            "AUROC",
            "Hit@20",
            "Hit@50",
            "lambda_meta",
            "shift_scale",
            "max_delta_norm",
        ]
        shown = metrics[metrics["split"].isin(["official_test", "pseudo_val", "seen_test"])].copy()
        table_text = shown[[c for c in keep_cols if c in shown.columns]].to_string(index=False)
        if len(selection_summary):
            protocol = selection_summary[
                (selection_summary["selection_type"].eq("protocol_selected_by_pseudo_unseen"))
                & (selection_summary["evaluation_family"].eq("seen_668_support_adapt"))
            ].iloc[0]
            oracle = selection_summary[
                (selection_summary["selection_type"].eq("official_unseen_oracle_best_diagnostic"))
                & (selection_summary["evaluation_family"].eq("seen_668_support_adapt"))
            ]
            oracle_text = ""
            if len(oracle):
                oracle_row = oracle.iloc[0]
                oracle_text = (
                    f"\nOfficial-unseen oracle best diagnostic only: `{oracle_row['variant_core']}` "
                    f"Hit@10={oracle_row['Hit@10']:.6f}, MeanAP={oracle_row['MeanAP']:.6f}, "
                    f"NDCG@10={oracle_row['NDCG@10']:.6f}, AUROC={oracle_row['AUROC']:.6f}. "
                    "This row must not be used as protocol-selected evidence."
                )
            best_text = (
                f"Protocol-selected by pseudo-unseen validation: `{protocol['variant_core']}`. "
                f"Official unseen support-adapt Hit@10={protocol['Hit@10']:.6f}, "
                f"MeanAP={protocol['MeanAP']:.6f}, NDCG@10={protocol['NDCG@10']:.6f}, "
                f"AUROC={protocol['AUROC']:.6f}."
                f"{oracle_text}"
            )
        else:
            best_text = "No protocol-selection summary row yet."
    text = f"""# Round246 joint seen-target cross-cell meta-learning report

## Scope

Round246 trains the same 668-label seen-target predictor and support-conditioned cross-cell adapter as Round244 in one optimizer and one whole-model checkpoint. It changes only the cell split so PC3, A375 and MCF7 are included in training cells. It does not use the Round240 full-space target side.

## Main result

{best_text}

Round243 two-step official-unseen support-adapt Hit@10 reference: `{BASE_REFERENCE_HIT10:.6f}`.

## Metric table

```text
{table_text}
```

## Interpretation guardrails

- Official unseen cells are final main evidence only.
- Pseudo-unseen cells are model selection only.
- Seen-cell rows are diagnostics only.
- The selected checkpoint is a whole-model checkpoint; readout and support adapter states come from the same epoch.
"""
    (ROUND_ROOT / "round246_result_report_zh.md").write_text(text, encoding="utf-8")
    (SOURCE / "round246_result_report_zh.md").write_text(text, encoding="utf-8")


def run_training(args: argparse.Namespace, device: torch.device) -> None:
    require_torch()
    r243, r211, r208, labels, kg_path, kg_ckpt, tables, data_audit = load_context(args, device)
    r243.write_round243_audit_tables(tables, data_audit)
    data_audit["round246_policy"] = {
        "mode": "joint_seen_target_crosscell_meta_train_pc3_a375_mcf7",
        "kg_checkpoint": str(kg_path),
        "kg_checkpoint_sha256": r243.sha256_file(kg_path),
        "kg_retrained": False,
        "candidate_space": 668,
        "full_space_target_side": False,
        "forced_train_cells": FORCED_TRAIN_CELLS,
        "pseudo_unseen_cells": args.pseudo_unseen_cells,
        "official_unseen_cells": args.official_unseen_cells,
    }
    (SOURCE / "round246_protocol_audit.json").write_text(json.dumps(json_ready(data_audit), indent=2), encoding="utf-8")
    write_audit_md(data_audit)

    selected_records = {}
    for cfg in selected_configs(args):
        log(f"training {cfg['variant']} smoke={args.smoke}")
        metrics, per, history, selected = train_joint_variant(
            r243,
            r211,
            r208,
            kg_ckpt,
            labels,
            tables,
            device,
            cfg,
            smoke=bool(args.smoke),
        )
        suffix = selected["variant"]
        metrics.to_csv(SOURCE / f"metric_summary_{suffix}.tsv", sep="\t", index=False)
        history.to_csv(SOURCE / f"training_history_{suffix}.tsv", sep="\t", index=False)
        if len(per):
            per.to_csv(SOURCE / f"per_unit_metrics_{suffix}.tsv", sep="\t", index=False)
        (SOURCE / f"selected_{suffix}.json").write_text(json.dumps(json_ready(selected), indent=2), encoding="utf-8")
        selected_records[suffix] = selected
        update_summaries()
        write_method_lock(extra={"data_audit": data_audit, "selected_models": selected_records})
        write_report()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["audit_protocol", "smoke", "joint", "sweep", "report", "all"], default="joint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--kg-checkpoint", default=None)
    parser.add_argument("--pseudo-unseen-cells", default=",".join(DEFAULT_R246_PSEUDO_UNSEEN_VAL_CELLS))
    parser.add_argument("--official-unseen-cells", default="round215_heldout_panel")
    parser.add_argument("--seen-diagnostic-cells", default=",".join(FORCED_TRAIN_CELLS))
    parser.add_argument("--runs", choices=["default", "sweep"], default="default")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--lambda-meta", type=float, default=0.5)
    parser.add_argument("--shift-scale", type=float, default=0.15)
    parser.add_argument("--max-delta-norm", type=float, default=2.5)
    parser.add_argument("--sweep-lambda-meta", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--sweep-shift-scales", type=float, nargs="+", default=[0.10, 0.15])
    parser.add_argument("--sweep-max-delta-norms", type=float, nargs="+", default=[2.5, 3.5])
    return parser.parse_args()


def main() -> None:
    ensure_dirs()
    args = parse_args()
    set_seed(SEED)
    device = torch.device(resolve_device(args.device)) if torch is not None else None
    if args.phase == "audit_protocol":
        write_protocol_audit(args, device)
        write_method_lock(extra={"audit_only": str(SOURCE / "round246_protocol_audit.json")})
        return
    if args.phase == "report":
        write_report()
        return
    if args.phase == "smoke":
        args.smoke = True
        run_training(args, device)
        return
    if args.phase == "all":
        write_protocol_audit(args, device)
        args.phase = "joint"
        run_training(args, device)
        return
    run_training(args, device)


if __name__ == "__main__":
    main()
