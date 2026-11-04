#!/usr/bin/env python3
"""Leakage-safe cell splits, response scoring and support-adaptation primitives."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
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

ROUND211_ROOT = METAR_ROOT / "claude_round211_dim256_readout_capacity_sweep_20260515"
ROUND211_SCRIPT = ROUND211_ROOT / "scripts" / "run_round211_dim256_readout_capacity_sweep.py"
ROUND208_SCRIPT = (
    METAR_ROOT
    / "claude_round215_direct_query_shift_adapter_20260517"
    / "scripts"
    / "run_round208_a01_fixedtarget_maml_readout.py"
)
ROUND215_PANEL_JSON = (
    METAR_ROOT
    / "claude_round215_direct_query_shift_adapter_20260517"
    / "round215_fixed_cell_panel.json"
)
ROUND240_SCRIPT = (
    METAR_ROOT
    / "claude_round240_f7_exact_narrow_pc3_20260518"
    / "scripts"
    / "run_round240_f7_exact_narrow_pc3.py"
)

SEED = 20260519
CELL_LINE = "PC3"
KG_BASE_FLOW_ID = "r210_a02_reladd2_d256_f16_t020_edge050"
KG_CHECKPOINT_NAME = f"{KG_BASE_FLOW_ID}_kg_checkpoint_epoch030.pth"
KG_CHECKPOINT_RELATIVE = (
    Path("claude_round210_dim256_checkpoint_sweep_20260515")
    / "checkpoints"
    / "process_only"
    / KG_BASE_FLOW_ID
    / KG_CHECKPOINT_NAME
)

HIT_KS = [1, 5, 10, 20, 30, 50]
ROUND211_BEST_REFERENCE = {
    "source": "Round211 best archived result",
    "variant": "r210_a02_reladd2_d256_f16_t020_edge050_process_only_kg030_rwide256_hit10",
    "split": "test",
    "evaluation_family": "seen_668",
    "evaluation_unit": "drug-mean",
    "candidate_space": 668,
    "Hit@10": 0.3828,
    "MeanAP": 0.2092,
    "NDCG@10": 0.2342,
    "AUROC": 0.8258,
}
ROUND215_BEST_REFERENCE = {
    "source": "Round215 teacher-distilled support encoder archived result",
    "variant": "support_delta_encoder_h128_sc0p1_l21_mn3p5_do0p1_rw0.2",
    "split": "target_test",
    "evaluation_family": "seen_668_support_adapt",
    "evaluation_unit": "drug-mean",
    "candidate_space": 668,
    "Hit@10": 0.4980,
    "MeanAP": 0.2237,
    "AUROC": 0.7905,
}
ROUND240_BEST_REFERENCE = {
    "source": "Round240 best archived result",
    "variant": "e8_enc105e6",
    "split": "holdout_test",
    "evaluation_family": "holdout_fullspace_top5",
    "evaluation_unit": "profile",
    "candidate_space": 16583,
    "top5pct_success": 0.5026223776223776,
    "top5pct_success_remove_seen": 0.7159090909090909,
    "seen_668_Hit@10": 0.18490467484983025,
    "seen_668_MeanAP": 0.06854856396033691,
    "seen_668_AUROC": 0.7767615286596796,
}

READOUT_CFG = {
    "readout_id": "r243_round211_h256_unseencell_safe_hit10",
    "hidden": 256,
    "extra_layers": 0,
    "dropout": 0.35,
    "lr": 5.0e-4,
    "weight_decay": 3.0e-4,
    "target_bias_l2": 5.0e-4,
    "target_delta_l2": 5.0e-3,
    "scorer_l2": 2.0e-4,
    "selection": ["Hit@10", "MeanAP", "Hit@5", "NDCG@10"],
    "patience": 8,
    "max_epochs": 60,
    "min_epochs": 12,
}

SUPPORT_ENCODER_CFG = {
    "variant": "r243_support_delta_encoder_h128_sc0p15_l20p001_mn3p5_do0p1",
    "hidden_dim": 128,
    "shift_scale": 0.15,
    "delta_l2": 0.001,
    "max_delta_norm": 3.5,
    "dropout": 0.1,
    "support_size": 32,
    "query_size": 32,
    "tasks_per_epoch": 240,
    "max_epochs": 30,
    "min_epochs": 5,
    "patience": 7,
    "lr": 2.0e-4,
    "weight_decay": 1.0e-5,
}

PHASE2_EXTRA_VARIANTS: dict[str, dict[str, Any]] = {
    "n0_gnn36e5": {
        "base": "e1_gnn38e5",
        "updates": {"gnn_lr": 3.6e-5},
        "rationale": "narrow sweep below first-sweep winner e1_gnn38e5",
    },
    "n1_gnn34e5": {
        "base": "e1_gnn38e5",
        "updates": {"gnn_lr": 3.4e-5},
        "rationale": "test whether further reducing GNN learning rate helps full-space retrieval",
    },
    "n2_gnn38e5_label45e5": {
        "base": "e1_gnn38e5",
        "updates": {"label_lr": 4.5e-5},
        "rationale": "combine best GNN-rate direction with second-best label-rate direction",
    },
    "n3_gnn36e5_label45e5": {
        "base": "e1_gnn38e5",
        "updates": {"gnn_lr": 3.6e-5, "label_lr": 4.5e-5},
        "rationale": "joint low-GNN and low-label learning-rate probe around the observed optimum",
    },
}

DEFAULT_PSEUDO_UNSEEN_VAL_CELLS = [
    "HA1E",
    "HEK293",
    "MCF10A",
    "MCF7",
    "MDAMB231",
    "PC3",
    "PHH",
    "THP1",
]

DEFAULT_FULLSPACE_HOLDOUT_CELLS = ["HCC515", "HELA", "HT29", "VCAP"]

CELL_SPLIT_COLUMNS = [
    "cell_line",
    "split_role",
    "official_split",
    "included_in_base_train",
    "included_in_adapter_train",
    "included_in_adapter_val",
    "included_in_final_main_test",
    "included_in_seen_diagnostic",
    "support_groups",
    "val_groups",
    "test_groups",
    "test_profiles",
]


def ensure_dirs() -> None:
    for path in [SOURCE, QA, CHECKPOINTS, LOGS]:
        path.mkdir(parents=True, exist_ok=True)


def require_torch() -> None:
    if torch is None or F is None:
        raise ModuleNotFoundError(
            "torch is required for training/evaluation phases. "
            "Use the remote PertKGE/MeTar environment or install torch locally."
        )


def log(message: str) -> None:
    print(f"[round243] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
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


def load_round211_module():
    module = import_from_path("round243_round211", ROUND211_SCRIPT)
    module.ROOT = ROUND_ROOT
    module.METAR_ROOT = METAR_ROOT
    module.SOURCE = SOURCE
    module.QA = QA
    module.CHECKPOINTS = CHECKPOINTS
    module.LOGS = LOGS
    module.METHOD_NAME = "Round243_Phase1_Round211215"
    return module


def load_round208_module():
    module = import_from_path("round243_round208", ROUND208_SCRIPT)
    # Keep H07 paths computed by the original script, but redirect outputs.
    module.ROOT = ROUND_ROOT
    module.SOURCE = SOURCE
    module.QA = QA
    module.CHECKPOINTS = CHECKPOINTS
    module.LOGS = LOGS
    return module


def load_round240_module():
    module = import_from_path("round243_round240", ROUND240_SCRIPT)
    module.ROUND_ROOT = ROUND_ROOT
    return module


def candidate_kg_checkpoints() -> list[Path]:
    return [
        METAR_ROOT / KG_CHECKPOINT_RELATIVE,
        Path("/NFS2_home/NFS2_home_1/lwn/claude/manuscript/write") / KG_CHECKPOINT_RELATIVE,
        Path("/home/lwn/claude/manuscript/write") / KG_CHECKPOINT_RELATIVE,
    ]


def resolve_kg_checkpoint(user_path: str | None) -> Path:
    if user_path:
        path = Path(user_path).expanduser()
        if path.exists():
            return path.resolve()
        raise FileNotFoundError(path)
    for path in candidate_kg_checkpoints():
        if path.exists():
            return path.resolve()
    searched = "\n".join(str(p) for p in candidate_kg_checkpoints())
    raise FileNotFoundError(
        "Round211 KG checkpoint not found. Pass --kg-checkpoint explicitly. "
        f"Searched:\n{searched}"
    )


def current_target_tensor(model: nn.Module) -> torch.Tensor:
    return F.normalize(model.target_anchor + model.target_delta, dim=1)


def encode_q(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return F.normalize(model.query_norm(model.query_encoder(x)), dim=1)


def score_from_q(model: nn.Module, q: torch.Tensor, delta: Optional[torch.Tensor], scale: float) -> torch.Tensor:
    if delta is not None:
        q = F.normalize(q + float(scale) * delta[None, :], dim=1)
    target = current_target_tensor(model)
    score = (q * model.diag[None, :]) @ target.T
    score = score * torch.exp(torch.clamp(model.score_scale, -2.0, 2.0))
    return score + model.target_bias[None, :]


def retrieval_loss_from_logits(logits: torch.Tensor, y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    log_all = torch.logsumexp(logits, dim=1)
    log_pos = torch.logsumexp(logits.masked_fill(y <= 0, -1.0e9), dim=1)
    w = weights / torch.clamp(weights.sum(), min=1.0e-8)
    bag = torch.sum(w * (log_all - log_pos))
    per = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    pos = y > 0
    neg = ~pos
    pos_loss = (per * pos.float()).sum(dim=1) / torch.clamp(pos.float().sum(dim=1), min=1.0)
    neg_loss = (per * neg.float()).sum(dim=1) / torch.clamp(neg.float().sum(dim=1), min=1.0)
    bce = torch.sum(w * 0.5 * (pos_loss + neg_loss))
    return bag + 0.20 * bce


def predict_from_q_model(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    delta: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            report_progress(start, len(x), label='predict from q model rows')
            xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            q = encode_q(model, xb)
            out.append(score_from_q(model, q, delta=delta, scale=scale).detach().cpu().numpy())
        report_progress(len(x), len(x), label='predict from q model rows')
    return np.vstack(out) if out else np.zeros((0, model.n_targets), dtype=np.float32)


def precompute_q(model: nn.Module, item: dict, device: torch.device) -> None:
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(item["x"]), 2048):
            report_progress(start, len(item["x"]), label='Query encoding rows')
            xb = torch.tensor(item["x"][start : start + 2048], dtype=torch.float32, device=device)
            out.append(encode_q(model, xb).detach().cpu().numpy().astype(np.float32))
        report_progress(len(item["x"]), len(item["x"]), label='Query encoding rows')
    item["q"] = np.vstack(out).astype(np.float32) if out else np.zeros((0, model.dim), dtype=np.float32)


def tensor_slice(item: dict, key: str, idx: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(item[key][idx], dtype=torch.float32, device=device)


def sample_indices(indices: np.ndarray, size: int, rng: np.random.Generator, replace: bool = False) -> np.ndarray:
    if len(indices) == 0:
        return indices.astype(np.int64)
    if size <= 0 or len(indices) <= size:
        return np.asarray(indices, dtype=np.int64)
    return rng.choice(np.asarray(indices, dtype=np.int64), size=size, replace=replace)


def fmt_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


class SupportDeltaEncoder(nn.Module):
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


def fixed_panel_cells() -> list[str]:
    if not ROUND215_PANEL_JSON.exists():
        raise FileNotFoundError(ROUND215_PANEL_JSON)
    payload = json.loads(ROUND215_PANEL_JSON.read_text(encoding="utf-8"))
    return [str(row["cell_line"]) for row in payload["panel_cells"]]


def load_panel_rows() -> pd.DataFrame:
    if not ROUND215_PANEL_JSON.exists():
        raise FileNotFoundError(ROUND215_PANEL_JSON)
    payload = json.loads(ROUND215_PANEL_JSON.read_text(encoding="utf-8"))
    panel = pd.DataFrame(payload["panel_cells"])
    if panel.empty:
        raise RuntimeError(f"No panel cells in {ROUND215_PANEL_JSON}")
    panel["cell_line"] = panel["cell_line"].astype(str)
    return panel


def parse_cells(text: str) -> list[str]:
    if text == "round215_fixed_panel":
        return fixed_panel_cells()
    if text == "round243_official_unseen":
        return load_panel_rows().query("cell_split == 'heldout_cell'")["cell_line"].astype(str).tolist()
    if text == "round243_seen_diagnostic":
        return load_panel_rows().query("cell_split == 'train_cell'")["cell_line"].astype(str).tolist()
    return [x.strip() for x in text.split(",") if x.strip()]


def load_splits_v3(r208) -> dict[str, Any]:
    return json.loads(Path(r208.SPLITS_V3_PATH).read_text(encoding="utf-8"))


def official_split_map(splits_v3: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for cell in splits_v3.get("train_celllines", []):
        mapping[str(cell)] = "train_cell"
    for cell in splits_v3.get("test_celllines", []):
        mapping[str(cell)] = "holdout_cell"
    return mapping


def choose_round243_cells(
    r208,
    pseudo_unseen_text: str,
    official_unseen_text: str,
    seen_diag_text: str,
) -> dict[str, Any]:
    splits_v3 = load_splits_v3(r208)
    split_map = official_split_map(splits_v3)
    train_cells = [str(c) for c in splits_v3["train_celllines"]]
    holdout_cells = [str(c) for c in splits_v3["test_celllines"]]
    panel = load_panel_rows()

    if official_unseen_text == "round215_heldout_panel":
        official_unseen = panel.query("cell_split == 'heldout_cell'")["cell_line"].astype(str).tolist()
    else:
        official_unseen = parse_cells(official_unseen_text)
    if seen_diag_text == "round215_train_panel":
        seen_diag = panel.query("cell_split == 'train_cell'")["cell_line"].astype(str).tolist()
    else:
        seen_diag = parse_cells(seen_diag_text)
    if pseudo_unseen_text == "round215_train_panel":
        pseudo_unseen = panel.query("cell_split == 'train_cell'")["cell_line"].astype(str).tolist()
    elif pseudo_unseen_text == "default":
        pseudo_unseen = list(DEFAULT_PSEUDO_UNSEEN_VAL_CELLS)
    elif pseudo_unseen_text.strip().lower() in {"", "none"}:
        pseudo_unseen = []
    else:
        pseudo_unseen = parse_cells(pseudo_unseen_text)

    errors = []
    for cell in official_unseen:
        if split_map.get(cell) != "holdout_cell":
            errors.append(f"official unseen cell is not in splits_v3 test_celllines: {cell} ({split_map.get(cell, 'missing')})")
    for cell in seen_diag:
        if split_map.get(cell) != "train_cell":
            errors.append(f"seen diagnostic cell is not in splits_v3 train_celllines: {cell} ({split_map.get(cell, 'missing')})")
    for cell in pseudo_unseen:
        if split_map.get(cell) != "train_cell":
            errors.append(f"pseudo-unseen validation cell is not in splits_v3 train_celllines: {cell} ({split_map.get(cell, 'missing')})")
    if set(official_unseen).intersection(train_cells):
        errors.append("official unseen cells overlap train_celllines")
    if set(official_unseen).intersection(pseudo_unseen):
        errors.append("official unseen cells overlap pseudo-unseen validation cells")
    if errors:
        raise RuntimeError("Invalid Round243 cell protocol:\n- " + "\n- ".join(errors))

    excluded_from_base = sorted(set(official_unseen).union(pseudo_unseen))
    base_train_cells = [c for c in train_cells if c not in set(excluded_from_base)]
    adapter_train_cells = base_train_cells

    cell_sets = {
        "train_cells": train_cells,
        "holdout_cells": holdout_cells,
        "official_unseen_cells": official_unseen,
        "seen_diagnostic_cells": seen_diag,
        "pseudo_unseen_val_cells": pseudo_unseen,
        "base_train_cells": base_train_cells,
        "adapter_train_cells": adapter_train_cells,
        "base_excluded_cells": excluded_from_base,
        "split_map": split_map,
        "panel": panel,
        "splits_v3": splits_v3,
    }
    return cell_sets


def build_round243_tables(r208, labels: list[str], cell_sets: dict[str, Any], device: torch.device) -> tuple[dict[str, dict], dict[str, Any]]:
    report_step('Building leakage-safe bulk tables and standardizing features')
    module = r208.import_h07_module()
    protocol = module.bulk_base.load_protocol(r208.LOCK_PATH)
    prepared = module.bulk_base.prepare_data(protocol, device)
    split_frames = {}
    for split_name in ["inner_train", "val", "test"]:
        df, x = r208.split_to_numpy(prepared[split_name])
        df = df.copy()
        df["drug"] = df["drug"].astype(str)
        df["cell_line"] = df["cell_line"].astype(str)
        split_frames[split_name] = (df, x)

    specs = [
        ("meta_train", "inner_train", cell_sets["base_train_cells"], "meta_train"),
        ("meta_val", "val", cell_sets["base_train_cells"], "meta_val"),
        ("adapter_train", "inner_train", cell_sets["adapter_train_cells"], "adapter_train"),
        ("pseudo_support", "inner_train", cell_sets["pseudo_unseen_val_cells"], "pseudo_support"),
        ("pseudo_val", "val", cell_sets["pseudo_unseen_val_cells"], "pseudo_val"),
        ("official_support", "inner_train", cell_sets["official_unseen_cells"], "official_support"),
        ("official_val", "val", cell_sets["official_unseen_cells"], "official_val"),
        ("official_test", "test", cell_sets["official_unseen_cells"], "official_test"),
        ("seen_support", "inner_train", cell_sets["seen_diagnostic_cells"], "seen_support"),
        ("seen_val", "val", cell_sets["seen_diagnostic_cells"], "seen_val"),
        ("seen_test", "test", cell_sets["seen_diagnostic_cells"], "seen_test"),
    ]
    tables: dict[str, dict] = {}
    for _table_index, (key, source_split, cells, split_name) in enumerate(specs):
        report_progress(_table_index, len(specs), label="Bulk split tables")
        df_split, x_split = split_frames[source_split]
        rows, x_raw = r208.build_drugcell_table(df_split, x_split, labels, cells, split_name)
        tables[key] = {
            "rows": rows,
            "x_raw": x_raw,
            "y": r208.true_target_matrix(rows, labels),
            "weights": r208.row_weights(rows) if len(rows) else np.zeros((0,), dtype=np.float32),
        }
    report_progress(len(specs), len(specs), label="Bulk split tables")
    report_step("Building profile-level bulk evaluation tables")
    for key, cells in [
        ("official_test_profile", cell_sets["official_unseen_cells"]),
        ("seen_test_profile", cell_sets["seen_diagnostic_cells"]),
    ]:
        df_split, x_split = split_frames["test"]
        rows, x_raw = r208.build_profile_table(df_split, x_split, labels, cells, key)
        tables[key] = {
            "rows": rows,
            "x_raw": x_raw,
            "y": r208.true_target_matrix(rows, labels),
            "weights": r208.row_weights(rows) if len(rows) else np.zeros((0,), dtype=np.float32),
        }

    report_step("Standardizing bulk features with training-only statistics")
    std_audit = r208.standardize_tables(tables["meta_train"]["x_raw"], tables)
    for item in tables.values():
        rows = item["rows"]
        item["by_cell"] = {
            cell: np.where(rows["cell_line"].astype(str).eq(cell).to_numpy())[0]
            for cell in sorted(rows["cell_line"].astype(str).unique())
        }
    audit = {
        "mode": "round243_explicit_cell_disjoint",
        "protocol_path": str(r208.LOCK_PATH),
        "protocol_sha256": sha256_file(Path(r208.LOCK_PATH)),
        "splits_v3_path": str(r208.SPLITS_V3_PATH),
        "splits_v3_sha256": sha256_file(Path(r208.SPLITS_V3_PATH)),
        "n_train_cells": len(cell_sets["train_cells"]),
        "n_holdout_cells": len(cell_sets["holdout_cells"]),
        "base_train_cells": cell_sets["base_train_cells"],
        "adapter_train_cells": cell_sets["adapter_train_cells"],
        "pseudo_unseen_val_cells": cell_sets["pseudo_unseen_val_cells"],
        "official_unseen_cells": cell_sets["official_unseen_cells"],
        "seen_diagnostic_cells": cell_sets["seen_diagnostic_cells"],
        "base_excluded_cells": cell_sets["base_excluded_cells"],
        "drug_split_counts": {
            "inner_train_drugs": len(protocol["drug_split"]["inner_train_drugs"]),
            "val_drugs": len(protocol["drug_split"]["val_drugs"]),
            "test_drugs": len(protocol["drug_split"]["test_drugs"]),
        },
        "table_counts": table_counts(tables),
        "standardization": std_audit,
        "leakage_assertions": leakage_assertions(cell_sets, protocol),
    }
    return tables, audit


def table_counts(tables: dict[str, dict]) -> dict[str, dict[str, int]]:
    return {
        name: {
            "n_groups": int(len(item["rows"])),
            "n_cells": int(item["rows"]["cell_line"].nunique()) if len(item["rows"]) else 0,
            "n_drugs": int(item["rows"]["drug"].nunique()) if len(item["rows"]) else 0,
            "n_profiles": int(item["rows"]["n_profiles"].sum()) if len(item["rows"]) else 0,
        }
        for name, item in tables.items()
    }


def leakage_assertions(cell_sets: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    inner = set(map(str, protocol["drug_split"]["inner_train_drugs"]))
    val = set(map(str, protocol["drug_split"]["val_drugs"]))
    test = set(map(str, protocol["drug_split"]["test_drugs"]))
    return {
        "official_unseen_intersects_base_train": sorted(set(cell_sets["official_unseen_cells"]).intersection(cell_sets["base_train_cells"])),
        "pseudo_unseen_intersects_base_train": sorted(set(cell_sets["pseudo_unseen_val_cells"]).intersection(cell_sets["base_train_cells"])),
        "official_unseen_intersects_adapter_train": sorted(set(cell_sets["official_unseen_cells"]).intersection(cell_sets["adapter_train_cells"])),
        "inner_val_drug_overlap": sorted(inner.intersection(val))[:20],
        "inner_test_drug_overlap": sorted(inner.intersection(test))[:20],
        "val_test_drug_overlap": sorted(val.intersection(test))[:20],
        "official_unseen_main_test_policy": "only official_test / official_test_profile are used for final main metrics",
        "seen_cell_policy": "seen_test / seen_test_profile are diagnostic only",
        "pc3_fullspace_policy": "PC3 is a seen-cell unseen-target diagnostic under current splits",
    }


def write_round243_audit_tables(tables: dict[str, dict], data_audit: dict[str, Any]) -> None:
    rows = []
    split_map = official_split_map(json.loads(Path(data_audit["splits_v3_path"]).read_text(encoding="utf-8")))
    all_cells = sorted(
        set(data_audit["base_train_cells"])
        | set(data_audit["pseudo_unseen_val_cells"])
        | set(data_audit["official_unseen_cells"])
        | set(data_audit["seen_diagnostic_cells"])
    )
    counts_by_cell = {}
    for key in [
        "meta_train",
        "meta_val",
        "official_support",
        "official_val",
        "official_test",
        "seen_support",
        "seen_val",
        "seen_test",
        "pseudo_support",
        "pseudo_val",
    ]:
        item = tables[key]
        if len(item["rows"]):
            counts_by_cell[key] = item["rows"].groupby("cell_line").agg(
                groups=("drug", "size"),
                drugs=("drug", "nunique"),
                profiles=("n_profiles", "sum"),
            )
        else:
            counts_by_cell[key] = pd.DataFrame(columns=["groups", "drugs", "profiles"])
    for cell in all_cells:
        split_role = (
            "official_unseen_main"
            if cell in data_audit["official_unseen_cells"]
            else "pseudo_unseen_validation"
            if cell in data_audit["pseudo_unseen_val_cells"]
            else "seen_diagnostic"
            if cell in data_audit["seen_diagnostic_cells"]
            else "base_train"
        )
        support_groups = 0
        val_groups = 0
        test_groups = 0
        test_profiles = 0
        if split_role == "official_unseen_main":
            count_keys = ("official_support", "official_val", "official_test")
        elif split_role == "pseudo_unseen_validation":
            count_keys = ("pseudo_support", "pseudo_val", "seen_test")
        elif split_role == "seen_diagnostic":
            count_keys = ("seen_support", "seen_val", "seen_test")
        else:
            count_keys = ("meta_train", "meta_val", "meta_val")
        support_key, val_key, test_key = count_keys
        if support_key in counts_by_cell and cell in counts_by_cell[support_key].index:
            support_groups = int(counts_by_cell[support_key].loc[cell, "groups"])
        if val_key in counts_by_cell and cell in counts_by_cell[val_key].index:
            val_groups = int(counts_by_cell[val_key].loc[cell, "groups"])
        if test_key in counts_by_cell and cell in counts_by_cell[test_key].index:
            test_groups = int(counts_by_cell[test_key].loc[cell, "groups"])
            test_profiles = int(counts_by_cell[test_key].loc[cell, "profiles"])
        rows.append(
            {
                "cell_line": cell,
                "split_role": split_role,
                "official_split": split_map.get(cell, "missing"),
                "included_in_base_train": cell in data_audit["base_train_cells"],
                "included_in_adapter_train": cell in data_audit["adapter_train_cells"],
                "included_in_adapter_val": cell in data_audit["pseudo_unseen_val_cells"],
                "included_in_final_main_test": cell in data_audit["official_unseen_cells"],
                "included_in_seen_diagnostic": cell in data_audit["seen_diagnostic_cells"],
                "support_groups": support_groups,
                "val_groups": val_groups,
                "test_groups": test_groups,
                "test_profiles": test_profiles,
            }
        )
    pd.DataFrame(rows, columns=CELL_SPLIT_COLUMNS).to_csv(SOURCE / "cell_split_audit.tsv", sep="\t", index=False)
    drug_rows = []
    for left, right in [
        ("official_support", "official_test"),
        ("official_support", "official_val"),
        ("pseudo_support", "pseudo_val"),
        ("seen_support", "seen_test"),
    ]:
        left_drugs = set(tables[left]["rows"]["drug"].astype(str)) if len(tables[left]["rows"]) else set()
        right_drugs = set(tables[right]["rows"]["drug"].astype(str)) if len(tables[right]["rows"]) else set()
        drug_rows.append(
            {
                "left_split": left,
                "right_split": right,
                "n_left_drugs": len(left_drugs),
                "n_right_drugs": len(right_drugs),
                "n_overlap_drugs": len(left_drugs.intersection(right_drugs)),
                "overlap_examples": "|".join(sorted(left_drugs.intersection(right_drugs))[:20]),
            }
        )
    pd.DataFrame(drug_rows).to_csv(SOURCE / "drug_split_audit.tsv", sep="\t", index=False)
    pd.DataFrame(drug_rows).to_csv(SOURCE / "support_query_disjoint_audit.tsv", sep="\t", index=False)


def metric_macro_from_cells(cell_df: pd.DataFrame, variant: str, split_name: str, unit: str, source: str) -> dict:
    if "no_adapt" in variant:
        evaluation_family = "seen_668_no_adapt"
    elif "adapt" in variant:
        evaluation_family = "seen_668_support_adapt"
    else:
        evaluation_family = "seen_668"
    out = {
        "source": source,
        "variant": variant,
        "split": split_name,
        "evaluation_family": evaluation_family,
        "evaluation_unit": unit,
        "candidate_space": 668,
        "n_cells": int(cell_df["cell_line"].nunique()) if len(cell_df) else 0,
        "n_groups": int(cell_df["n_groups"].sum()) if len(cell_df) else 0,
        "n_drugs": int(cell_df["n_drugs"].sum()) if len(cell_df) else 0,
        "n_profiles": int(cell_df["n_profiles"].sum()) if len(cell_df) else 0,
    }
    if "delta_norm" in cell_df:
        out["delta_norm_mean"] = float(cell_df["delta_norm"].mean()) if len(cell_df) else float("nan")
    for metric in [f"Hit@{k}" for k in HIT_KS] + ["AUROC", "NDCG@10", "MeanAP"]:
        out[metric] = float(cell_df[metric].mean()) if len(cell_df) else float("nan")
    return out


def evaluate_by_cell(
    r208,
    model: nn.Module,
    encoder: Optional[SupportDeltaEncoder],
    support_item: dict,
    query_item: dict,
    device: torch.device,
    variant: str,
    support_size: int,
    shift_scale: float,
    unit: str,
    keep_per_unit: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell_summaries = []
    per_rows = []
    if encoder is not None:
        encoder.eval()
    model.eval()
    with torch.no_grad():
        for _cell_index, (cell, qidx_raw) in enumerate(query_item["by_cell"].items()):
            report_progress(_cell_index, len(query_item["by_cell"]), label=f"{variant}: cells inspected")
            qidx = np.asarray(qidx_raw, dtype=np.int64)
            if len(qidx) == 0:
                continue
            delta = None
            delta_norm = float("nan")
            if encoder is not None:
                support_all = support_item["by_cell"].get(cell, np.asarray([], dtype=np.int64))
                if len(support_all) == 0:
                    continue
                local = deterministic_support_indices(support_item["rows"].iloc[support_all], support_size)
                support_idx = np.asarray(support_all, dtype=np.int64)[local]
                support_q = tensor_slice(support_item, "q", support_idx, device)
                support_y = tensor_slice(support_item, "y", support_idx, device)
                support_w = tensor_slice(support_item, "weights", support_idx, device)
                delta = encoder(support_q, support_y, support_w)
                delta_norm = float(torch.linalg.vector_norm(delta).detach().cpu())
            qrows = query_item["rows"].iloc[qidx].reset_index(drop=True)
            q = tensor_slice(query_item, "q", qidx, device)
            scores = score_from_q(model, q, delta=delta, scale=shift_scale).detach().cpu().numpy()
            summary, per = r208.compute_metrics(scores, query_item["y"][qidx], qrows, "Round243_Phase1", variant, unit)
            summary.update(
                {
                    "cell_line": cell,
                    "adapt_support_size": int(support_size if encoder is not None else 0),
                    "shift_scale": float(shift_scale if encoder is not None else 0.0),
                    "delta_norm": delta_norm,
                }
            )
            cell_summaries.append(summary)
            if keep_per_unit:
                per["adapt_support_size"] = int(support_size if encoder is not None else 0)
                per["shift_scale"] = float(shift_scale if encoder is not None else 0.0)
                per_rows.append(per)
        report_progress(len(query_item["by_cell"]), len(query_item["by_cell"]), label=f"{variant}: cells inspected")
    return pd.DataFrame(cell_summaries), pd.concat(per_rows, ignore_index=True) if per_rows else pd.DataFrame()


def better_by_metrics(candidate: dict, incumbent: Optional[dict], keys: Iterable[str]) -> bool:
    if incumbent is None:
        return True
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


def train_crosscell_base(
    r211,
    r208,
    kg_ckpt: dict,
    labels: list[str],
    tables: dict,
    device: torch.device,
    smoke: bool,
) -> tuple[nn.Module, pd.DataFrame, dict, Path]:
    report_step('Preparing cross-cell readout training')
    cfg = deepcopy(READOUT_CFG)
    cfg["readout_id"] = "r243_phase1_base_round211_h256_cell_disjoint"
    if smoke:
        cfg["max_epochs"] = 1
        cfg["min_epochs"] = 1
        cfg["patience"] = 1
    target_init, target_audit = r211.target_embeddings_from_ckpt(kg_ckpt, labels, cfg["readout_id"])
    model = r211.DiagReadout(target_init, cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    train = tables["meta_train"]
    val = tables["meta_val"]
    tensors = {
        name: {
            "x": torch.tensor(item["x"], dtype=torch.float32, device=device),
            "y": torch.tensor(item["y"], dtype=torch.float32, device=device),
            "w": torch.tensor(item["weights"], dtype=torch.float32, device=device),
        }
        for name, item in {"meta_train": train, "meta_val": val}.items()
    }
    rng = np.random.default_rng(SEED)
    best_state = None
    best_val = None
    best_epoch = 0
    stale = 0
    history = []
    batch_size = 256
    report_progress(0, int(cfg["max_epochs"]), label='crosscell base epochs')
    report_metrics(cfg, label="Training settings")
    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        model.train()
        order = rng.permutation(len(train["x"]))
        losses = []
        for start in range(0, len(order), batch_size):
            report_progress(start, len(order), label=f"Epoch {epoch} training rows")
            idx = torch.tensor(order[start : start + batch_size], dtype=torch.long, device=device)
            xb = tensors["meta_train"]["x"][idx]
            yb = tensors["meta_train"]["y"][idx]
            wb = tensors["meta_train"]["w"][idx]
            logits = model(xb)
            loss = retrieval_loss_from_logits(logits, yb, wb) + model.regularization()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        report_progress(len(order), len(order), label=f"Epoch {epoch} training rows")
        report_step(f"Epoch {epoch}: evaluating existing training and validation metrics")
        val_scores = predict_from_q_model(model, val["x"], device, batch_size=1024)
        val_summary, _ = r208.compute_metrics(val_scores, val["y"], val["rows"], "Round243_Phase1", cfg["readout_id"], "drug-mean")
        train_scores = predict_from_q_model(model, train["x"], device, batch_size=1024)
        train_summary, _ = r208.compute_metrics(train_scores, train["y"], train["rows"], "Round243_Phase1", cfg["readout_id"], "drug-mean")
        rec = {
            "stage": "phase1_base_cell_disjoint",
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_Hit@10": float(train_summary["Hit@10"]),
            "train_MeanAP": float(train_summary["MeanAP"]),
            "val_Hit@10": float(val_summary["Hit@10"]),
            "val_MeanAP": float(val_summary["MeanAP"]),
            "val_NDCG@10": float(val_summary["NDCG@10"]),
            "val_AUROC": float(val_summary["AUROC"]),
        }
        history.append(rec)
        report_progress(epoch, int(cfg["max_epochs"]), label='crosscell base epochs')
        report_metrics(
            {key: rec[key] for key in ("train_loss", "val_Hit@10", "val_MeanAP")},
            label=f"Epoch {epoch}/{int(cfg['max_epochs'])}",
        )
        if epoch == 1 or epoch % 5 == 0:
            log(
                "phase1 cell-disjoint base "
                f"epoch={epoch} val_Hit@10={rec['val_Hit@10']:.4f} val_MeanAP={rec['val_MeanAP']:.4f}"
            )
        if better_by_metrics(
            val_summary,
            best_val,
            ["Hit@10", "MeanAP", "Hit@5", "NDCG@10"],
        ):
            best_val = val_summary
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch >= int(cfg["min_epochs"]) and stale >= int(cfg["patience"]):
            break
    if best_state is None or best_val is None:
        raise RuntimeError("No phase1 cell-disjoint base checkpoint selected")
    report_metrics(
        {"epochs_run": len(history), "best_epoch": best_epoch, "Hit@10": best_val["Hit@10"], "MeanAP": best_val["MeanAP"]},
        label="Selected readout validation metrics",
    )
    model.load_state_dict(best_state)
    ckpt_dir = CHECKPOINTS / cfg["readout_id"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "readout_checkpoint.pth"
    torch.save(
        {
            "stage": "phase1_base_cell_disjoint",
            "state_dict": best_state,
            "selected": {"best_epoch": best_epoch, "best_val": best_val},
            "readout_cfg": cfg,
            "target_audit": target_audit,
            "labels": labels,
            "kg_flow": kg_ckpt.get("flow", {}),
        },
        ckpt_path,
    )
    return model, pd.DataFrame(history), {"best_epoch": best_epoch, "best_val": best_val, "target_audit": target_audit}, ckpt_path


def train_support_encoder(
    model: nn.Module,
    tables: dict,
    r208,
    device: torch.device,
    cfg: dict,
    smoke: bool,
) -> tuple[SupportDeltaEncoder, pd.DataFrame, dict, Path]:
    report_step('Precomputing query representations for support-encoder training')
    for item in tables.values():
        precompute_q(model, item, device)
    support_target = current_target_tensor(model).detach()
    encoder = SupportDeltaEncoder(
        support_target,
        hidden_dim=int(cfg["hidden_dim"]),
        dropout=float(cfg["dropout"]),
        max_delta_norm=float(cfg["max_delta_norm"]),
    ).to(device)
    opt = torch.optim.AdamW(encoder.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    rng = np.random.default_rng(SEED + 42)
    train = tables["adapter_train"]
    cells = [
        c
        for c, idx in train["by_cell"].items()
        if len(idx) >= int(cfg["support_size"]) + max(2, int(cfg["query_size"]))
    ]
    if not cells:
        raise RuntimeError("No eligible cells for support encoder training")
    max_epochs = 1 if smoke else int(cfg["max_epochs"])
    min_epochs = 1 if smoke else int(cfg["min_epochs"])
    tasks_per_epoch = 4 if smoke else int(cfg["tasks_per_epoch"])
    best = None
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    report_progress(0, max_epochs, label='support encoder epochs')
    report_metrics(cfg, label="Training settings")
    for epoch in range(1, max_epochs + 1):
        encoder.train()
        losses = []
        for _ in range(tasks_per_epoch):
            report_progress(_, tasks_per_epoch, label=f"Epoch {epoch} training episodes")
            cell = str(rng.choice(cells))
            all_idx = np.asarray(train["by_cell"][cell], dtype=np.int64)
            perm = rng.permutation(all_idx)
            support = perm[: int(cfg["support_size"])]
            rest = perm[int(cfg["support_size"]) :]
            if len(rest) == 0:
                rest = perm
            query = sample_indices(rest, int(cfg["query_size"]), rng, replace=len(rest) < int(cfg["query_size"]))
            support_q = tensor_slice(train, "q", support, device)
            support_y = tensor_slice(train, "y", support, device)
            support_w = tensor_slice(train, "weights", support, device)
            query_q = tensor_slice(train, "q", query, device)
            query_y = tensor_slice(train, "y", query, device)
            query_w = tensor_slice(train, "weights", query, device)
            delta = encoder(support_q, support_y, support_w)
            logits = score_from_q(model, query_q, delta=delta, scale=float(cfg["shift_scale"]))
            loss = retrieval_loss_from_logits(logits, query_y, query_w) + float(cfg["delta_l2"]) * torch.square(delta).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        report_progress(tasks_per_epoch, tasks_per_epoch, label=f"Epoch {epoch} training episodes")
        val_cells, _ = evaluate_by_cell(
            r208,
            model,
            encoder,
            tables["pseudo_support"],
            tables["pseudo_val"],
            device,
            f"{cfg['variant']}_adapt",
            int(cfg["support_size"]),
            float(cfg["shift_scale"]),
            "drug-mean",
            keep_per_unit=False,
        )
        val_macro = metric_macro_from_cells(val_cells, f"{cfg['variant']}_adapt", "pseudo_val", "drug-mean", "Round243_support_delta_encoder")
        rec = {
            "stage": "phase1_support_encoder",
            "variant": str(cfg["variant"]),
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_Hit@10": float(val_macro["Hit@10"]),
            "val_MeanAP": float(val_macro["MeanAP"]),
            "val_NDCG@10": float(val_macro["NDCG@10"]),
            "val_AUROC": float(val_macro["AUROC"]),
            "delta_norm_mean": float(val_macro.get("delta_norm_mean", float("nan"))),
        }
        history.append(rec)
        report_progress(epoch, max_epochs, label='support encoder epochs')
        report_metrics(
            {key: rec[key] for key in ("train_loss", "val_Hit@10", "val_MeanAP")},
            label=f"Epoch {epoch}/{max_epochs}",
        )
        if epoch == 1 or epoch % 5 == 0:
            log(
                "phase1 support encoder "
                f"epoch={epoch} val_Hit@10={rec['val_Hit@10']:.4f} val_MeanAP={rec['val_MeanAP']:.4f}"
            )
        if better_by_metrics(
            {"Hit@10": rec["val_Hit@10"], "MeanAP": rec["val_MeanAP"], "NDCG@10": rec["val_NDCG@10"], "AUROC": rec["val_AUROC"]},
            best,
            ["Hit@10", "MeanAP", "NDCG@10", "AUROC"],
        ):
            best = {"Hit@10": rec["val_Hit@10"], "MeanAP": rec["val_MeanAP"], "NDCG@10": rec["val_NDCG@10"], "AUROC": rec["val_AUROC"]}
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch >= min_epochs and stale >= int(cfg["patience"]):
            break
    if best_state is None or best is None:
        raise RuntimeError("No support encoder checkpoint selected")
    report_metrics(
        {"epochs_run": len(history), "best_epoch": best_epoch, "Hit@10": best["Hit@10"], "MeanAP": best["MeanAP"]},
        label="Selected support encoder validation metrics",
    )
    encoder.load_state_dict(best_state)
    ckpt_path = CHECKPOINTS / f"{cfg['variant']}.pth"
    torch.save(
        {
            "stage": "phase1_support_encoder",
            "model_state": best_state,
            "best_epoch": best_epoch,
            "best_val": best,
            "cfg": cfg,
        },
        ckpt_path,
    )
    return encoder, pd.DataFrame(history), {"best_epoch": best_epoch, "best_val": best}, ckpt_path


def load_base_model_from_checkpoint(
    r211,
    kg_ckpt: dict,
    labels: list[str],
    ckpt_path: Path,
    device: torch.device,
) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = deepcopy(ckpt.get("readout_cfg", READOUT_CFG))
    target_init, _ = r211.target_embeddings_from_ckpt(kg_ckpt, labels, cfg.get("readout_id", "r243_loaded_base"))
    model = r211.DiagReadout(target_init, cfg).to(device)
    state = ckpt.get("state_dict", ckpt.get("model_state", ckpt))
    model.load_state_dict(state)
    model.eval()
    return model


def support_sweep_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs = []
    for shift_scale in args.sweep_shift_scales:
        for max_delta_norm in args.sweep_max_delta_norms:
            for delta_l2 in args.sweep_delta_l2s:
                cfg = deepcopy(SUPPORT_ENCODER_CFG)
                cfg["shift_scale"] = float(shift_scale)
                cfg["max_delta_norm"] = float(max_delta_norm)
                cfg["delta_l2"] = float(delta_l2)
                cfg["variant"] = (
                    "r243_support_sweep"
                    f"_h{cfg['hidden_dim']}"
                    f"_sc{fmt_float(float(shift_scale))}"
                    f"_l2{fmt_float(float(delta_l2))}"
                    f"_mn{fmt_float(float(max_delta_norm))}"
                    f"_do{fmt_float(float(cfg['dropout']))}"
                )
                if args.smoke:
                    cfg["variant"] += "_smoke"
                configs.append(cfg)
    return configs


def write_phase1_pc3(args: argparse.Namespace, device: torch.device) -> Path:
    ensure_dirs()
    set_seed()
    kg_path = resolve_kg_checkpoint(args.kg_checkpoint)
    r211 = load_round211_module()
    r211.ensure_dirs()
    r176 = r211.load_round176_module()
    _landmarks, labels, rows, x, y = r176.load_response()
    if int(x.shape[1]) != 978:
        raise RuntimeError(f"Phase1 PC3 expected 978 input features, observed {x.shape[1]}")
    if len(labels) != 668:
        raise RuntimeError(f"Phase1 PC3 expected 668 labels, observed {len(labels)}")
    data = r211.split_data(rows, x, y, r176)
    kg_ckpt = torch.load(kg_path, map_location="cpu", weights_only=False)
    cfg = deepcopy(READOUT_CFG)
    if args.smoke:
        cfg["readout_id"] = "r243_smoke_round211_h256"
        cfg["max_epochs"] = 1
        cfg["min_epochs"] = 1
        cfg["patience"] = 1
    test_summary, history_df = r211.train_readout(kg_ckpt["flow"], kg_ckpt, data, labels, r176, device, cfg)
    flow_id = f"{kg_ckpt['flow']['flow_id']}_{cfg['readout_id']}"
    metric_src = SOURCE / f"metric_summary_{flow_id}.tsv"
    per_src = SOURCE / f"per_drug_metrics_{flow_id}.tsv"
    hist_src = SOURCE / f"readout_history_{flow_id}.tsv"
    metric_df = pd.read_csv(metric_src, sep="\t")
    metric_df["stage"] = "phase1_pc3_closed_set"
    metric_df["source"] = "Round243_phase1_pc3_round211_architecture"
    metric_df["evaluation_family"] = "seen_668"
    metric_df["candidate_space"] = 668
    metric_df.to_csv(SOURCE / "phase1_pc3_metric_summary.tsv", sep="\t", index=False)
    metric_df.to_csv(SOURCE / "phase1_metric_summary.tsv", sep="\t", index=False)
    if per_src.exists():
        per = pd.read_csv(per_src, sep="\t")
        per["stage"] = "phase1_pc3_closed_set"
        per.to_csv(SOURCE / "phase1_pc3_per_unit_metrics.tsv", sep="\t", index=False)
        per.to_csv(SOURCE / "phase1_per_unit_metrics.tsv", sep="\t", index=False)
    hist = pd.read_csv(hist_src, sep="\t") if hist_src.exists() else history_df
    hist["stage"] = "phase1_pc3_closed_set"
    hist.to_csv(SOURCE / "phase1_pc3_training_history.tsv", sep="\t", index=False)
    hist.to_csv(SOURCE / "phase1_training_history.tsv", sep="\t", index=False)
    ckpt_path = CHECKPOINTS / flow_id / "readout_checkpoint.pth"
    selected = {
        "phase1_pc3_checkpoint": str(ckpt_path),
        "phase1_pc3_checkpoint_sha256": sha256_file(ckpt_path),
        "kg_checkpoint": str(kg_path),
        "kg_checkpoint_sha256": sha256_file(kg_path),
        "flow_id": flow_id,
        "test_summary": test_summary,
    }
    (SOURCE / "phase1_selected_checkpoint.json").write_text(json.dumps(json_ready(selected), indent=2), encoding="utf-8")
    write_method_lock(extra={"phase1_pc3": selected})
    update_combined_summaries()
    write_audit(extra_lines=[
        f"- Phase 1 PC3 loaded KG checkpoint: `{kg_path}`.",
        "- Phase 1 PC3 used Round211 DiagReadout with hidden=256 and candidate space=668.",
        "- KG pretraining was not rerun; only the downstream readout/profile side and target readout parameters were trained.",
    ])
    log(f"phase1 PC3 wrote checkpoint {ckpt_path}")
    return ckpt_path


def add_metric_context(df: pd.DataFrame, claim_scope: str, split_role: str) -> pd.DataFrame:
    out = df.copy()
    out["claim_scope"] = claim_scope
    out["split_role"] = split_role
    return out


def phase1_eval_set(
    r208,
    base: nn.Module,
    encoder: SupportDeltaEncoder,
    tables: dict[str, dict],
    cfg: dict[str, Any],
    device: torch.device,
    support_key: str,
    test_key: str,
    profile_key: str,
    claim_scope: str,
    split_role: str,
    source: str,
    keep_per_unit: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    no_variant = f"r243_{split_role}_no_adapt"
    adapt_variant = f"{cfg['variant']}_{split_role}_adapt"
    profile_variant = f"{cfg['variant']}_{split_role}_adapt_profile"
    no_cells, no_per = evaluate_by_cell(
        r208,
        base,
        None,
        tables[support_key],
        tables[test_key],
        device,
        no_variant,
        0,
        0.0,
        "drug-mean",
        keep_per_unit=keep_per_unit,
    )
    adapt_cells, adapt_per = evaluate_by_cell(
        r208,
        base,
        encoder,
        tables[support_key],
        tables[test_key],
        device,
        adapt_variant,
        int(cfg["support_size"]),
        float(cfg["shift_scale"]),
        "drug-mean",
        keep_per_unit=keep_per_unit,
    )
    profile_cells, profile_per = evaluate_by_cell(
        r208,
        base,
        encoder,
        tables[support_key],
        tables[profile_key],
        device,
        profile_variant,
        int(cfg["support_size"]),
        float(cfg["shift_scale"]),
        "true-profile-specific",
        keep_per_unit=keep_per_unit,
    )
    metric_rows = [
        metric_macro_from_cells(no_cells, no_variant, test_key, "drug-mean", source),
        metric_macro_from_cells(adapt_cells, adapt_variant, test_key, "drug-mean", source),
        metric_macro_from_cells(profile_cells, profile_variant, profile_key, "true-profile-specific", source),
    ]
    metrics = add_metric_context(pd.DataFrame(metric_rows), claim_scope, split_role)
    per = pd.concat(
        [df for df in [no_per, adapt_per, profile_per] if not df.empty],
        ignore_index=True,
        sort=False,
    )
    if len(per):
        per = add_metric_context(per, claim_scope, split_role)
    return metrics, per


def write_phase1_unseencell(args: argparse.Namespace, device: torch.device) -> Path:
    ensure_dirs()
    set_seed()
    kg_path = resolve_kg_checkpoint(args.kg_checkpoint)
    r211 = load_round211_module()
    r208 = load_round208_module()
    kg_ckpt = torch.load(kg_path, map_location="cpu", weights_only=False)
    target_entities = [str(x) for x in kg_ckpt["target_entities"]]
    labels = [x.split(":", 1)[1] if ":" in x else x for x in target_entities]
    if len(labels) != 668 or len(set(labels)) != 668:
        raise RuntimeError(f"Expected 668 unique target labels, observed {len(labels)} / {len(set(labels))}")
    cell_sets = choose_round243_cells(
        r208,
        args.pseudo_unseen_cells,
        args.official_unseen_cells,
        args.seen_diagnostic_cells,
    )
    tables, data_audit = build_round243_tables(r208, labels, cell_sets, device)
    write_round243_audit_tables(tables, data_audit)
    for item_name, item in tables.items():
        if item["x"].shape[1] != 978:
            raise RuntimeError(f"{item_name} expected 978 input features, observed {item['x'].shape[1]}")
        if item["y"].shape[1] != 668:
            raise RuntimeError(f"{item_name} expected 668 labels, observed {item['y'].shape[1]}")
    base, base_history, base_selected, base_ckpt = train_crosscell_base(
        r211, r208, kg_ckpt, labels, tables, device, smoke=bool(args.smoke)
    )
    cfg = deepcopy(SUPPORT_ENCODER_CFG)
    if args.smoke:
        cfg["variant"] = "r243_smoke_support_delta_encoder"
    encoder, enc_history, enc_selected, enc_ckpt = train_support_encoder(
        base, tables, r208, device, cfg, smoke=bool(args.smoke)
    )
    official_metrics, official_per = phase1_eval_set(
        r208,
        base,
        encoder,
        tables,
        cfg,
        device,
        "official_support",
        "official_test",
        "official_test_profile",
        "unseen_cell_main",
        "official_unseen",
        "Round243_phase1_unseencell_main",
    )
    seen_metrics, seen_per = phase1_eval_set(
        r208,
        base,
        encoder,
        tables,
        cfg,
        device,
        "seen_support",
        "seen_test",
        "seen_test_profile",
        "seen_cell_diagnostic",
        "seen_diagnostic",
        "Round243_phase1_seen_diagnostic",
    )
    pseudo_val_cells, _ = evaluate_by_cell(
        r208,
        base,
        encoder,
        tables["pseudo_support"],
        tables["pseudo_val"],
        device,
        f"{cfg['variant']}_pseudo_val_adapt",
        int(cfg["support_size"]),
        float(cfg["shift_scale"]),
        "drug-mean",
        keep_per_unit=False,
    )
    pseudo_metrics = add_metric_context(
        pd.DataFrame([
            metric_macro_from_cells(
                pseudo_val_cells,
                f"{cfg['variant']}_pseudo_val_adapt",
                "pseudo_val",
                "drug-mean",
                "Round243_phase1_pseudo_unseen_validation",
            )
        ]),
        "model_selection_diagnostic",
        "pseudo_unseen_validation",
    )
    metrics = pd.concat([official_metrics, pseudo_metrics, seen_metrics], ignore_index=True, sort=False)
    metrics["stage"] = "phase1_unseencell_support_adapter"
    metrics.to_csv(SOURCE / "phase1_unseencell_metric_summary.tsv", sep="\t", index=False)
    official_metrics.to_csv(SOURCE / "metric_summary_official_unseen_main.tsv", sep="\t", index=False)
    seen_metrics.to_csv(SOURCE / "metric_summary_seen_cell_diagnostic.tsv", sep="\t", index=False)
    pseudo_metrics.to_csv(SOURCE / "metric_summary_pseudo_unseen_validation.tsv", sep="\t", index=False)
    phase1_existing = pd.read_csv(SOURCE / "phase1_metric_summary.tsv", sep="\t") if (SOURCE / "phase1_metric_summary.tsv").exists() else pd.DataFrame()
    pd.concat([phase1_existing, metrics], ignore_index=True, sort=False).to_csv(SOURCE / "phase1_metric_summary.tsv", sep="\t", index=False)
    per = pd.concat(
        [df for df in [official_per, seen_per] if not df.empty],
        ignore_index=True,
        sort=False,
    )
    if len(per):
        per["stage"] = "phase1_unseencell_support_adapter"
        per.to_csv(SOURCE / "phase1_unseencell_per_unit_metrics.tsv", sep="\t", index=False)
        official_per.to_csv(SOURCE / "per_unit_metrics_official_unseen_main.tsv", sep="\t", index=False)
        seen_per.to_csv(SOURCE / "per_unit_metrics_seen_cell_diagnostic.tsv", sep="\t", index=False)
        existing_per = pd.read_csv(SOURCE / "phase1_per_unit_metrics.tsv", sep="\t") if (SOURCE / "phase1_per_unit_metrics.tsv").exists() else pd.DataFrame()
        pd.concat([existing_per, per], ignore_index=True, sort=False).to_csv(SOURCE / "phase1_per_unit_metrics.tsv", sep="\t", index=False)
    history = pd.concat([base_history, enc_history], ignore_index=True, sort=False)
    history.to_csv(SOURCE / "phase1_unseencell_training_history.tsv", sep="\t", index=False)
    existing_hist = pd.read_csv(SOURCE / "phase1_training_history.tsv", sep="\t") if (SOURCE / "phase1_training_history.tsv").exists() else pd.DataFrame()
    pd.concat([existing_hist, history], ignore_index=True, sort=False).to_csv(SOURCE / "phase1_training_history.tsv", sep="\t", index=False)
    selected = {
        "phase1_unseencell_base_checkpoint": str(base_ckpt),
        "phase1_unseencell_base_checkpoint_sha256": sha256_file(base_ckpt),
        "phase1_support_encoder_checkpoint": str(enc_ckpt),
        "phase1_support_encoder_checkpoint_sha256": sha256_file(enc_ckpt),
        "base_selected": base_selected,
        "support_encoder_selected": enc_selected,
        "official_unseen_cells": data_audit["official_unseen_cells"],
        "seen_diagnostic_cells": data_audit["seen_diagnostic_cells"],
        "pseudo_unseen_val_cells": data_audit["pseudo_unseen_val_cells"],
        "base_train_cells": data_audit["base_train_cells"],
        "data_audit": data_audit,
        "kg_checkpoint": str(kg_path),
        "kg_checkpoint_sha256": sha256_file(kg_path),
    }
    (SOURCE / "phase1_unseencell_selected_checkpoint.json").write_text(json.dumps(json_ready(selected), indent=2), encoding="utf-8")
    write_method_lock(extra={"phase1_unseencell": selected})
    update_combined_summaries()
    write_audit(extra_lines=[
        "- Phase 1 cell-generalization support adaptation kept target side in the 668-label Round211 KG-readout space.",
        "- Base readout excluded official unseen cells and pseudo-unseen validation cells from base training.",
        "- Support encoder was trained on adapter_train/base cells and selected on pseudo-unseen validation cells.",
        "- Official unseen cells were used only for final main evaluation.",
        "- Seen-cell panel was evaluated only as diagnostic.",
    ])
    log(f"phase1 unseen-cell wrote base checkpoint {base_ckpt} and support checkpoint {enc_ckpt}")
    return base_ckpt


def write_protocol_audit(args: argparse.Namespace, device: torch.device) -> None:
    ensure_dirs()
    kg_path = resolve_kg_checkpoint(args.kg_checkpoint)
    r208 = load_round208_module()
    kg_ckpt = torch.load(kg_path, map_location="cpu", weights_only=False)
    target_entities = [str(x) for x in kg_ckpt["target_entities"]]
    labels = [x.split(":", 1)[1] if ":" in x else x for x in target_entities]
    if len(labels) != 668 or len(set(labels)) != 668:
        raise RuntimeError(f"Expected 668 unique target labels, observed {len(labels)} / {len(set(labels))}")
    cell_sets = choose_round243_cells(
        r208,
        args.pseudo_unseen_cells,
        args.official_unseen_cells,
        args.seen_diagnostic_cells,
    )
    tables, data_audit = build_round243_tables(r208, labels, cell_sets, device)
    write_round243_audit_tables(tables, data_audit)
    (SOURCE / "round243_protocol_audit.json").write_text(json.dumps(json_ready(data_audit), indent=2), encoding="utf-8")
    write_method_lock(extra={"protocol_audit": data_audit})
    write_audit(extra_lines=[
        "- Protocol audit generated explicit cell, drug, and support/query split tables before full training.",
        f"- Official unseen cells: `{','.join(data_audit['official_unseen_cells'])}`.",
        f"- Pseudo-unseen validation cells: `{','.join(data_audit['pseudo_unseen_val_cells'])}`.",
        f"- Seen-cell diagnostic cells: `{','.join(data_audit['seen_diagnostic_cells'])}`.",
    ])
    log(f"protocol audit wrote {SOURCE / 'cell_split_audit.tsv'}")


def write_support_sweep(args: argparse.Namespace, device: torch.device) -> None:
    ensure_dirs()
    set_seed()
    base_manifest = SOURCE / "phase1_unseencell_selected_checkpoint.json"
    if not base_manifest.exists():
        raise FileNotFoundError(f"Run phase1_unseencell first; missing {base_manifest}")
    base_payload = json.loads(base_manifest.read_text(encoding="utf-8"))
    base_ckpt_path = Path(base_payload["phase1_unseencell_base_checkpoint"])
    if not base_ckpt_path.exists():
        raise FileNotFoundError(base_ckpt_path)
    kg_path = resolve_kg_checkpoint(args.kg_checkpoint)
    r211 = load_round211_module()
    r208 = load_round208_module()
    kg_ckpt = torch.load(kg_path, map_location="cpu", weights_only=False)
    target_entities = [str(x) for x in kg_ckpt["target_entities"]]
    labels = [x.split(":", 1)[1] if ":" in x else x for x in target_entities]
    cell_sets = choose_round243_cells(
        r208,
        args.pseudo_unseen_cells,
        args.official_unseen_cells,
        args.seen_diagnostic_cells,
    )
    tables, data_audit = build_round243_tables(r208, labels, cell_sets, device)
    write_round243_audit_tables(tables, data_audit)
    base = load_base_model_from_checkpoint(r211, kg_ckpt, labels, base_ckpt_path, device)
    metric_parts = []
    per_parts = []
    history_parts = []
    selected_rows = []
    for cfg in support_sweep_configs(args):
        log(f"support sweep training {cfg['variant']}")
        encoder, enc_history, enc_selected, enc_ckpt = train_support_encoder(
            base, tables, r208, device, cfg, smoke=bool(args.smoke)
        )
        official_metrics, official_per = phase1_eval_set(
            r208,
            base,
            encoder,
            tables,
            cfg,
            device,
            "official_support",
            "official_test",
            "official_test_profile",
            "unseen_cell_main",
            "official_unseen",
            "Round243_support_sweep_unseencell_main",
        )
        seen_metrics, seen_per = phase1_eval_set(
            r208,
            base,
            encoder,
            tables,
            cfg,
            device,
            "seen_support",
            "seen_test",
            "seen_test_profile",
            "seen_cell_diagnostic",
            "seen_diagnostic",
            "Round243_support_sweep_seen_diagnostic",
        )
        pseudo_val_cells, _ = evaluate_by_cell(
            r208,
            base,
            encoder,
            tables["pseudo_support"],
            tables["pseudo_val"],
            device,
            f"{cfg['variant']}_pseudo_val_adapt",
            int(cfg["support_size"]),
            float(cfg["shift_scale"]),
            "drug-mean",
            keep_per_unit=False,
        )
        pseudo_metrics = add_metric_context(
            pd.DataFrame([
                metric_macro_from_cells(
                    pseudo_val_cells,
                    f"{cfg['variant']}_pseudo_val_adapt",
                    "pseudo_val",
                    "drug-mean",
                    "Round243_support_sweep_pseudo_unseen_validation",
                )
            ]),
            "model_selection_diagnostic",
            "pseudo_unseen_validation",
        )
        metrics = pd.concat([official_metrics, pseudo_metrics, seen_metrics], ignore_index=True, sort=False)
        metrics["stage"] = "phase1_support_sweep"
        metric_parts.append(metrics)
        per = pd.concat([df for df in [official_per, seen_per] if not df.empty], ignore_index=True, sort=False)
        if len(per):
            per["stage"] = "phase1_support_sweep"
            per_parts.append(per)
        hist = enc_history.copy()
        hist["sweep_variant"] = cfg["variant"]
        history_parts.append(hist)
        selected_rows.append(
            {
                "variant": cfg["variant"],
                "checkpoint": str(enc_ckpt),
                "checkpoint_sha256": sha256_file(enc_ckpt),
                "best_epoch": int(enc_selected["best_epoch"]),
                **{f"pseudo_val_{k}": v for k, v in enc_selected["best_val"].items()},
                "shift_scale": float(cfg["shift_scale"]),
                "max_delta_norm": float(cfg["max_delta_norm"]),
                "delta_l2": float(cfg["delta_l2"]),
            }
        )
    all_metrics = pd.concat(metric_parts, ignore_index=True, sort=False)
    all_metrics.to_csv(SOURCE / "support_sweep_metric_summary.tsv", sep="\t", index=False)
    official_main = all_metrics[all_metrics["claim_scope"].eq("unseen_cell_main")].copy()
    official_main.to_csv(SOURCE / "support_sweep_official_unseen_main.tsv", sep="\t", index=False)
    pd.DataFrame(selected_rows).to_csv(SOURCE / "support_sweep_selection.tsv", sep="\t", index=False)
    pd.concat(history_parts, ignore_index=True, sort=False).to_csv(SOURCE / "support_sweep_training_history.tsv", sep="\t", index=False)
    if per_parts:
        pd.concat(per_parts, ignore_index=True, sort=False).to_csv(
            SOURCE / "support_sweep_per_unit_metrics.tsv.gz",
            sep="\t",
            index=False,
            compression="gzip",
        )
    write_method_lock(
        extra={
            "support_sweep": {
                "base_checkpoint": str(base_ckpt_path),
                "base_checkpoint_sha256": sha256_file(base_ckpt_path),
                "grid": {
                    "shift_scales": args.sweep_shift_scales,
                    "max_delta_norms": args.sweep_max_delta_norms,
                    "delta_l2s": args.sweep_delta_l2s,
                },
                "selection_table": str(SOURCE / "support_sweep_selection.tsv"),
                "data_audit": data_audit,
            }
        }
    )
    update_combined_summaries()
    write_audit(extra_lines=[
        "- Support-adapter sweep reused the leakage-safe phase1 base checkpoint; KG was not retrained.",
        "- Sweep selection used pseudo-unseen validation cells; official unseen cells remained final evaluation only.",
    ])
    log(f"support sweep wrote {SOURCE / 'support_sweep_metric_summary.tsv'}")


def load_phase1_query_state(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt.get("model_state", ckpt.get("model_state_dict", ckpt)))
    query_state = {
        k: v.detach().clone().float()
        for k, v in state.items()
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
        raise RuntimeError(f"Phase1 query encoder checkpoint missing keys: {missing}")
    audit = {
        "phase1_checkpoint": str(path),
        "phase1_checkpoint_sha256": sha256_file(path),
        "loaded_query_keys": sorted(query_state),
        "state_shapes": {k: list(v.shape) for k, v in query_state.items()},
    }
    return query_state, audit


def default_phase1_checkpoint() -> Path:
    unseencell_manifest = SOURCE / "phase1_unseencell_selected_checkpoint.json"
    if unseencell_manifest.exists():
        payload = json.loads(unseencell_manifest.read_text(encoding="utf-8"))
        path = Path(payload["phase1_unseencell_base_checkpoint"])
        if path.exists():
            return path
    manifest = SOURCE / "phase1_selected_checkpoint.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        path = Path(payload["phase1_pc3_checkpoint"])
        if path.exists():
            return path
    raise FileNotFoundError(
        "No phase1 checkpoint manifest found. Run --phase phase1_unseencell or pass --phase1-checkpoint."
    )


def phase2_variants(round240_module, runs: str, smoke: bool) -> dict[str, dict[str, Any]]:
    selected = [x.strip() for x in runs.split(",") if x.strip()]
    out = {}
    for name in selected:
        extra_spec = None
        if name in round240_module.VARIANTS:
            cfg = deepcopy(round240_module.VARIANTS[name])
            run_id = f"r243_{name}_from_phase1"
        elif name in PHASE2_EXTRA_VARIANTS:
            extra_spec = deepcopy(PHASE2_EXTRA_VARIANTS[name])
            cfg = deepcopy(round240_module.VARIANTS[extra_spec["base"]])
            cfg.update(extra_spec["updates"])
            run_id = f"r243_{name}_from_phase1"
        elif (name.startswith("r243_") or name.startswith("r242_")) and name.endswith("_from_phase1"):
            prefix_len = len("r243_") if name.startswith("r243_") else len("r242_")
            base_name = name[prefix_len : -len("_from_phase1")]
            if base_name in round240_module.VARIANTS:
                cfg = deepcopy(round240_module.VARIANTS[base_name])
            elif base_name in PHASE2_EXTRA_VARIANTS:
                extra_spec = deepcopy(PHASE2_EXTRA_VARIANTS[base_name])
                cfg = deepcopy(round240_module.VARIANTS[extra_spec["base"]])
                cfg.update(extra_spec["updates"])
            else:
                raise ValueError(f"Unknown phase2 variant: {name}")
            run_id = name
        else:
            raise ValueError(f"Unknown phase2 variant: {name}")
        cfg["target_mode"] = "phase1_round211215_encoder_to_round240_fullspace_target"
        cfg["ablation"] = f"round243_{run_id}_phase1_encoder_round240_fullspace"
        if extra_spec is not None:
            cfg["round243_extra_variant_base"] = extra_spec["base"]
            cfg["round243_extra_variant_updates"] = extra_spec["updates"]
            cfg["round243_extra_variant_rationale"] = extra_spec["rationale"]
        if smoke:
            cfg["max_epochs"] = 1
            cfg["min_epochs"] = 1
            cfg["patience"] = 1
        out[run_id] = cfg
    return out


def write_phase2_pc3(args: argparse.Namespace, device: torch.device) -> None:
    ensure_dirs()
    set_seed()
    phase1_ckpt = Path(args.phase1_checkpoint).resolve() if args.phase1_checkpoint else default_phase1_checkpoint()
    query_state, phase1_audit = load_phase1_query_state(phase1_ckpt)
    r240 = load_round240_module()
    module = r240.load_round230_module()
    module.ROUND_ROOT = ROUND_ROOT
    module.SOURCE = SOURCE
    module.QA = QA
    module.CHECKPOINTS = CHECKPOINTS
    module.ensure_dirs()
    module.set_seed(module.SEED)
    wrapper, data, protocol = module.load_h07_data()
    splits = {name: module.subset_split_by_cell(data[name], module.CELL_LINE) for name in ["inner_train", "val", "test"]}
    holdouts = {name: module.subset_holdout_by_cell(data[name], module.CELL_LINE) for name in ["holdout_val", "holdout_test"]}
    module.assert_expected_counts(data, splits, holdouts)
    train_idx = data["train_gene_global_idx"].detach().cpu().numpy().astype(np.int64)
    variants = phase2_variants(r240, args.phase2_runs, bool(args.smoke))
    metrics_all = []
    history_all = []
    per_all = []
    audits = []
    for variant, cfg in variants.items():
        module.set_seed(module.SEED)
        model = r240.build_model(module, data, device, train_idx, cfg, query_state)
        audits.append(
            {
                "variant": variant,
                "phase1_query_encoder_initialized": True,
                "phase1_audit": phase1_audit,
                "target_side": "Round240 trainable untrained MeTar h07 PPI GCN plus trainable unseen label tower",
                "full_target_universe": int(len(data["gene_list"])),
                "seen_label_space": int(len(data["train_label_genes"])),
                "single_epoch_checkpoint_policy": "whole model state selected at one validation epoch",
                "cfg": cfg,
                "build_h07_audit": getattr(model, "build_audit", {}),
            }
        )
        metrics, history, per = r240.train_variant(
            module,
            variant,
            cfg,
            model,
            splits,
            holdouts,
            train_idx,
            device,
            smoke=bool(args.smoke),
        )
        metrics["stage"] = "phase2_pc3_fullspace_extrapolation"
        history["stage"] = "phase2_pc3_fullspace_extrapolation"
        metrics["phase1_initializer"] = str(phase1_ckpt)
        history["phase1_initializer"] = str(phase1_ckpt)
        metrics_all.append(metrics)
        history_all.append(history)
        if per is not None:
            per["stage"] = "phase2_pc3_fullspace_extrapolation"
            per["phase1_initializer"] = str(phase1_ckpt)
            per_all.append(per)
    metrics_df = pd.concat(metrics_all, ignore_index=True)
    history_df = pd.concat(history_all, ignore_index=True)
    metrics_df.to_csv(SOURCE / "phase2_metric_summary.tsv", sep="\t", index=False)
    history_df.to_csv(SOURCE / "phase2_training_history.tsv", sep="\t", index=False)
    if per_all:
        pd.concat(per_all, ignore_index=True).to_csv(
            SOURCE / "phase2_per_unit_metrics.tsv.gz",
            sep="\t",
            index=False,
            compression="gzip",
        )
    (SOURCE / "phase2_target_mode_audit.json").write_text(json.dumps(json_ready(audits), indent=2), encoding="utf-8")
    (SOURCE / "phase2_phase1_checkpoint_audit.json").write_text(json.dumps(json_ready(phase1_audit), indent=2), encoding="utf-8")
    write_method_lock(
        extra={
            "phase2_pc3": {
                "phase1_initializer": phase1_audit,
                "round240_base_script": str(ROUND240_SCRIPT),
                "round230_base_script": str(r240.ROUND230_SCRIPT),
                "counts": module.EXPECTED_COUNTS,
                "variants": variants,
                "extra_variant_specs": PHASE2_EXTRA_VARIANTS,
                "protocol_keys": sorted(protocol.keys()) if isinstance(protocol, dict) else [],
            }
        }
    )
    update_combined_summaries()
    write_audit(extra_lines=[
        "- Phase 2 is the only branch that switches to full-space target-side.",
        "- Target universe was checked through Round240/Round230 expected counts: 16,583 genes and 668 seen labels.",
        "- Holdout target labels are used only in holdout-val/test evaluation, not as train labels.",
        "- Selected checkpoints are whole-model states containing phase1-initialized query encoder and full-space target side from the same epoch.",
    ])
    write_round243_report()
    log(f"phase2 PC3 wrote {SOURCE / 'phase2_metric_summary.tsv'}")


def actual_fullspace_counts(splits: dict[str, dict], holdouts: dict[str, dict], cell_line: str, data: dict) -> dict[str, int]:
    return {
        "cell_line": cell_line,
        "n_genes_full_universe": int(len(data["gene_list"])),
        "n_train_label_genes": int(len(data["train_label_genes"])),
        "inner_train_profiles": int(len(splits["inner_train"]["x"])),
        "val_profiles": int(len(splits["val"]["x"])),
        "test_profiles": int(len(splits["test"]["x"])),
        "holdout_val_profiles": int(len(holdouts["holdout_val"]["x"])),
        "holdout_test_profiles": int(len(holdouts["holdout_test"]["x"])),
    }


def write_phase2_holdout_cell(args: argparse.Namespace, device: torch.device) -> None:
    ensure_dirs()
    set_seed()
    phase1_ckpt = Path(args.phase1_checkpoint).resolve() if args.phase1_checkpoint else default_phase1_checkpoint()
    query_state, phase1_audit = load_phase1_query_state(phase1_ckpt)
    holdout_cell = str(args.holdout_fullspace_cell)
    r208 = load_round208_module()
    splits_v3 = load_splits_v3(r208)
    split_map = official_split_map(splits_v3)
    if split_map.get(holdout_cell) != "holdout_cell":
        raise RuntimeError(f"{holdout_cell} is not an official holdout/test cell in splits_v3")
    r240 = load_round240_module()
    module = r240.load_round230_module()
    module.ROUND_ROOT = ROUND_ROOT
    module.SOURCE = SOURCE
    module.QA = QA
    module.CHECKPOINTS = CHECKPOINTS
    module.ensure_dirs()
    module.set_seed(module.SEED)
    wrapper, data, protocol = module.load_h07_data()
    splits = {name: module.subset_split_by_cell(data[name], holdout_cell) for name in ["inner_train", "val", "test"]}
    holdouts = {name: module.subset_holdout_by_cell(data[name], holdout_cell) for name in ["holdout_val", "holdout_test"]}
    counts = actual_fullspace_counts(splits, holdouts, holdout_cell, data)
    if counts["inner_train_profiles"] == 0 or counts["test_profiles"] == 0 or counts["holdout_test_profiles"] == 0:
        raise RuntimeError(f"Insufficient full-space data for {holdout_cell}: {counts}")
    train_idx = data["train_gene_global_idx"].detach().cpu().numpy().astype(np.int64)
    variants = phase2_variants(r240, args.phase2_runs, bool(args.smoke))
    renamed = {}
    for run_id, cfg in variants.items():
        new_run_id = f"{run_id}_{holdout_cell}"
        cfg = deepcopy(cfg)
        cfg["cell_line"] = holdout_cell
        cfg["target_mode"] = "round243_official_holdout_cell_fullspace_target"
        cfg["ablation"] = f"{cfg['ablation']}_{holdout_cell}"
        renamed[new_run_id] = cfg
    metrics_all = []
    history_all = []
    per_all = []
    audits = []
    for variant, cfg in renamed.items():
        module.set_seed(module.SEED)
        model = r240.build_model(module, data, device, train_idx, cfg, query_state)
        audits.append(
            {
                "variant": variant,
                "cell_line": holdout_cell,
                "claim_scope": "official_holdout_cell_unseen_cell_plus_unseen_target",
                "phase1_query_encoder_initialized": True,
                "phase1_audit": phase1_audit,
                "target_side": "Round240 trainable untrained MeTar h07 PPI GCN plus trainable unseen label tower",
                "full_target_universe": int(len(data["gene_list"])),
                "seen_label_space": int(len(data["train_label_genes"])),
                "single_epoch_checkpoint_policy": "whole model state selected at one validation epoch",
                "counts": counts,
                "cfg": cfg,
                "build_h07_audit": getattr(model, "build_audit", {}),
            }
        )
        metrics, history, per = r240.train_variant(
            module,
            variant,
            cfg,
            model,
            splits,
            holdouts,
            train_idx,
            device,
            smoke=bool(args.smoke),
        )
        for df in [metrics, history]:
            df["stage"] = "phase2_holdout_cell_fullspace_extrapolation"
            df["phase1_initializer"] = str(phase1_ckpt)
            df["cell_line"] = holdout_cell
            df["claim_scope"] = "unseen_cell_plus_unseen_target"
        metrics_all.append(metrics)
        history_all.append(history)
        if per is not None:
            per["stage"] = "phase2_holdout_cell_fullspace_extrapolation"
            per["phase1_initializer"] = str(phase1_ckpt)
            per["cell_line"] = holdout_cell
            per_all.append(per)
    metrics_df = pd.concat(metrics_all, ignore_index=True)
    history_df = pd.concat(history_all, ignore_index=True)
    metrics_df.to_csv(SOURCE / f"phase2_{holdout_cell}_metric_summary.tsv", sep="\t", index=False)
    history_df.to_csv(SOURCE / f"phase2_{holdout_cell}_training_history.tsv", sep="\t", index=False)
    if per_all:
        pd.concat(per_all, ignore_index=True).to_csv(
            SOURCE / f"phase2_{holdout_cell}_per_unit_metrics.tsv.gz",
            sep="\t",
            index=False,
            compression="gzip",
        )
    (SOURCE / f"phase2_{holdout_cell}_target_mode_audit.json").write_text(json.dumps(json_ready(audits), indent=2), encoding="utf-8")
    write_method_lock(
        extra={
            f"phase2_holdout_cell_{holdout_cell}": {
                "phase1_initializer": phase1_audit,
                "round240_base_script": str(ROUND240_SCRIPT),
                "round230_base_script": str(r240.ROUND230_SCRIPT),
                "counts": counts,
                "variants": renamed,
                "protocol_keys": sorted(protocol.keys()) if isinstance(protocol, dict) else [],
            }
        }
    )
    update_combined_summaries()
    write_audit(extra_lines=[
        f"- Phase 2 holdout-cell branch used official holdout cell `{holdout_cell}` for unseen-cell + unseen-target diagnostic.",
        f"- Full-space counts for `{holdout_cell}`: `{counts}`.",
        "- Holdout-cell full-space branch used a whole-model checkpoint from one epoch.",
    ])
    write_round243_report()
    log(f"phase2 {holdout_cell} wrote {SOURCE / f'phase2_{holdout_cell}_metric_summary.tsv'}")


def update_combined_summaries() -> None:
    metric_files = [
        SOURCE / "phase1_metric_summary.tsv",
        SOURCE / "phase2_metric_summary.tsv",
    ]
    metrics = []
    for path in metric_files:
        if path.exists():
            metrics.append(pd.read_csv(path, sep="\t"))
    if metrics:
        pd.concat(metrics, ignore_index=True, sort=False).to_csv(SOURCE / "metric_summary.tsv", sep="\t", index=False)
    history_files = [
        SOURCE / "phase1_training_history.tsv",
        SOURCE / "phase2_training_history.tsv",
    ]
    histories = []
    for path in history_files:
        if path.exists():
            histories.append(pd.read_csv(path, sep="\t"))
    if histories:
        pd.concat(histories, ignore_index=True, sort=False).to_csv(SOURCE / "training_history.tsv", sep="\t", index=False)
    comparisons = [
        pd.DataFrame([ROUND211_BEST_REFERENCE]),
        pd.DataFrame([ROUND215_BEST_REFERENCE]),
        pd.DataFrame([ROUND240_BEST_REFERENCE]),
    ]
    if (SOURCE / "metric_summary.tsv").exists():
        current = pd.read_csv(SOURCE / "metric_summary.tsv", sep="\t")
        current = current.copy()
        if "source" not in current:
            current["source"] = "Round243"
        comparisons.insert(0, current)
    pd.concat(comparisons, ignore_index=True, sort=False).to_csv(
        SOURCE / "round211_215_240_comparison_summary.tsv",
        sep="\t",
        index=False,
    )


def write_method_lock(extra: Optional[dict[str, Any]] = None) -> None:
    ensure_dirs()
    existing: dict[str, Any] = {}
    path = ROUND_ROOT / "method_lock.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    lock = {
        "round": "round243_leakage_safe_unseen_cell_support_adapter",
        "created_at": "2026-05-19",
        "seed": SEED,
        "scope": "Support-conditioned prediction in held-out cell lines with separate training, validation and evaluation compounds",
        "phase1_closed_set_before_unseen_target": {
            "architecture_source": "Round211",
            "input_features": 978,
            "readout_hidden": 256,
            "candidate_space": 668,
            "target_side": "Round211 KG-pretrained target embeddings plus target_delta/readout fine-tuning",
            "kg_checkpoint_policy": "reuse Round211-selected Round210 KG checkpoint; do not retrain KG",
            "default_kg_checkpoint_relative": str(KG_CHECKPOINT_RELATIVE),
            "round215_integration": "support-conditioned query-space delta_q on profile/query side only",
            "forbidden_in_phase1": ["Round240 full-space target side", "16,583-gene target universe"],
            "readout_cfg": READOUT_CFG,
            "support_encoder_cfg": SUPPORT_ENCODER_CFG,
            "cell_protocol": {
                "main_claim": "official unseen cells only",
                "model_selection": "pseudo-unseen train cells only",
                "seen_cell_results": "diagnostic only",
            },
        },
        "phase2_unseen_target_pc3": {
            "architecture_source": "Round240 target-side",
            "cell_line": "PC3",
            "profile_side_initializer": "new phase1 Round211+Round215 checkpoint query_encoder/query_norm",
            "target_side": "Round240 full-space MeTar PPI target-side",
            "target_universe": 16583,
            "seen_label_training_space": 668,
            "objective": "full-space retrieval loss with seen labels scattered into the full target universe",
            "checkpoint_policy": "whole-model checkpoint selected by one validation epoch; no mixed profile/target epochs",
            "primary_metric": "holdout-test full-space top-5% any-hit recovery",
            "diagnostics": ["remove-seen top-5%", "seen-668 Hit@10/MeanAP/AUROC/NDCG@10"],
        },
        "reference_results": {
            "round211": ROUND211_BEST_REFERENCE,
            "round215": ROUND215_BEST_REFERENCE,
            "round240": ROUND240_BEST_REFERENCE,
        },
        "paths": {
            "round_root": str(ROUND_ROOT),
            "round211_script": str(ROUND211_SCRIPT),
            "round208_round215_data_script": str(ROUND208_SCRIPT),
            "round240_script": str(ROUND240_SCRIPT),
        },
    }
    lock.update(existing)
    if extra:
        lock.setdefault("run_artifacts", {}).update(extra)
    path.write_text(json.dumps(json_ready(lock), indent=2), encoding="utf-8")


def write_audit(extra_lines: Optional[list[str]] = None) -> None:
    ensure_dirs()
    existing_extra: list[str] = []
    audit_path = QA / "leakage_and_denominator_audit.md"
    if audit_path.exists():
        old_lines = audit_path.read_text(encoding="utf-8").splitlines()
        in_extra = False
        for line in old_lines:
            if line.strip() == "## Extra run evidence":
                in_extra = True
                continue
            if in_extra and line.startswith("## "):
                break
            if in_extra and line.startswith("- "):
                existing_extra.append(line)
    merged_extra = list(dict.fromkeys(existing_extra + (extra_lines or [])))
    lines = [
        "# Leakage and denominator audit",
        "",
        "## Outputs",
        "",
        "- Model outputs and evaluation records are written to the selected output directory.",
        "- Outputs contain model predictions, evaluation metrics and input-split records.",
        "",
        "## Phase 1 denominator",
        "",
        "- Input dimensionality must be 978 landmark features.",
        "- Candidate target space must be 668 labels.",
        "- KG embeddings come from the Round211-selected Round210 checkpoint.",
        "- KG pretraining is not rerun in this round.",
        "- Round215-style adaptation is query-side only: `q' = normalize(q + scale * delta_q)`.",
        "- Phase 1 does not use the Round240 full-space target side.",
        "- Official unseen cells are not used in base training, adapter training, or model selection.",
        "- Seen-cell target-panel results are diagnostic only and are not mixed with the official unseen-cell main table.",
        "",
        "## Phase 2 denominator",
        "",
        "- Phase 2 PC3 is a seen-cell unseen-target diagnostic under the current split.",
        "- Target universe must be 16,583 genes.",
        "- Seen labels remain the 668-label training space and are scattered into the full universe for training.",
        "- Holdout target labels are not training labels; they are used only for holdout validation/test evaluation.",
        "- The selected checkpoint is a whole-model checkpoint from one epoch.",
        "",
        "## Extra run evidence",
        "",
    ]
    if merged_extra:
        lines.extend(merged_extra)
    else:
        lines.append("- No phase results are available.")
    audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_round243_report() -> None:
    update_combined_summaries()
    table_text = "No metrics available yet."
    compare_path = SOURCE / "round211_215_240_comparison_summary.tsv"
    if compare_path.exists():
        df = pd.read_csv(compare_path, sep="\t")
        show_cols = [
            "source",
            "stage",
            "variant",
            "split",
            "evaluation_family",
            "evaluation_unit",
            "candidate_space",
            "Hit@10",
            "MeanAP",
            "NDCG@10",
            "AUROC",
            "top5pct_success",
            "top5pct_success_remove_seen",
            "best_epoch",
        ]
        cols = [c for c in show_cols if c in df.columns]
        table_text = df[cols].tail(30).to_string(index=False)
    text = f"""# Support-conditioned prediction in held-out cell lines

## Scope

The model uses separate training, model-selection and held-out evaluation
cell lines to evaluate support-conditioned prediction in new cell lines.

## Method boundary

The phase-1 model stays in the Round211 target space: 978 landmark input,
hidden=256 readout, 668 labels, and the fixed Round211-selected KG checkpoint.
Round215 is integrated only as a support-conditioned query-side shift.

Official unseen cells are excluded from base training and adapter training.
Pseudo-unseen train cells are used for model selection. Seen-cell results are
diagnostic only.

## Current comparison table

```text
{table_text}
```
"""
    (ROUND_ROOT / "training_summary.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        type=str,
        default="write_lock",
        choices=[
            "write_lock",
            "audit_protocol",
            "phase1_pc3",
            "phase1_unseencell",
            "support_sweep",
            "phase2_pc3",
            "phase2_holdout_cell",
            "all",
        ],
    )
    parser.add_argument("--kg-checkpoint", type=str, default=None)
    parser.add_argument("--phase1-checkpoint", type=str, default=None)
    parser.add_argument("--official-unseen-cells", type=str, default="round215_heldout_panel")
    parser.add_argument("--seen-diagnostic-cells", type=str, default="round215_train_panel")
    parser.add_argument("--pseudo-unseen-cells", type=str, default="round215_train_panel")
    parser.add_argument("--sweep-shift-scales", nargs="+", type=float, default=[0.05, 0.10, 0.15])
    parser.add_argument("--sweep-max-delta-norms", nargs="+", type=float, default=[1.5, 2.5, 3.5])
    parser.add_argument("--sweep-delta-l2s", nargs="+", type=float, default=[1.0e-3])
    parser.add_argument("--phase2-runs", type=str, default="e8_enc105e6")
    parser.add_argument("--holdout-fullspace-cell", type=str, default="HT29")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if args.phase == "write_lock":
        write_method_lock()
        write_audit()
        update_combined_summaries()
        write_round243_report()
        log("wrote bootstrap method_lock and audit")
        return
    require_torch()
    device = torch.device(resolve_device(args.device))
    log(f"device={device} phase={args.phase} smoke={args.smoke}")
    if args.phase == "audit_protocol":
        write_protocol_audit(args, device)
        write_round243_report()
        return
    if args.phase == "support_sweep":
        write_support_sweep(args, device)
        write_round243_report()
        return
    if args.phase == "phase2_holdout_cell":
        write_phase2_holdout_cell(args, device)
        write_round243_report()
        return
    if args.phase in {"phase1_pc3", "all"}:
        phase1_ckpt = write_phase1_pc3(args, device)
        if args.phase == "all" and not args.phase1_checkpoint:
            args.phase1_checkpoint = str(phase1_ckpt)
    if args.phase in {"phase1_unseencell", "all"}:
        cross_ckpt = write_phase1_unseencell(args, device)
        if args.phase == "all":
            args.phase1_checkpoint = str(cross_ckpt)
    if args.phase in {"phase2_pc3", "all"}:
        write_phase2_pc3(args, device)
    write_round243_report()


if __name__ == "__main__":
    main()
