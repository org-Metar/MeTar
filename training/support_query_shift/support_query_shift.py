from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
METAR_ROOT = ROOT.parent
SOURCE = ROOT / "source_data"
QA = ROOT / "qa"
CHECKPOINTS = ROOT / "checkpoints"
LOGS = ROOT / "logs"

SEED = 20260515
HIT_KS = [1, 5, 10, 20, 30, 50]
META_COLUMNS = ["split", "profile_id", "drug", "cell_line", "n_profiles", "true_targets"]

H07_MED = METAR_ROOT / "bulk_sharedsample_indunseenlabel_20260330_h07_w060_u8e4_med"
H07_SCRIPT = (
    METAR_ROOT
    / "bulk_sharedsample_indunseenlabel_20260330_h07"
    / "scripts"
    / "train_bulk_sharedsample_indunseenlabel_v1_20260330.py"
)
LOCK_PATH = H07_MED / "manifests" / "bulk_protocol_lock_v2.json"
SPLITS_V3_PATH = Path("/home/lwn/claude/metar/splits_v3.json")

ROUND205_ROOT = METAR_ROOT / "claude_round205_kg_pretrain_iterative_optimization_20260514"
A01_FLOW_ID = "r205_a01_reladd2_d300_f32_t020_edge050"
A01_CKPT = (
    ROUND205_ROOT
    / "checkpoints"
    / "process_only"
    / A01_FLOW_ID
    / f"{A01_FLOW_ID}_kg_checkpoint_epoch035.pth"
)

TARGET_CELLS = ["PC3", "A375", "MCF7"]
READOUT_BATCH_SIZE = 512
POOLED_MIN_EPOCHS = 8
POOLED_PATIENCE = 5
DNBCE_WEIGHT = 0.20
TARGET_BIAS_L2 = 5.0e-4
SCORER_L2 = 2.0e-4


def ensure_dirs() -> None:
    for path in [SOURCE, QA, CHECKPOINTS, LOGS]:
        path.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    print(f"[round208] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_targets(value: Iterable[str] | str) -> Tuple[str, ...]:
    if isinstance(value, (list, tuple, set)):
        return tuple(
            str(v).strip()
            for v in value
            if str(v).strip() and str(v).strip() not in {"0", "-666", "nan", "None"}
        )
    text = str(value).strip()
    if text.startswith(("(", "[")):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return tuple(
                    str(v).strip()
                    for v in parsed
                    if str(v).strip() and str(v).strip() not in {"0", "-666", "nan", "None"}
                )
        except Exception:
            pass
    return tuple(g.strip() for g in text.split("|") if g.strip() and g.strip() not in {"0", "-666", "nan", "None"})


def import_h07_module():
    scratch = ROOT / "h07_loader_scratch"
    (scratch / "derived_tables").mkdir(parents=True, exist_ok=True)
    os.environ["METAR_BULK_PROJECT_ROOT"] = str(scratch)
    os.environ["METAR_BULK_PROTOCOL_PATH"] = str(LOCK_PATH)
    spec = importlib.util.spec_from_file_location("round208_h07", H07_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import H07 script: {H07_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gene_from_entity(entity: str) -> str:
    return str(entity).split(":", 1)[1] if ":" in str(entity) else str(entity)


def labels_and_targets_from_a01(device: torch.device) -> Tuple[List[str], np.ndarray, dict]:
    ckpt = torch.load(A01_CKPT, map_location="cpu", weights_only=False)
    target_entities = [str(x) for x in ckpt["target_entities"]]
    labels = [gene_from_entity(x) for x in target_entities]
    if len(labels) != 668 or len(set(labels)) != 668:
        raise RuntimeError(f"Expected 668 unique labels from A01 target_entities, observed {len(labels)} / {len(set(labels))}")
    ent_tensor = ckpt.get("export_entity_weight", ckpt["model_state"]["entity.weight"])
    ent = ent_tensor.detach().cpu().numpy().astype(np.float32)
    ent2id = ckpt["ent2id"]
    target = np.zeros((len(labels), int(ckpt["kg_dim"])), dtype=np.float32)
    rows = []
    for i, (gene, entity) in enumerate(zip(labels, target_entities)):
        idx = ent2id.get(entity)
        if idx is None:
            raise RuntimeError(f"Missing A01 target embedding for {gene}: {entity}")
        target[i] = ent[int(idx)]
        rows.append(
            {
                "gene": gene,
                "target_entity": entity,
                "target_entity_type": entity.split(":", 1)[0] if ":" in entity else "OTHER",
            }
        )
    pd.DataFrame(rows).to_csv(SOURCE / "a01_target_entities.tsv", sep="\t", index=False)
    audit = {
        "a01_checkpoint": str(A01_CKPT),
        "a01_checkpoint_sha256": sha256_file(A01_CKPT),
        "kg_dim": int(ckpt["kg_dim"]),
        "kg_checkpoint_epoch": int(ckpt.get("kg_checkpoint_epoch", 35)),
        "flow": ckpt.get("flow", {}),
        "target_embedding_status": "fixed buffer; not trainable in pooled, inner-loop or meta-update",
        "target_entity_type_counts": pd.Series([r["target_entity_type"] for r in rows]).value_counts().to_dict(),
    }
    return labels, target, audit


def true_target_matrix(rows: pd.DataFrame, labels: Sequence[str]) -> np.ndarray:
    label_to_idx = {gene: i for i, gene in enumerate(labels)}
    y = np.zeros((len(rows), len(labels)), dtype=np.float32)
    for row_i, value in enumerate(rows["true_targets"].astype(str)):
        for gene in split_targets(value):
            idx = label_to_idx.get(gene)
            if idx is not None:
                y[row_i, idx] = 1.0
    if len(rows) and not (y.sum(axis=1) > 0).all():
        raise RuntimeError(f"Non-evaluable rows found: {int((y.sum(axis=1) <= 0).sum())}")
    return y


def row_weights(rows: pd.DataFrame) -> np.ndarray:
    w = np.log1p(np.minimum(rows["n_profiles"].to_numpy(dtype=np.float32), 50.0))
    w = w / max(float(w.mean()), 1.0e-8)
    return w.astype(np.float32)


def build_drugcell_table(
    df: pd.DataFrame,
    x_source: np.ndarray,
    labels: Sequence[str],
    cells: Sequence[str],
    split_name: str,
) -> Tuple[pd.DataFrame, np.ndarray]:
    label_set = set(labels)
    mask = df["cell_line"].astype(str).isin(cells).to_numpy()
    sub = df.loc[mask].copy().reset_index(drop=True)
    rows = []
    xs = []
    x = x_source[mask].astype(np.float32, copy=False)
    for (drug, cell), group_index in sub.groupby(["drug", "cell_line"], sort=True).groups.items():
        idx = np.asarray(list(group_index), dtype=np.int64)
        targets = sorted({g for value in sub.iloc[idx]["target"] for g in split_targets(value) if g in label_set})
        if not targets:
            continue
        rows.append(
            {
                "split": split_name,
                "profile_id": f"{cell}:{drug}",
                "drug": str(drug),
                "cell_line": str(cell),
                "n_profiles": int(len(idx)),
                "true_targets": "|".join(targets),
            }
        )
        xs.append(x[idx].mean(axis=0))
    if not rows:
        return pd.DataFrame(columns=META_COLUMNS), np.zeros((0, x_source.shape[1]), dtype=np.float32)
    return pd.DataFrame(rows, columns=META_COLUMNS), np.vstack(xs).astype(np.float32)


def build_profile_table(
    df: pd.DataFrame,
    x_source: np.ndarray,
    labels: Sequence[str],
    cells: Sequence[str],
    split_name: str,
) -> Tuple[pd.DataFrame, np.ndarray]:
    label_set = set(labels)
    mask = df["cell_line"].astype(str).isin(cells).to_numpy()
    sub = df.loc[mask].copy().reset_index(drop=True)
    x = x_source[mask].astype(np.float32, copy=False)
    records = []
    keep_idx = []
    for i, row in sub.iterrows():
        targets = sorted({g for g in split_targets(row["target"]) if g in label_set})
        if not targets:
            continue
        records.append(
            {
                "split": split_name,
                "profile_id": str(row.get("profile_id", i)),
                "drug": str(row["drug"]),
                "cell_line": str(row["cell_line"]),
                "n_profiles": 1,
                "true_targets": "|".join(targets),
            }
        )
        keep_idx.append(i)
    if not records:
        return pd.DataFrame(columns=META_COLUMNS), np.zeros((0, x_source.shape[1]), dtype=np.float32)
    return pd.DataFrame(records, columns=META_COLUMNS), x[np.asarray(keep_idx, dtype=np.int64)]


def standardize_tables(train_x: np.ndarray, tables: Dict[str, dict]) -> dict:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1.0e-6] = 1.0
    for item in tables.values():
        item["x"] = ((item["x_raw"] - mean) / std).astype(np.float32)
    return {
        "standardization_source": "meta_train drug-cell means",
        "mean_first5": mean.reshape(-1)[:5].astype(float).tolist(),
        "std_first5": std.reshape(-1)[:5].astype(float).tolist(),
    }


def split_to_numpy(split: dict) -> Tuple[pd.DataFrame, np.ndarray]:
    df = split["df"].copy().reset_index(drop=True)
    x = split["x"].detach().cpu().numpy().astype(np.float32) if torch.is_tensor(split["x"]) else split["x"].astype(np.float32)
    return df, x


def prepare_data(labels: Sequence[str], target_cells: Sequence[str], mode: str, device: torch.device) -> Tuple[Dict[str, dict], dict]:
    module = import_h07_module()
    protocol = module.bulk_base.load_protocol(LOCK_PATH)
    prepared = module.bulk_base.prepare_data(protocol, device)
    splits_v3 = json.loads(SPLITS_V3_PATH.read_text())
    train_cells = [str(c) for c in splits_v3["train_celllines"]]
    holdout_cells = [str(c) for c in splits_v3["test_celllines"]]
    target_cells = [str(c) for c in target_cells]
    if mode == "leave_target_cells_out":
        meta_train_cells = [c for c in train_cells if c not in set(target_cells)]
    elif mode == "all_train_cells":
        meta_train_cells = train_cells
    else:
        raise ValueError(f"Unknown meta mode: {mode}")

    split_frames = {}
    for split_name in ["inner_train", "val", "test"]:
        df, x = split_to_numpy(prepared[split_name])
        df = df.copy()
        df["drug"] = df["drug"].astype(str)
        df["cell_line"] = df["cell_line"].astype(str)
        split_frames[split_name] = (df, x)

    tables: Dict[str, dict] = {}
    for key, source_split, cells, split_name in [
        ("meta_train", "inner_train", meta_train_cells, "meta_train"),
        ("meta_val", "val", meta_train_cells, "meta_val"),
        ("target_support", "inner_train", target_cells, "target_support"),
        ("target_val", "val", target_cells, "target_val"),
        ("target_test", "test", target_cells, "target_test"),
    ]:
        df_split, x_split = split_frames[source_split]
        rows, x_raw = build_drugcell_table(df_split, x_split, labels, cells, split_name)
        tables[key] = {
            "rows": rows,
            "x_raw": x_raw,
            "y": true_target_matrix(rows, labels),
            "weights": row_weights(rows) if len(rows) else np.zeros((0,), dtype=np.float32),
        }
    test_df, test_x = split_frames["test"]
    prof_rows, prof_x_raw = build_profile_table(test_df, test_x, labels, target_cells, "target_test_profile")
    tables["target_test_profile"] = {
        "rows": prof_rows,
        "x_raw": prof_x_raw,
        "y": true_target_matrix(prof_rows, labels),
        "weights": row_weights(prof_rows) if len(prof_rows) else np.zeros((0,), dtype=np.float32),
    }
    std_audit = standardize_tables(tables["meta_train"]["x_raw"], tables)
    for item in tables.values():
        rows = item["rows"]
        item["by_cell"] = {
            cell: np.where(rows["cell_line"].astype(str).eq(cell).to_numpy())[0]
            for cell in sorted(rows["cell_line"].astype(str).unique())
        }
    audit = {
        "mode": mode,
        "protocol_path": str(LOCK_PATH),
        "protocol_sha256": sha256_file(LOCK_PATH),
        "splits_v3_path": str(SPLITS_V3_PATH),
        "splits_v3_sha256": sha256_file(SPLITS_V3_PATH),
        "n_train_cells": len(train_cells),
        "n_holdout_cells": len(holdout_cells),
        "target_cells": target_cells,
        "target_cell_membership": {
            c: "train_cell" if c in train_cells else "holdout_cell" if c in holdout_cells else "missing"
            for c in target_cells
        },
        "meta_train_cell_policy": mode,
        "n_meta_train_cells": len(meta_train_cells),
        "drug_split_counts": {
            "inner_train_drugs": len(protocol["drug_split"]["inner_train_drugs"]),
            "val_drugs": len(protocol["drug_split"]["val_drugs"]),
            "test_drugs": len(protocol["drug_split"]["test_drugs"]),
        },
        "table_counts": {
            name: {
                "n_groups": int(len(item["rows"])),
                "n_cells": int(item["rows"]["cell_line"].nunique()) if len(item["rows"]) else 0,
                "n_drugs": int(item["rows"]["drug"].nunique()) if len(item["rows"]) else 0,
                "n_profiles": int(item["rows"]["n_profiles"].sum()) if len(item["rows"]) else 0,
            }
            for name, item in tables.items()
        },
        "standardization": std_audit,
    }
    return tables, audit


def batch_tensors(item: dict, idx: np.ndarray, device: torch.device):
    return (
        torch.tensor(item["x"][idx], dtype=torch.float32, device=device),
        torch.tensor(item["y"][idx], dtype=torch.float32, device=device),
        torch.tensor(item["weights"][idx], dtype=torch.float32, device=device),
    )


def bag_softmax_loss(logits: torch.Tensor, y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    log_all = torch.logsumexp(logits, dim=1)
    pos_logits = logits.masked_fill(y <= 0, -1.0e9)
    log_pos = torch.logsumexp(pos_logits, dim=1)
    w = weights / torch.clamp(weights.sum(), min=1.0e-8)
    return torch.sum(w * (log_all - log_pos))


def drug_normalized_bce_loss(logits: torch.Tensor, y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    per = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    pos = y > 0
    neg = ~pos
    pos_loss = (per * pos.float()).sum(dim=1) / torch.clamp(pos.float().sum(dim=1), min=1.0)
    neg_loss = (per * neg.float()).sum(dim=1) / torch.clamp(neg.float().sum(dim=1), min=1.0)
    w = weights / torch.clamp(weights.sum(), min=1.0e-8)
    return torch.sum(w * 0.5 * (pos_loss + neg_loss))


def retrieval_loss(model: "FixedTargetReadout", x: torch.Tensor, y: torch.Tensor, weights: torch.Tensor, params: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
    logits = model(x, params=params)
    loss = bag_softmax_loss(logits, y, weights)
    loss = loss + DNBCE_WEIGHT * drug_normalized_bce_loss(logits, y, weights)
    if params is None:
        loss = loss + model.regularization()
    return loss


class FixedTargetReadout(torch.nn.Module):
    def __init__(self, target_init: np.ndarray, dropout: float = 0.35) -> None:
        super().__init__()
        dim = int(target_init.shape[1])
        self.dim = dim
        self.n_targets = int(target_init.shape[0])
        self.dropout = float(dropout)
        self.hidden = max(128, min(192, dim * 3))
        self.register_buffer("target_anchor", F.normalize(torch.tensor(target_init, dtype=torch.float32), dim=1))
        self.query_encoder = torch.nn.Sequential(
            torch.nn.Linear(978, self.hidden),
            torch.nn.LayerNorm(self.hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden, dim),
            torch.nn.LayerNorm(dim),
        )
        self.query_norm = torch.nn.LayerNorm(dim)
        self.diag = torch.nn.Parameter(torch.ones(dim))
        self.score_scale = torch.nn.Parameter(torch.tensor(0.7))
        self.target_bias = torch.nn.Parameter(torch.zeros(self.n_targets))

    def p(self, name: str, params: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        if params is not None and name in params:
            return params[name]
        current = self
        for part in name.split("."):
            if part.isdigit():
                current = current[int(part)]
            else:
                current = getattr(current, part)
        return current

    def forward(self, x: torch.Tensor, params: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        h = F.linear(x, self.p("query_encoder.0.weight", params), self.p("query_encoder.0.bias", params))
        h = F.layer_norm(h, (self.hidden,), self.p("query_encoder.1.weight", params), self.p("query_encoder.1.bias", params))
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        q = F.linear(h, self.p("query_encoder.4.weight", params), self.p("query_encoder.4.bias", params))
        q = F.layer_norm(q, (self.dim,), self.p("query_encoder.5.weight", params), self.p("query_encoder.5.bias", params))
        q = F.normalize(F.layer_norm(q, (self.dim,), self.p("query_norm.weight", params), self.p("query_norm.bias", params)), dim=1)
        score = (q * self.p("diag", params)[None, :]) @ self.target_anchor.T
        score = score * torch.exp(torch.clamp(self.p("score_scale", params), -2.0, 2.0))
        return score + self.p("target_bias", params)[None, :]

    def regularization(self) -> torch.Tensor:
        return TARGET_BIAS_L2 * torch.square(self.target_bias).sum() + SCORER_L2 * torch.square(self.diag - 1.0).sum()


def binary_auroc(y: np.ndarray, score: np.ndarray) -> float:
    y = y.astype(np.int8)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    return float((ranks[y > 0].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(score)[::-1]
    y_sorted = y[order].astype(np.float64)
    n_pos = y_sorted.sum()
    if n_pos <= 0:
        return float("nan")
    precision = np.cumsum(y_sorted) / (np.arange(len(y_sorted)) + 1.0)
    return float((precision * y_sorted).sum() / n_pos)


def ndcg_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float:
    order = np.argsort(score)[::-1][:k]
    gains = y[order].astype(np.float64)
    if gains.sum() <= 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float((gains * discounts).sum())
    ideal = np.sort(y.astype(np.float64))[::-1][:k]
    idcg = float((ideal * discounts[: len(ideal)]).sum())
    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics(scores: np.ndarray, y: np.ndarray, rows: pd.DataFrame, method: str, variant: str, unit: str) -> Tuple[dict, pd.DataFrame]:
    records = []
    for i in range(scores.shape[0]):
        s = np.nan_to_num(scores[i].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        yy = y[i].astype(np.int8)
        order = np.argsort(s)[::-1]
        pos = set(np.where(yy > 0)[0].tolist())
        rec = {
            "method": method,
            "variant": variant,
            "split": str(rows.iloc[i]["split"]),
            "evaluation_unit": unit,
            "drug": str(rows.iloc[i]["drug"]),
            "cell_line": str(rows.iloc[i]["cell_line"]),
            "n_profiles": int(rows.iloc[i]["n_profiles"]),
            "n_true_targets": int(yy.sum()),
        }
        for k in HIT_KS:
            rec[f"Hit@{k}"] = int(bool(set(order[:k]) & pos))
        rec["AUROC"] = binary_auroc(yy, s)
        rec["NDCG@10"] = ndcg_at_k(yy, s, 10)
        rec["MeanAP"] = average_precision(yy, s)
        records.append(rec)
    per = pd.DataFrame(records)
    summary = {
        "method": method,
        "variant": variant,
        "split": str(rows["split"].iloc[0]) if len(rows) else "",
        "evaluation_unit": unit,
        "n_groups": int(len(rows)),
        "n_drugs": int(rows["drug"].nunique()) if len(rows) else 0,
        "n_profiles": int(rows["n_profiles"].sum()) if len(rows) else 0,
    }
    for metric in [f"Hit@{k}" for k in HIT_KS] + ["AUROC", "NDCG@10", "MeanAP"]:
        summary[metric] = float(per[metric].mean()) if len(per) else float("nan")
    return summary, per


def predict(model: FixedTargetReadout, x: np.ndarray, device: torch.device, params: Optional[Dict[str, torch.Tensor]] = None, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            outs.append(model(xb, params=params).detach().cpu().numpy())
    return np.vstack(outs)


def better_metric(candidate: dict, best: Optional[dict]) -> bool:
    if best is None:
        return True
    for key in ["Hit@10", "MeanAP", "Hit@5", "NDCG@10"]:
        diff = float(candidate.get(key, 0.0)) - float(best.get(key, 0.0))
        if abs(diff) > 1.0e-9:
            return diff > 0
    return False


def evaluate_item(model: FixedTargetReadout, item: dict, device: torch.device, method: str, variant: str, unit: str, params: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[dict, pd.DataFrame]:
    scores = predict(model, item["x"], device, params=params)
    return compute_metrics(scores, item["y"], item["rows"], method, variant, unit)


def train_pooled(model: FixedTargetReadout, tables: Dict[str, dict], device: torch.device, args: argparse.Namespace, tag: str) -> Tuple[dict, List[dict]]:
    opt = torch.optim.AdamW(model.parameters(), lr=args.pooled_lr, weight_decay=args.weight_decay)
    train = tables["meta_train"]
    val = tables["meta_val"]
    best_state = None
    best_val = None
    best_epoch = 0
    stale = 0
    history: List[dict] = []
    for epoch in range(1, args.pooled_epochs + 1):
        model.train()
        order = np.random.permutation(len(train["rows"]))
        losses = []
        for start in range(0, len(order), args.batch_size):
            idx = order[start : start + args.batch_size]
            xb, yb, wb = batch_tensors(train, idx, device)
            opt.zero_grad(set_to_none=True)
            loss = retrieval_loss(model, xb, yb, wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        train_summary, _ = evaluate_item(model, train, device, "Round208_A01_fixedtarget", f"{tag}_pooled", "drug-mean")
        val_summary, _ = evaluate_item(model, val, device, "Round208_A01_fixedtarget", f"{tag}_pooled", "drug-mean")
        rec = {
            "stage": "pooled",
            "tag": tag,
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "train_Hit@10": train_summary["Hit@10"],
            "train_MeanAP": train_summary["MeanAP"],
            "val_Hit@10": val_summary["Hit@10"],
            "val_MeanAP": val_summary["MeanAP"],
            "val_NDCG@10": val_summary["NDCG@10"],
        }
        history.append(rec)
        if epoch == 1 or epoch % 5 == 0:
            log(f"{tag} pooled epoch={epoch} val_Hit@10={rec['val_Hit@10']:.4f} val_MeanAP={rec['val_MeanAP']:.4f}")
        if better_metric(val_summary, best_val):
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val = val_summary
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch >= POOLED_MIN_EPOCHS and stale >= POOLED_PATIENCE:
            log(f"{tag} pooled early stop epoch={epoch}; best_epoch={best_epoch}")
            break
    if best_state is None or best_val is None:
        raise RuntimeError(f"No pooled checkpoint selected for {tag}")
    model.load_state_dict(best_state)
    torch.save({"model_state": best_state, "best_epoch": best_epoch, "best_val": best_val}, CHECKPOINTS / f"{tag}_pooled_fixedtarget.pth")
    return best_val, history


def sample_indices(indices: np.ndarray, size: int, replace: bool = False) -> np.ndarray:
    if len(indices) == 0:
        raise RuntimeError("Cannot sample from empty index set")
    if size <= 0:
        return indices.copy()
    if len(indices) >= size:
        return np.random.choice(indices, size=size, replace=False)
    return np.random.choice(indices, size=size, replace=True if replace else True)


def inner_parameter_names(kind: str) -> List[str]:
    base = [
        "query_encoder.4.weight",
        "query_encoder.4.bias",
        "query_encoder.5.weight",
        "query_encoder.5.bias",
        "query_norm.weight",
        "query_norm.bias",
        "diag",
        "score_scale",
    ]
    if kind == "query_bias":
        return base + ["target_bias"]
    if kind == "query_only":
        return base
    raise ValueError(f"Unknown inner kind: {kind}")


def adapt_params(
    model: FixedTargetReadout,
    item: dict,
    idx: np.ndarray,
    device: torch.device,
    inner_names: Sequence[str],
    inner_lr: float,
    inner_steps: int,
) -> Dict[str, torch.Tensor]:
    params = {name: p for name, p in model.named_parameters()}
    for _ in range(inner_steps):
        xb, yb, wb = batch_tensors(item, idx, device)
        loss = retrieval_loss(model, xb, yb, wb, params=params)
        selected = [params[name] for name in inner_names]
        grads = torch.autograd.grad(loss, selected, create_graph=False, allow_unused=False)
        new_params = dict(params)
        for name, grad in zip(inner_names, grads):
            new_params[name] = params[name] - inner_lr * grad
        params = new_params
    return params


def eval_after_adaptation(
    model: FixedTargetReadout,
    support_item: dict,
    query_item: dict,
    device: torch.device,
    variant: str,
    support_size: int,
    inner_names: Sequence[str],
    args: argparse.Namespace,
    unit: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    per_rows = []
    for cell, qidx in query_item["by_cell"].items():
        sidx_all = support_item["by_cell"].get(cell, np.asarray([], dtype=np.int64))
        if len(sidx_all) == 0 or len(qidx) == 0:
            continue
        if support_size == 0:
            params = None
        else:
            sidx = sample_indices(sidx_all, support_size)
            params = adapt_params(model, support_item, sidx, device, inner_names, args.inner_lr, args.inner_steps)
        sub = {
            "rows": query_item["rows"].iloc[qidx].reset_index(drop=True),
            "x": query_item["x"][qidx],
            "y": query_item["y"][qidx],
        }
        summary, per = evaluate_item(
            model,
            sub,
            device,
            "Round208_A01_fixedtarget_MAML",
            f"{variant}_adapt{support_size}",
            unit,
            params=params,
        )
        summary["cell_line"] = cell
        per["adapt_support_size"] = support_size
        summaries.append(summary)
        per_rows.append(per)
    return pd.DataFrame(summaries), pd.concat(per_rows, ignore_index=True) if per_rows else pd.DataFrame()


def meta_val_macro(
    model: FixedTargetReadout,
    tables: Dict[str, dict],
    device: torch.device,
    variant: str,
    inner_names: Sequence[str],
    args: argparse.Namespace,
) -> dict:
    summary, _ = eval_after_adaptation(
        model,
        tables["target_support"],
        tables["target_val"],
        device,
        variant,
        args.eval_support_size,
        inner_names,
        args,
        "drug-mean",
    )
    if summary.empty:
        return {"Hit@10": float("nan"), "MeanAP": float("nan"), "Hit@5": float("nan"), "NDCG@10": float("nan")}
    out = {"split": "target_val_macro", "n_cells": int(summary["cell_line"].nunique())}
    for metric in ["Hit@5", "Hit@10", "NDCG@10", "MeanAP"]:
        out[metric] = float(summary[metric].mean())
    return out


def train_maml(
    model: FixedTargetReadout,
    tables: Dict[str, dict],
    device: torch.device,
    variant: str,
    inner_kind: str,
    args: argparse.Namespace,
) -> Tuple[dict, List[dict]]:
    inner_names = inner_parameter_names(inner_kind)
    opt = torch.optim.AdamW(model.parameters(), lr=args.meta_lr, weight_decay=args.weight_decay)
    train = tables["meta_train"]
    cells = [c for c, idx in train["by_cell"].items() if len(idx) >= args.support_size + 2]
    if not cells:
        raise RuntimeError("No eligible cells for MAML training")
    best_state = None
    best_val = None
    best_epoch = 0
    stale = 0
    history: List[dict] = []
    for epoch in range(1, args.meta_epochs + 1):
        model.train()
        losses = []
        for _ in range(args.tasks_per_epoch):
            cell = random.choice(cells)
            all_idx = np.asarray(train["by_cell"][cell], dtype=np.int64)
            perm = np.random.permutation(all_idx)
            support = perm[: min(args.support_size, len(perm))]
            remaining = perm[min(args.support_size, len(perm)) :]
            if len(remaining) == 0:
                remaining = perm
            query = sample_indices(remaining, args.query_size)
            adapted = adapt_params(model, train, support, device, inner_names, args.inner_lr, args.inner_steps)
            xb, yb, wb = batch_tensors(train, query, device)
            loss = retrieval_loss(model, xb, yb, wb, params=adapted)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_summary = meta_val_macro(model, tables, device, variant, inner_names, args)
        rec = {
            "stage": "maml",
            "variant": variant,
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_Hit@10": val_summary["Hit@10"],
            "val_MeanAP": val_summary["MeanAP"],
            "val_NDCG@10": val_summary["NDCG@10"],
        }
        history.append(rec)
        if epoch == 1 or epoch % 5 == 0:
            log(f"{variant} epoch={epoch} val_Hit@10={rec['val_Hit@10']:.4f} val_MeanAP={rec['val_MeanAP']:.4f}")
        if better_metric(val_summary, best_val):
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val = val_summary
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch >= args.min_meta_epochs and stale >= args.meta_patience:
            log(f"{variant} early stop epoch={epoch}; best_epoch={best_epoch}")
            break
    if best_state is None or best_val is None:
        raise RuntimeError(f"No MAML checkpoint selected for {variant}")
    model.load_state_dict(best_state)
    torch.save(
        {
            "model_state": best_state,
            "best_epoch": best_epoch,
            "best_val": best_val,
            "inner_parameter_names": inner_names,
            "target_anchor_trainable": False,
        },
        CHECKPOINTS / f"{variant}.pth",
    )
    return best_val, history


def evaluate_variant(
    model: FixedTargetReadout,
    tables: Dict[str, dict],
    device: torch.device,
    variant: str,
    inner_kind: str,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    inner_names = inner_parameter_names(inner_kind)
    summaries = []
    per = []
    for support_size in args.report_support_sizes:
        summary, per_rows = eval_after_adaptation(
            model,
            tables["target_support"],
            tables["target_test"],
            device,
            variant,
            int(support_size),
            inner_names,
            args,
            "drug-mean",
        )
        summaries.append(summary)
        per.append(per_rows)
        profile_summary, profile_per = eval_after_adaptation(
            model,
            tables["target_support"],
            tables["target_test_profile"],
            device,
            variant,
            int(support_size),
            inner_names,
            args,
            "true-profile-specific",
        )
        summaries.append(profile_summary)
        per.append(profile_per)
    return pd.concat(summaries, ignore_index=True), pd.concat(per, ignore_index=True)


def write_report(summary_df: pd.DataFrame, audit: dict) -> None:
    test = summary_df[summary_df["split"].isin(["target_test", "target_test_profile"])].copy()
    metric_cols = ["Hit@1", "Hit@5", "Hit@10", "Hit@20", "Hit@30", "Hit@50", "AUROC", "NDCG@10", "MeanAP"]
    report_table = (
        test[["cell_line", "variant", "evaluation_unit", "n_groups", "n_drugs", "n_profiles"] + metric_cols]
        .sort_values(["evaluation_unit", "cell_line", "variant"])
        .to_csv(sep="\t", index=False)
        .strip()
    )
    lines = [
        "# Round208 A01 fixed-target MAML readout",
        "",
        "## Key guardrails",
        "",
        "- A01 pretrained target embeddings are registered as a fixed buffer.",
        "- No `target_delta` parameter exists in this model.",
        "- Inner-loop and outer-loop updates only affect query/readout parameters.",
        "- PC3, A375 and MCF7 are train cells under the legacy `splits_v3` cell split; this run evaluates drug-level adaptation on those benchmark cell lines, not held-out-cell extrapolation.",
        "",
        "## Test metrics",
        "",
        "```tsv",
        report_table,
        "```",
        "",
        "## Audit",
        "",
        "```json",
        json.dumps(audit, indent=2, ensure_ascii=False, default=str),
        "```",
    ]
    (ROOT / "round208_result_report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_mode(mode: str, target_init: np.ndarray, labels: Sequence[str], device: torch.device, args: argparse.Namespace, base_audit: dict):
    log(f"preparing data mode={mode}")
    tables, data_audit = prepare_data(labels, args.target_cells, mode, device)
    target = torch.tensor(target_init, dtype=torch.float32)
    summaries = []
    per_rows = []
    histories = []

    pooled_model = FixedTargetReadout(target_init, dropout=args.dropout).to(device)
    pooled_val, pooled_history = train_pooled(pooled_model, tables, device, args, mode)
    histories.extend(pooled_history)
    pooled_summary, pooled_per = evaluate_variant(pooled_model, tables, device, f"{mode}_pooled_fixedtarget", "query_only", args)
    summaries.append(pooled_summary)
    per_rows.append(pooled_per)

    for inner_kind in ["query_only", "query_bias"]:
        variant = f"{mode}_fo_maml_{inner_kind}_fixedtarget"
        model = FixedTargetReadout(target_init, dropout=args.dropout).to(device)
        model.load_state_dict(pooled_model.state_dict())
        _, hist = train_maml(model, tables, device, variant, inner_kind, args)
        histories.extend(hist)
        summary, per = evaluate_variant(model, tables, device, variant, inner_kind, args)
        summaries.append(summary)
        per_rows.append(per)

    audit = {
        "base": base_audit,
        "data": data_audit,
        "mode": mode,
        "target_anchor_requires_grad": bool(pooled_model.target_anchor.requires_grad),
        "model_has_target_delta": bool(any("target_delta" in name for name, _ in pooled_model.named_parameters())),
        "trainable_parameter_names": [name for name, _ in pooled_model.named_parameters()],
    }
    return pd.concat(summaries, ignore_index=True), pd.concat(per_rows, ignore_index=True), pd.DataFrame(histories), audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--target-cells", nargs="+", default=TARGET_CELLS)
    parser.add_argument("--modes", nargs="+", default=["all_train_cells", "leave_target_cells_out"])
    parser.add_argument("--pooled-epochs", type=int, default=24)
    parser.add_argument("--meta-epochs", type=int, default=24)
    parser.add_argument("--min-meta-epochs", type=int, default=8)
    parser.add_argument("--meta-patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=READOUT_BATCH_SIZE)
    parser.add_argument("--support-size", type=int, default=64)
    parser.add_argument("--query-size", type=int, default=192)
    parser.add_argument("--eval-support-size", type=int, default=64)
    parser.add_argument("--report-support-sizes", nargs="+", type=int, default=[0, 8, 16, 64])
    parser.add_argument("--tasks-per-epoch", type=int, default=48)
    parser.add_argument("--inner-steps", type=int, default=1)
    parser.add_argument("--inner-lr", type=float, default=8.0e-3)
    parser.add_argument("--meta-lr", type=float, default=2.5e-4)
    parser.add_argument("--pooled-lr", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=3.0e-4)
    parser.add_argument("--dropout", type=float, default=0.35)
    args = parser.parse_args()

    ensure_dirs()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log(f"using device={device}")
    labels, target_init, a01_audit = labels_and_targets_from_a01(device)
    all_summaries = []
    all_per = []
    all_hist = []
    audits = {"config": vars(args), "a01": a01_audit}
    for mode in args.modes:
        summary, per, hist, audit = run_mode(mode, target_init, labels, device, args, a01_audit)
        all_summaries.append(summary)
        all_per.append(per)
        all_hist.append(hist)
        audits[mode] = audit

    summary_df = pd.concat(all_summaries, ignore_index=True)
    per_df = pd.concat(all_per, ignore_index=True)
    hist_df = pd.concat(all_hist, ignore_index=True)
    summary_df.to_csv(SOURCE / "metric_summary.tsv", sep="\t", index=False)
    per_df.to_csv(SOURCE / "per_unit_metrics.tsv", sep="\t", index=False)
    hist_df.to_csv(SOURCE / "training_history.tsv", sep="\t", index=False)
    (ROOT / "method_lock.json").write_text(json.dumps(audits, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (QA / "fixedtarget_maml_audit.md").write_text(
        "\n".join(
            [
                "# Round208 fixed-target MAML audit",
                "",
                "- A01 target embeddings are loaded from the Round205 A01 epoch-35 checkpoint.",
                "- The readout class registers `target_anchor` as a buffer and defines no `target_delta` parameter.",
                "- MAML inner-loop parameter sets are query/readout-only (`query_only`) or query/readout plus free target bias (`query_bias`).",
                "- Test-drug labels are used only for final metric calculation.",
                "- PC3/A375/MCF7 are train cells under `splits_v3`; `leave_target_cells_out` excludes them from meta-training as a pseudo-heldout-cell sensitivity.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_report(summary_df, audits)
    log(f"done; outputs written to {ROOT}")


if __name__ == "__main__":
    main()
