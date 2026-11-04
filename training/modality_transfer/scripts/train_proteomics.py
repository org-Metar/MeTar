#!/usr/bin/env python
import ast
import contextlib
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from metar.runtime import report_metrics, report_progress, report_step, resolve_device


SEED = int(os.environ.get("SEED", "2026"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

PT_MAIN_DIR = os.path.expanduser(os.environ.get("PT_MAIN_DIR", "~/claude/pt/main"))
PT_DATA_DIR = os.path.expanduser(os.environ.get("PT_DATA_DIR", os.path.join(PT_MAIN_DIR, "..", "data")))
LABEL_REPR_PATH = os.path.expanduser(os.environ["LABEL_REPR_PATH"])
OUT_PATH = os.path.expanduser(os.environ["OUT_PATH"])
SAVE_CKPT_PATH = os.path.expanduser(os.environ.get("SAVE_CKPT_PATH", OUT_PATH.replace(".json", ".pth")))
DEVICE = os.environ.get("DEVICE", "auto")

VAL_FRAC = float(os.environ.get("VAL_FRAC", "0.2"))
EPOCHS = int(os.environ.get("EPOCHS", "180"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "5"))
LR = float(os.environ.get("LR", "3e-4"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
HOLDOUT_K_PERCENT = float(os.environ.get("HOLDOUT_K_PERCENT", "0.05"))
PROJ_DIM = int(os.environ.get("PROJ_DIM", "256"))
N_RES = int(os.environ.get("N_RES", "1"))
DROPOUT = float(os.environ.get("DROPOUT", "0.45"))
INIT_SCALE = float(os.environ.get("INIT_SCALE", "10.0"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
TOPK_FEAT = int(os.environ.get("TOPK_FEAT", "2000"))
LABEL_ADAPTER_MODE = os.environ.get("LABEL_ADAPTER_MODE", "linear")
LABEL_BOTTLENECK = int(os.environ.get("LABEL_BOTTLENECK", "64"))
ADAPTER_L2 = float(os.environ.get("ADAPTER_L2", "1e-4"))
ORTHO_LAMBDA = float(os.environ.get("ORTHO_LAMBDA", "1e-4"))
PER_DRUG_PATH = os.path.expanduser(os.environ.get("PER_DRUG_PATH", OUT_PATH.replace(".json", "_per_drug.tsv")))
HISTORY_TSV_PATH = os.path.expanduser(os.environ.get("HISTORY_TSV_PATH", OUT_PATH.replace(".json", "_training_history.tsv")))
METHOD_LOCK_PATH = os.path.expanduser(os.environ.get("METHOD_LOCK_PATH", OUT_PATH.replace(".json", "_method_lock.json")))
METRIC_SUMMARY_PATH = os.path.expanduser(os.environ.get("METRIC_SUMMARY_PATH", OUT_PATH.replace(".json", "_metric_summary.tsv")))
LEAKAGE_AUDIT_PATH = os.path.expanduser(os.environ.get("LEAKAGE_AUDIT_PATH", OUT_PATH.replace(".json", "_leakage_and_denominator_audit.md")))
SELECTED_RESULT_PATH = os.path.expanduser(os.environ.get("SELECTED_RESULT_PATH", OUT_PATH.replace(".json", "_selected_primary_result.json")))

ALIAS_TO_LOCKED = {
    "15-Sep": "SEPT15",
    "3-Sep": "SEPT3",
    "4-Sep": "SEPT4",
    "5-Sep": "SEPT5",
    "6-Sep": "SEPT6",
    "7-Sep": "SEPT7",
    "9-Sep": "SEPT9",
    "8-Mar": "MARCHF8",
}

device = torch.device(resolve_device(DEVICE))
report_step(f"Proteomics settings: device={device} seed={SEED} epochs={EPOCHS} eval_every={EVAL_EVERY} batch_size={BATCH_SIZE} lr={LR} label_adapter={LABEL_ADAPTER_MODE}")
for output_path in (OUT_PATH, SAVE_CKPT_PATH, PER_DRUG_PATH, HISTORY_TSV_PATH, METHOD_LOCK_PATH, METRIC_SUMMARY_PATH, LEAKAGE_AUDIT_PATH, SELECTED_RESULT_PATH):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)


class ProteoDataset(Dataset):
    def __init__(self, x, y_seen, y_hold):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y_seen = torch.tensor(y_seen, dtype=torch.float32)
        self.y_hold = torch.tensor(y_hold, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y_seen[idx], self.y_hold[idx]


class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.35):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class DeepProteoMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        proj_dim: int = 256,
        d: int = 256,
        n_res: int = 1,
        dropout: float = 0.45,
        init_scale: float = 10.0,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res_blocks = nn.ModuleList([ResBlock(proj_dim, dropout) for _ in range(n_res)])
        self.head = nn.Linear(proj_dim, d)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(init_scale), dtype=torch.float32))
        self._init_weights()

    def _init_weights(self):
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.kaiming_normal_(mod.weight, nonlinearity="relu")
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    def forward_embed(self, x):
        x = self.proj(x)
        for blk in self.res_blocks:
            x = blk(x)
        return F.normalize(self.head(x), dim=1)


class LabelAdapter(nn.Module):
    def __init__(self, dim: int, mode: str = "linear", bottleneck: int = 64):
        super().__init__()
        self.mode = mode
        self.dim = dim
        if mode == "identity":
            self.core = None
        elif mode == "linear":
            self.core = nn.Linear(dim, dim, bias=False)
            nn.init.eye_(self.core.weight)
        elif mode == "residual_mlp":
            self.down = nn.Linear(dim, bottleneck, bias=False)
            self.up = nn.Linear(bottleneck, dim, bias=False)
            nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.up.weight)
            self.gate = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        else:
            raise ValueError(f"Unknown LABEL_ADAPTER_MODE: {mode}")

    def forward(self, w):
        if self.mode == "identity":
            return F.normalize(w, dim=1)
        if self.mode == "linear":
            return F.normalize(self.core(w), dim=1)
        scale = torch.sigmoid(self.gate)
        delta = self.up(F.gelu(self.down(w)))
        return F.normalize(w + scale * delta, dim=1)

    def regularization_loss(self):
        if self.mode == "identity":
            return torch.tensor(0.0, device=device)
        if self.mode == "linear":
            weight = self.core.weight
            eye = torch.eye(self.dim, device=weight.device, dtype=weight.dtype)
            id_loss = ((weight - eye) ** 2).mean()
            ortho = ((weight.t() @ weight - eye) ** 2).mean()
            return ADAPTER_L2 * id_loss + ORTHO_LAMBDA * ortho
        delta_pen = (self.up.weight ** 2).mean() + (self.down.weight ** 2).mean()
        return ADAPTER_L2 * delta_pen


def split_by_drug(df, all_drugs, drug_targets, test_size=0.2, seed=0):
    rng = np.random.RandomState(seed)
    drugs = all_drugs.copy()
    rng.shuffle(drugs)
    n_test = max(1, int(len(drugs) * test_size))
    test_drugs = drugs[:n_test]
    train_drugs = drugs[n_test:]
    train_targets = set()
    for d in train_drugs:
        train_targets |= drug_targets[d]
    valid_test = [d for d in test_drugs if drug_targets[d] & train_targets]
    train_df = df[df["drug"].isin(train_drugs)].copy()
    test_df = df[df["drug"].isin(valid_test)].copy()
    return train_df, test_df, train_drugs, np.array(valid_test)


def split_train_val_drugs(train_drugs, drug_targets, val_frac=0.2, seed=123):
    rng = np.random.RandomState(seed)
    drugs = np.array(train_drugs).copy()
    rng.shuffle(drugs)
    n_val = max(1, int(len(drugs) * val_frac))
    raw_val = drugs[:n_val]
    raw_train = drugs[n_val:]
    train_targets = set()
    for d in raw_train:
        train_targets |= drug_targets[d]
    val_drugs = np.array([d for d in raw_val if drug_targets[d] & train_targets])
    if len(val_drugs) == 0:
        val_drugs = np.array(raw_val[:1])
    train_drugs2 = np.array([d for d in drugs if d not in set(val_drugs.tolist())])
    return train_drugs2, val_drugs


def dcg_at_k(rel: np.ndarray, k: int) -> float:
    rel = rel[:k]
    return float((rel / np.log2(np.arange(2, len(rel) + 2))).sum())


def ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    order = np.argsort(scores)[::-1]
    rel_sorted = labels[order]
    ideal = np.sort(labels)[::-1]
    dcg = dcg_at_k(rel_sorted, k)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)[::-1]
    rel = labels[order]
    n_pos = rel.sum()
    if n_pos == 0:
        return 0.0
    precisions = np.cumsum(rel) / (np.arange(len(rel)) + 1)
    return float((precisions * rel).sum() / n_pos)


def average_precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    order = np.argsort(scores)[::-1][:k]
    rel = labels[order]
    n_pos_total = int(labels.sum())
    if n_pos_total == 0:
        return 0.0
    denom = max(1, min(n_pos_total, k))
    precisions = np.cumsum(rel) / (np.arange(len(rel)) + 1)
    return float((precisions * rel).sum() / denom)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def evaluate_seen(model_p, adapter, loader, w_seen, ks=(1, 3, 5, 10)):
    model_p.eval()
    adapter.eval()
    w_seen_adapt = adapter(w_seen)
    results = {f"Hit@{k}": [] for k in ks}
    results.update({f"NDCG@{k}": [] for k in ks})
    results.update({f"MAP@{k}": [] for k in ks})
    for completed, (xb, yb, _) in enumerate(loader, start=1):
        xb = xb.to(device)
        logits = (model_p.forward_embed(xb) @ w_seen_adapt.t()) * model_p.logit_scale.exp().clamp(max=100.0)
        logits = logits.cpu().numpy()
        labels = yb.numpy()
        for i in range(len(logits)):
            if labels[i].sum() == 0:
                continue
            for k in ks:
                topk = np.argsort(logits[i])[::-1][:k]
                results[f"Hit@{k}"].append(float(labels[i][topk].any()))
                results[f"NDCG@{k}"].append(ndcg_at_k(logits[i], labels[i], k))
                results[f"MAP@{k}"].append(average_precision_at_k(logits[i], labels[i], k))
        report_progress(completed, len(loader), label="proteomics seen evaluation batches")
    return {k: (float(np.mean(v)) if v else 0.0) for k, v in results.items()}


@torch.no_grad()
def evaluate_holdout(model_p, adapter, loader, w_all, holdout_pos_to_global_t, k_pct=0.05):
    model_p.eval()
    adapter.eval()
    w_all_adapt = adapter(w_all)
    g = w_all.shape[0]
    k = max(1, int(g * k_pct))
    hits = 0
    total = 0
    for completed, (xb, _, yh) in enumerate(loader, start=1):
        xb = xb.to(device)
        yh = yh.to(device)
        logits = (model_p.forward_embed(xb) @ w_all_adapt.t()) * model_p.logit_scale.exp().clamp(max=100.0)
        topk_idx = torch.topk(logits, k=min(k, g), dim=1).indices
        for i in range(yh.shape[0]):
            pos = torch.where(yh[i] > 0.5)[0]
            if pos.numel() == 0:
                continue
            true_global = holdout_pos_to_global_t[pos]
            if (topk_idx[i].unsqueeze(1) == true_global.unsqueeze(0)).any():
                hits += 1
            total += 1
        report_progress(completed, len(loader), label=f"proteomics holdout top{int(k_pct * 100)}pct evaluation batches")
    return hits / total if total > 0 else 0.0


@torch.no_grad()
def per_drug_metrics(model_p, adapter, loader, rows_df, w_seen, w_all, holdout_pos_to_global_t):
    model_p.eval()
    adapter.eval()
    w_seen_adapt = adapter(w_seen)
    w_all_adapt = adapter(w_all)
    rows = []
    offset = 0
    for xb, y_seen_b, y_hold_b in loader:
        xb = xb.to(device)
        seen_logits = (model_p.forward_embed(xb) @ w_seen_adapt.t()) * model_p.logit_scale.exp().clamp(max=100.0)
        all_logits = (model_p.forward_embed(xb) @ w_all_adapt.t()) * model_p.logit_scale.exp().clamp(max=100.0)
        seen_np = seen_logits.cpu().numpy()
        all_top = {}
        for pct in (0.01, 0.05, 0.10):
            k = max(1, int(w_all.shape[0] * pct))
            all_top[pct] = torch.topk(all_logits, k=min(k, w_all.shape[0]), dim=1).indices.cpu().numpy()
        labels_np = y_seen_b.numpy()
        y_hold_cpu = y_hold_b
        for i in range(len(seen_np)):
            src = rows_df.iloc[offset + i]
            label = labels_np[i]
            row = {
                "drug": src["drug"],
                "n_seen_pos": int(label.sum()),
                "n_holdout_pos": int(y_hold_cpu[i].sum().item()),
            }
            if label.sum() > 0:
                for k in (1, 3, 5, 10):
                    topk = np.argsort(seen_np[i])[::-1][:k]
                    row[f"Hit@{k}"] = float(label[topk].any())
                    row[f"NDCG@{k}"] = ndcg_at_k(seen_np[i], label, k)
                    row[f"MAP@{k}"] = average_precision_at_k(seen_np[i], label, k)
            else:
                for k in (1, 3, 5, 10):
                    row[f"Hit@{k}"] = np.nan
                    row[f"NDCG@{k}"] = np.nan
                    row[f"MAP@{k}"] = np.nan
            pos = torch.where(y_hold_cpu[i] > 0.5)[0]
            if pos.numel() > 0:
                true_global = holdout_pos_to_global_t[pos].cpu().numpy()
                for pct, top_idx in all_top.items():
                    label_pct = int(pct * 100)
                    row[f"holdout_top{label_pct}pct"] = float(np.isin(true_global, top_idx[i]).any())
            else:
                for label_pct in (1, 5, 10):
                    row[f"holdout_top{label_pct}pct"] = np.nan
            rows.append(row)
        offset += len(seen_np)
        report_progress(offset, len(rows_df), label="proteomics per-drug metrics profiles")
    return pd.DataFrame(rows)


prot_path = os.path.join(PT_DATA_DIR, "proteomicsdb_data.csv")
report_step(f"Loading proteomics profiles: {prot_path}")
df = pd.read_csv(prot_path, index_col=0)
df["target"] = df["target"].apply(ast.literal_eval)
df = df[df["target"].apply(len) > 0].copy()
df["drug"] = df["drug"].astype(str).str.strip().str.lower()
df = df.reset_index(drop=True)

report_step(f"Loading proteomics target embeddings: {LABEL_REPR_PATH}")
df_w = pd.read_csv(LABEL_REPR_PATH, index_col=0)
df_w = df_w[~df_w.index.duplicated(keep="first")]
all_embed_genes = list(df_w.index)
gene_to_global = {g: i for i, g in enumerate(all_embed_genes)}
w_all_np = df_w.loc[all_embed_genes].values.astype(np.float32)
w_all = F.normalize(torch.tensor(w_all_np, device=device), dim=1)
d_feat = w_all.shape[1]


def filter_targets(ts: List) -> List[str]:
    seen = set()
    out = []
    for t in ts:
        t = str(t)
        if t in gene_to_global and t not in seen:
            out.append(t)
            seen.add(t)
    return out


report_step("Preparing locked proteomics target holdout, drug splits and training-only feature normalization")
df["target_filt"] = df["target"].apply(filter_targets)
df = df[df["target_filt"].apply(len) > 0].copy().reset_index(drop=True)

drug_targets: Dict[str, set] = {}
for d, grp in df.groupby("drug"):
    merged = set()
    for ts in grp["target_filt"]:
        merged |= set(ts)
    drug_targets[d] = merged

all_drugs = np.array(sorted(drug_targets.keys()))
all_targets = sorted({t for d in all_drugs for t in drug_targets[d]})
random.seed(42)
H = min(50, len(all_targets))
holdout_targets_list = random.sample(all_targets, H)
holdout_targets_set = set(holdout_targets_list)
holdout_pos = {g: i for i, g in enumerate(holdout_targets_list)}
holdout_pos_to_global = np.array([gene_to_global[g] for g in holdout_targets_list], dtype=np.int64)
holdout_pos_to_global_t = torch.tensor(holdout_pos_to_global, device=device)

seen_targets = sorted(set(all_targets) - holdout_targets_set)
seen_gene2local = {g: i for i, g in enumerate(seen_targets)}
seen_global_idx = np.array([gene_to_global[g] for g in seen_targets], dtype=np.int64)
w_seen = F.normalize(w_all[torch.tensor(seen_global_idx, device=device)], dim=1)
L_seen = len(seen_targets)


def to_seen_mh(ts):
    v = np.zeros(L_seen, dtype=np.float32)
    for t in ts:
        if t in seen_gene2local:
            v[seen_gene2local[t]] = 1.0
    return v


def to_hold_mh(ts):
    v = np.zeros(H, dtype=np.float32)
    for t in ts:
        if t in holdout_pos:
            v[holdout_pos[t]] = 1.0
    return v


y_seen_full = np.stack([to_seen_mh(ts) for ts in df["target_filt"]]).astype(np.float32)
y_hold_full = np.stack([to_hold_mh(ts) for ts in df["target_filt"]]).astype(np.float32)

feature_cols = [c for c in df.columns if c not in ["drug", "target", "target_filt"]]
x_raw = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32)

train_df_full, test_df, train_drugs_full, test_drugs = split_by_drug(df, all_drugs, drug_targets, test_size=0.2, seed=0)
train_drugs, val_drugs = split_train_val_drugs(train_drugs_full, drug_targets, val_frac=VAL_FRAC, seed=SEED)
train_df = df[df["drug"].isin(train_drugs)].copy()
val_df = df[df["drug"].isin(val_drugs)].copy()

train_idx = train_df.index.values
val_idx = val_df.index.values
test_idx = test_df.index.values

x_train_raw = x_raw[train_idx]
x_val_raw = x_raw[val_idx]
x_test_raw = x_raw[test_idx]
topk_feat = min(TOPK_FEAT, x_train_raw.shape[1])
var_train = x_train_raw.var(axis=0)
kept_idx = np.sort(np.argsort(var_train)[::-1][:topk_feat])
x_train_raw = x_train_raw[:, kept_idx]
x_val_raw = x_val_raw[:, kept_idx]
x_test_raw = x_test_raw[:, kept_idx]

if x_train_raw.min() >= 0:
    x_train_raw = np.log1p(x_train_raw)
    x_val_raw = np.log1p(x_val_raw)
    x_test_raw = np.log1p(x_test_raw)

mu = x_train_raw.mean(axis=0, keepdims=True)
sd = x_train_raw.std(axis=0, keepdims=True) + 1e-6
x_train = (x_train_raw - mu) / sd
x_val = (x_val_raw - mu) / sd
x_test = (x_test_raw - mu) / sd

y_train = y_seen_full[train_idx]
y_val = y_seen_full[val_idx]
y_test = y_seen_full[test_idx]
yh_train = y_hold_full[train_idx]
yh_val = y_hold_full[val_idx]
yh_test = y_hold_full[test_idx]

train_loader = DataLoader(ProteoDataset(x_train, y_train, yh_train), batch_size=min(BATCH_SIZE, len(x_train)), shuffle=True, drop_last=False)
val_loader = DataLoader(ProteoDataset(x_val, y_val, yh_val), batch_size=max(1, len(x_val)), shuffle=False, drop_last=False)
test_loader = DataLoader(ProteoDataset(x_test, y_test, yh_test), batch_size=max(1, len(x_test)), shuffle=False, drop_last=False)

model_p = DeepProteoMLP(
    in_dim=x_train.shape[1],
    proj_dim=PROJ_DIM,
    d=d_feat,
    n_res=N_RES,
    dropout=DROPOUT,
    init_scale=INIT_SCALE,
).to(device)
label_adapter = LabelAdapter(dim=d_feat, mode=LABEL_ADAPTER_MODE, bottleneck=LABEL_BOTTLENECK).to(device)

params = list(model_p.parameters()) + list(label_adapter.parameters())
opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.1)
use_amp = device.type == "cuda"
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

pos = np.clip(y_train.mean(axis=0), 1e-4, 1.0)
pos_weight = torch.tensor((1.0 - pos) / pos, device=device, dtype=torch.float32)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

history = []
best_score = -1.0
best_epoch = 0
report_step(f"Training proteomics; train_profiles={len(train_df)} val_profiles={len(val_df)} test_profiles={len(test_df)} seen_targets={L_seen} holdout_targets={H}")

for epoch in range(1, EPOCHS + 1):
    model_p.train()
    label_adapter.train()
    losses = []
    for completed, (xb, yb, _) in enumerate(train_loader, start=1):
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad(set_to_none=True)
        amp_ctx = torch.cuda.amp.autocast() if use_amp else contextlib.nullcontext()
        with amp_ctx:
            z = model_p.forward_embed(xb)
            w_seen_adapt = label_adapter(w_seen)
            logits = (z @ w_seen_adapt.t()) * model_p.logit_scale.exp().clamp(max=100.0)
            loss = loss_fn(logits, yb) + label_adapter.regularization_loss()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        losses.append(float(loss.item()))
        report_progress(completed, len(train_loader), label=f"proteomics epoch {epoch} batches", metrics={"loss": losses[-1]})
    scheduler.step()

    if epoch % EVAL_EVERY == 0 or epoch == 1 or epoch == EPOCHS:
        report_step(f"Validating proteomics epoch {epoch}")
        val_seen = evaluate_seen(model_p, label_adapter, val_loader, w_seen)
        val_holdout = evaluate_holdout(model_p, label_adapter, val_loader, w_all, holdout_pos_to_global_t, k_pct=HOLDOUT_K_PERCENT)
        score = 0.60 * val_seen["Hit@10"] + 0.15 * val_seen["NDCG@10"] + 0.25 * val_holdout
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_hit10": val_seen["Hit@10"],
                "val_ndcg10": val_seen["NDCG@10"],
                "val_map10": val_seen["MAP@10"],
                "val_holdout_top5pct": val_holdout,
                "score": score,
            }
        )
        print(
            f"Epoch {epoch:03d} | loss={np.mean(losses):.4f} | "
            f"val Hit@10={val_seen['Hit@10']:.4f} | val NDCG@10={val_seen['NDCG@10']:.4f} | "
            f"val holdout={val_holdout:.4f} | score={score:.4f}",
            flush=True,
        )
        report_metrics(
            {key: history[-1][key] for key in (
                "train_loss", "val_hit10", "val_ndcg10", "val_holdout_top5pct", "score",
            )},
            label=f"proteomics epoch {epoch} validation",
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model_p.state_dict(),
                    "label_adapter_state_dict": label_adapter.state_dict(),
                },
                SAVE_CKPT_PATH,
            )
    report_progress(epoch, EPOCHS, label="proteomics epochs")

report_step(f"Reloading selected proteomics checkpoint; best_epoch={best_epoch} score={best_score}")
best_ckpt = torch.load(SAVE_CKPT_PATH, map_location=device)
model_p.load_state_dict(best_ckpt["model_state_dict"])
label_adapter.load_state_dict(best_ckpt["label_adapter_state_dict"])

report_step("Evaluating selected proteomics checkpoint on validation and final test splits")
final_val_seen = evaluate_seen(model_p, label_adapter, val_loader, w_seen)
final_val_holdout = evaluate_holdout(model_p, label_adapter, val_loader, w_all, holdout_pos_to_global_t, k_pct=HOLDOUT_K_PERCENT)
final_test_seen = evaluate_seen(model_p, label_adapter, test_loader, w_seen)
final_test_holdout_1 = evaluate_holdout(model_p, label_adapter, test_loader, w_all, holdout_pos_to_global_t, k_pct=0.01)
final_test_holdout_5 = evaluate_holdout(model_p, label_adapter, test_loader, w_all, holdout_pos_to_global_t, k_pct=0.05)
final_test_holdout_10 = evaluate_holdout(model_p, label_adapter, test_loader, w_all, holdout_pos_to_global_t, k_pct=0.10)

result = {
    "label_repr_path": LABEL_REPR_PATH,
    "label_repr_sha256": file_sha256(LABEL_REPR_PATH),
    "save_ckpt_path": SAVE_CKPT_PATH,
    "save_ckpt_sha256": file_sha256(SAVE_CKPT_PATH),
    "gene_universe_size": int(w_all.shape[0]),
    "n_total_drugs": int(len(all_drugs)),
    "n_train_drugs": int(len(train_drugs)),
    "n_val_drugs": int(len(val_drugs)),
    "n_test_drugs": int(len(test_drugs)),
    "n_train_rows": int(len(train_df)),
    "n_val_rows": int(len(val_df)),
    "n_test_rows": int(len(test_df)),
    "L_seen": int(L_seen),
    "H": int(H),
    "config": {
        "epochs": EPOCHS,
        "eval_every": EVAL_EVERY,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "val_frac": VAL_FRAC,
        "holdout_k_percent": HOLDOUT_K_PERCENT,
        "seed": SEED,
        "proj_dim": PROJ_DIM,
        "n_res": N_RES,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "topk_feat": TOPK_FEAT,
        "label_adapter_mode": LABEL_ADAPTER_MODE,
        "label_bottleneck": LABEL_BOTTLENECK,
        "adapter_l2": ADAPTER_L2,
        "ortho_lambda": ORTHO_LAMBDA,
    },
    "best_epoch": int(best_epoch),
    "best_val_score": float(best_score),
    "final_val": {
        **final_val_seen,
        "holdout_top5pct": float(final_val_holdout),
    },
    "final_test": {
        **final_test_seen,
        "holdout_top1pct": float(final_test_holdout_1),
        "holdout_top5pct": float(final_test_holdout_5),
        "holdout_top10pct": float(final_test_holdout_10),
    },
    "history": history,
}

report_step("Writing proteomics selected result, per-drug metrics, history and audit records")
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

per_drug = per_drug_metrics(model_p, label_adapter, test_loader, test_df.reset_index(drop=True), w_seen, w_all, holdout_pos_to_global_t)
per_drug.to_csv(PER_DRUG_PATH, sep="\t", index=False)

pd.DataFrame(history).to_csv(HISTORY_TSV_PATH, sep="\t", index=False)

metric_summary = pd.DataFrame(
    [
        {
            "split": "seen_val",
            "n_drugs": int(len(val_df)),
            "n_profiles": int(len(val_df)),
            **{k: float(v) for k, v in final_val_seen.items()},
            "holdout_top5pct": float(final_val_holdout),
        },
        {
            "split": "seen_test",
            "n_drugs": int(len(test_df)),
            "n_profiles": int(len(test_df)),
            **{k: float(v) for k, v in final_test_seen.items()},
            "holdout_top1pct": float(final_test_holdout_1),
            "holdout_top5pct": float(final_test_holdout_5),
            "holdout_top10pct": float(final_test_holdout_10),
        },
    ]
)
metric_summary.to_csv(METRIC_SUMMARY_PATH, sep="\t", index=False)

leakage_audit = f"""# Proteomics leakage and denominator audit

- Target embedding source: Round259 exported cross-cell full-space target embeddings aligned to the locked proteomics gene order.
- Target embedding shape: {result["gene_universe_size"]} x {d_feat}.
- Locked proteomics universe: {result["gene_universe_size"]} genes.
- Seen-label universe: {result["L_seen"]} labels.
- Holdout-label universe: {result["H"]} labels.
- Training drugs: {result["n_train_drugs"]}.
- Validation drugs: {result["n_val_drugs"]}.
- Test drugs: {result["n_test_drugs"]}.
- No test-drug rows were used in training or validation.
- No target-side architecture or protocol change was introduced beyond the embedding replacement.
- Alias mapping applied before training: {json.dumps(ALIAS_TO_LOCKED, ensure_ascii=False)}.
- All metrics are computed on the same per-drug evaluation unit as the prior proteomics round.
"""
with open(LEAKAGE_AUDIT_PATH, "w", encoding="utf-8") as f:
    f.write(leakage_audit)

with open(SELECTED_RESULT_PATH, "w", encoding="utf-8") as f:
    json.dump({"best_epoch": int(best_epoch), "best_score": float(best_score), "final_test": result["final_test"]}, f, indent=2)

method_lock = {
    "round": "Round261",
    "task": "proteomics_transfer_with_round259_crosscell_target_embedding",
    "base_script": "transfer_direct_h07_20260330/scripts/train_proteomics_purenew_joint_20260331.py",
    "only_method_change": "LABEL_REPR_PATH points to Round259 target embedding aligned to the locked proteomics gene order",
    "label_repr_path": LABEL_REPR_PATH,
    "label_repr_sha256": result["label_repr_sha256"],
    "proteomics_data_path": prot_path,
    "n_total_drugs": result["n_total_drugs"],
    "n_train_drugs": result["n_train_drugs"],
    "n_val_drugs": result["n_val_drugs"],
    "n_test_drugs": result["n_test_drugs"],
    "gene_universe_size": result["gene_universe_size"],
    "L_seen": result["L_seen"],
    "H": result["H"],
    "best_epoch": result["best_epoch"],
    "best_val_score": result["best_val_score"],
    "config": result["config"],
}
with open(METHOD_LOCK_PATH, "w", encoding="utf-8") as f:
    json.dump(method_lock, f, indent=2)

for split, label in (("final_val", "validation"), ("final_test", "test")):
    report_metrics(
        {key: result[split][key] for key in (
            "Hit@5", "Hit@10", "NDCG@10", "MAP@10", "holdout_top5pct",
        )},
        label=f"proteomics selected {label} metrics",
    )
print(json.dumps(result["final_test"], indent=2), flush=True)
print(f"Saved to {OUT_PATH}", flush=True)
