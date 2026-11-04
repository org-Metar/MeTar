#!/usr/bin/env python
"""
train_bulk_dualhead_v1_20260328.py

Dual-head bulk training script for MeTar.

Goals:
1. Keep the locked clean protocol, GNN label space, and MAML-style task training.
2. Separate seen-target optimization from unseen/open-world optimization.
3. Let the seen head maximize core benchmark metrics without forcing the unseen head
   into the same label geometry.
4. Export the unseen head embedding for downstream transfer.
"""

import argparse
import json
import logging
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from metar.runtime import report_metrics, report_progress, report_step, resolve_device
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR


PROJECT_ROOT = Path(
    os.environ.get(
        "METAR_BULK_PROJECT_ROOT",
        "/home/lwn/claude/manuscript/write/bulk_dualhead_20260328",
    )
)
DATA_DIR = Path("/home/lwn/claude/bulk/data")
MANIFEST_PATH = PROJECT_ROOT / "manifests" / "bulk_protocol_lock_v2.json"


DATE_GENE_FIX = {
    "1-Mar": "MARCHF1",
    "2-Mar": "MARCHF2",
    "3-Mar": "MARCHF3",
    "4-Mar": "MARCHF4",
    "5-Mar": "MARCHF5",
    "6-Mar": "MARCHF6",
    "7-Mar": "MARCHF7",
    "8-Mar": "MARCHF8",
    "9-Mar": "MARCHF9",
    "10-Mar": "MARCHF10",
    "11-Mar": "MARCHF11",
    "1-Sep": "SEPT1",
    "2-Sep": "SEPT2",
    "3-Sep": "SEPT3",
    "4-Sep": "SEPT4",
    "5-Sep": "SEPT5",
    "6-Sep": "SEPT6",
    "7-Sep": "SEPT7",
    "8-Sep": "SEPT8",
    "9-Sep": "SEPT9",
    "10-Sep": "SEPT10",
    "11-Sep": "SEPT11",
    "12-Sep": "SEPT12",
    "14-Sep": "SEPT14",
    "15-Sep": "SEPT15",
    "1-Dec": "DECR1",
    "2-Dec": "DECR2",
}


logger = logging.getLogger("bulk_dualhead_v1")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs() -> None:
    for name in ["artifacts", "checkpoints", "derived_tables", "logs", "scripts"]:
        (PROJECT_ROOT / name).mkdir(parents=True, exist_ok=True)


class GeneGNN(nn.Module):
    def __init__(
        self,
        n_genes: int,
        d_in: int = 64,
        d_hidden: int = 256,
        d_out: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_emb = nn.Embedding(n_genes, d_in)
        self.w1 = nn.Linear(d_in, d_hidden, bias=True)
        self.ln1 = nn.LayerNorm(d_hidden)
        self.w2 = nn.Linear(d_hidden, d_out, bias=True)
        self.ln2 = nn.LayerNorm(d_out)
        self.dropout = dropout
        nn.init.xavier_uniform_(self.node_emb.weight)
        nn.init.xavier_uniform_(self.w1.weight, gain=0.5)
        nn.init.xavier_uniform_(self.w2.weight, gain=0.5)

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        h = self.node_emb.weight
        h = torch.sparse.mm(adj, h)
        h = self.w1(h)
        h = self.ln1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.sparse.mm(adj, h)
        h = self.ln2(self.w2(h))
        return h


class GeneGNNResidual(nn.Module):
    def __init__(
        self,
        n_genes: int,
        d_in: int = 64,
        d_hidden: int = 256,
        d_out: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_emb = nn.Embedding(n_genes, d_in)
        self.w1 = nn.Linear(d_in, d_hidden, bias=True)
        self.ln1 = nn.LayerNorm(d_hidden)
        self.w2 = nn.Linear(d_hidden, d_out, bias=True)
        self.skip = nn.Linear(d_in, d_out, bias=False)
        self.ln2 = nn.LayerNorm(d_out)
        self.dropout = dropout
        nn.init.xavier_uniform_(self.node_emb.weight)
        nn.init.xavier_uniform_(self.w1.weight, gain=0.5)
        nn.init.xavier_uniform_(self.w2.weight, gain=0.5)
        nn.init.xavier_uniform_(self.skip.weight, gain=0.5)

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        h0 = self.node_emb.weight
        h = torch.sparse.mm(adj, h0)
        h = self.w1(h)
        h = self.ln1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.sparse.mm(adj, h)
        h = self.ln2(self.w2(h) + self.skip(h0))
        return h


class SampleEncoder(nn.Module):
    def __init__(self, in_dim: int = 978, hidden: int = 512, out_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)

    def functional_forward(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = F.linear(x, params["fc1.weight"], params["fc1.bias"])
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return F.linear(x, params["fc2.weight"], params["fc2.bias"])


class SampleEncoderResidual(nn.Module):
    def __init__(self, in_dim: int = 978, hidden: int = 512, out_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.dropout = dropout
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

    def functional_forward(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        h = F.linear(x, params["fc1.weight"], params["fc1.bias"])
        h = F.layer_norm(h, (h.shape[-1],), params["ln1.weight"], params["ln1.bias"])
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.linear(h, params["fc2.weight"], params["fc2.bias"])
        h = F.layer_norm(h, (h.shape[-1],), params["ln2.weight"], params["ln2.bias"])
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return F.linear(h, params["fc3.weight"], params["fc3.bias"]) + F.linear(x, params["skip.weight"], None)


class LabelEncoder(nn.Module):
    def __init__(self, in_dim: int = 128, hidden: int = 512, out_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))

    def functional_forward(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = F.linear(x, params["fc1.weight"], params["fc1.bias"])
        x = F.relu(x)
        return F.linear(x, params["fc2.weight"], params["fc2.bias"])


class LabelEncoderResidual(nn.Module):
    def __init__(self, in_dim: int = 128, hidden: int = 512, out_dim: int = 256, dropout: float = 0.10):
        super().__init__()
        self.dropout = dropout
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

    def functional_forward(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        h = F.linear(x, params["fc1.weight"], params["fc1.bias"])
        h = F.layer_norm(h, (h.shape[-1],), params["ln1.weight"], params["ln1.bias"])
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.linear(h, params["fc2.weight"], params["fc2.bias"])
        h = F.layer_norm(h, (h.shape[-1],), params["ln2.weight"], params["ln2.bias"])
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return F.linear(h, params["fc3.weight"], params["fc3.bias"]) + F.linear(x, params["skip.weight"], None)


class LabelEncoderLinear(nn.Module):
    def __init__(self, in_dim: int = 128, out_dim: int = 256):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def functional_forward(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.linear(x, params["fc.weight"], params["fc.bias"])


class MetaTwoTowerGNN(nn.Module):
    def __init__(
        self,
        gnn: nn.Module,
        n_expr_genes: int = 978,
        hidden: int = 512,
        out_dim: int = 256,
        sample_variant: str = "default",
        label_variant: str = "default",
    ):
        super().__init__()
        sample_cls = SampleEncoderResidual if sample_variant == "residual" else SampleEncoder
        if label_variant == "residual":
            label_cls = LabelEncoderResidual
        elif label_variant == "linear":
            label_cls = LabelEncoderLinear
        else:
            label_cls = LabelEncoder
        self.sample = sample_cls(in_dim=n_expr_genes, hidden=hidden, out_dim=out_dim)
        self.gnn = gnn
        if label_variant == "linear":
            self.label = label_cls(in_dim=128, out_dim=out_dim)
        else:
            self.label = label_cls(in_dim=128, hidden=hidden, out_dim=out_dim)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))

    def get_parameters_dict(self) -> Dict[str, torch.Tensor]:
        return {name: p for name, p in self.named_parameters()}

    @staticmethod
    def _subparams(params: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        return {k[len(prefix):]: v for k, v in params.items() if k.startswith(prefix)}

    def functional_logits(self, x: torch.Tensor, gene_repr: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        sp = self._subparams(params, "sample.")
        lp = self._subparams(params, "label.")
        logit_scale = params.get("logit_scale", self.logit_scale).exp().clamp(max=100.0)
        z = self.sample.functional_forward(x, sp)
        w = self.label.functional_forward(gene_repr, lp)
        z = F.normalize(z, dim=1)
        w = F.normalize(w, dim=1)
        return (z @ w.t()) * logit_scale

    def forward(self, x: torch.Tensor, gene_repr: torch.Tensor) -> torch.Tensor:
        z = F.normalize(self.sample(x), dim=1)
        w = F.normalize(self.label(gene_repr), dim=1)
        return (z @ w.t()) * self.logit_scale.exp().clamp(max=100.0)


class DualHeadMetaTwoTowerGNN(nn.Module):
    def __init__(
        self,
        gnn: nn.Module,
        n_expr_genes: int = 978,
        hidden: int = 512,
        out_dim: int = 256,
        seen_sample_variant: str = "residual",
        unseen_sample_variant: str = "residual",
        seen_label_variant: str = "residual",
        unseen_label_variant: str = "linear",
    ):
        super().__init__()
        seen_sample_cls = SampleEncoderResidual if seen_sample_variant == "residual" else SampleEncoder
        unseen_sample_cls = SampleEncoderResidual if unseen_sample_variant == "residual" else SampleEncoder
        self.sample_seen = seen_sample_cls(in_dim=n_expr_genes, hidden=hidden, out_dim=out_dim)
        self.sample_unseen = unseen_sample_cls(in_dim=n_expr_genes, hidden=hidden, out_dim=out_dim)
        self.gnn = gnn

        if seen_label_variant == "residual":
            self.label_seen = LabelEncoderResidual(in_dim=128, hidden=hidden, out_dim=out_dim)
        elif seen_label_variant == "linear":
            self.label_seen = LabelEncoderLinear(in_dim=128, out_dim=out_dim)
        else:
            self.label_seen = LabelEncoder(in_dim=128, hidden=hidden, out_dim=out_dim)

        if unseen_label_variant == "residual":
            self.label_unseen = LabelEncoderResidual(in_dim=128, hidden=hidden, out_dim=out_dim)
        elif unseen_label_variant == "linear":
            self.label_unseen = LabelEncoderLinear(in_dim=128, out_dim=out_dim)
        else:
            self.label_unseen = LabelEncoder(in_dim=128, hidden=hidden, out_dim=out_dim)

        self.logit_scale_seen = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))
        self.logit_scale_unseen = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))

    def get_parameters_dict(self) -> Dict[str, torch.Tensor]:
        return {name: p for name, p in self.named_parameters()}

    @staticmethod
    def _subparams(params: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        return {k[len(prefix):]: v for k, v in params.items() if k.startswith(prefix)}

    def _head_modules(self, head: str):
        if head == "seen":
            return self.sample_seen, self.label_seen, "sample_seen.", "label_seen.", "logit_scale_seen"
        if head == "unseen":
            return self.sample_unseen, self.label_unseen, "sample_unseen.", "label_unseen.", "logit_scale_unseen"
        raise ValueError(f"Unknown head: {head}")

    def sample_forward(self, x: torch.Tensor, head: str) -> torch.Tensor:
        sample_module, _, _, _, _ = self._head_modules(head)
        return sample_module(x)

    def label_forward(self, gene_repr: torch.Tensor, head: str) -> torch.Tensor:
        _, label_module, _, _, _ = self._head_modules(head)
        return label_module(gene_repr)

    def functional_logits(self, x: torch.Tensor, gene_repr: torch.Tensor, params: Dict[str, torch.Tensor], head: str) -> torch.Tensor:
        sample_module, label_module, sample_prefix, label_prefix, scale_key = self._head_modules(head)
        sp = self._subparams(params, sample_prefix)
        lp = self._subparams(params, label_prefix)
        logit_scale = params.get(scale_key, getattr(self, scale_key)).exp().clamp(max=100.0)
        z = sample_module.functional_forward(x, sp)
        w = label_module.functional_forward(gene_repr, lp)
        z = F.normalize(z, dim=1)
        w = F.normalize(w, dim=1)
        return (z @ w.t()) * logit_scale

    def forward(self, x: torch.Tensor, gene_repr: torch.Tensor, head: str) -> torch.Tensor:
        z = F.normalize(self.sample_forward(x, head=head), dim=1)
        w = F.normalize(self.label_forward(gene_repr, head), dim=1)
        scale = getattr(self, f"logit_scale_{head}").exp().clamp(max=100.0)
        return (z @ w.t()) * scale

def load_protocol(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_drug_to_targets() -> Dict[str, Tuple[str, ...]]:
    dti = pd.read_csv(DATA_DIR / "compoundinfo_beta.txt", sep="\t", usecols=["pert_id", "target"]).dropna()
    dti["pert_id"] = dti["pert_id"].astype(str)
    dti["target"] = dti["target"].astype(str).str.strip()
    dti = dti[dti["target"] != "0"].copy()
    return dti.groupby("pert_id")["target"].agg(lambda s: tuple(pd.unique(s))).to_dict()


def load_ppi_and_gene_universe(drug_to_targets: Dict[str, Tuple[str, ...]], device: torch.device):
    ppi = pd.read_csv(DATA_DIR / "pertkge_ppi_graph.csv")
    ppi_genes = sorted(set(ppi["start"].astype(str)) | set(ppi["end"].astype(str)))
    target_genes = sorted({g for genes in drug_to_targets.values() for g in genes})
    extra_genes = sorted(set(target_genes) - set(ppi_genes))
    gene_list = ppi_genes + extra_genes
    gene2idx = {g: i for i, g in enumerate(gene_list)}
    n_genes = len(gene_list)

    rows_list: List[int] = []
    cols_list: List[int] = []
    ppi_edges: List[Tuple[int, int]] = []
    for _, row in ppi.iterrows():
        s = str(row["start"])
        e = str(row["end"])
        if s in gene2idx and e in gene2idx:
            i = gene2idx[s]
            j = gene2idx[e]
            if i == j:
                continue
            rows_list.extend([i, j])
            cols_list.extend([j, i])
            ppi_edges.append((i, j))
    for i in range(n_genes):
        rows_list.append(i)
        cols_list.append(i)

    rows_arr = np.array(rows_list, dtype=np.int64)
    cols_arr = np.array(cols_list, dtype=np.int64)
    degree = np.bincount(rows_arr, minlength=n_genes).astype(np.float32)
    deg_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    vals_arr = (deg_inv_sqrt[rows_arr] * deg_inv_sqrt[cols_arr]).astype(np.float32)
    adj = torch.sparse_coo_tensor(
        torch.tensor(np.stack([rows_arr, cols_arr], axis=0), dtype=torch.long),
        torch.tensor(vals_arr, dtype=torch.float32),
        (n_genes, n_genes),
    ).coalesce().to(device)
    return gene_list, gene2idx, adj, np.array(ppi_edges, dtype=np.int64), set(ppi_edges)


def load_expression_dataframe(drug_to_targets: Dict[str, Tuple[str, ...]]) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_parquet(DATA_DIR / "large_data.parquet")
    df = df[df["method"] == 0].copy()
    df["drug"] = df["drug"].astype(str)
    df["cell_line"] = df["cell_line"].astype(str)
    df = df[df["drug"].isin(drug_to_targets)].copy()
    df["target"] = df["drug"].map(drug_to_targets)
    expr_cols = [c for c in df.columns if c not in ("drug", "cell_line", "method", "target")]
    return df, expr_cols


def list_to_multihot(genes: Tuple[str, ...], mapping: Dict[str, int], width: int) -> np.ndarray:
    vec = np.zeros(width, dtype=np.uint8)
    for gene in genes:
        idx = mapping.get(gene)
        if idx is not None:
            vec[idx] = 1
    return vec


def build_split_arrays(df_split: pd.DataFrame, expr_cols: List[str], label_mapping: Dict[str, int]) -> dict:
    df_split = df_split.reset_index(drop=True)
    x = df_split[expr_cols].to_numpy(dtype=np.float32)
    y = np.stack([list_to_multihot(tuple(t), label_mapping, len(label_mapping)) for t in df_split["target"]])
    pool = defaultdict(list)
    for idx, cell in enumerate(df_split["cell_line"].tolist()):
        pool[cell].append(idx)
    pool = {cell: np.asarray(idxs, dtype=np.int64) for cell, idxs in pool.items()}
    return {"df": df_split, "x": x, "y": y, "pool": pool}


def build_holdout_bundle(df_holdout: pd.DataFrame, expr_cols: List[str], split_targets: set, gene2idx: Dict[str, int]) -> dict:
    true_indices: List[List[int]] = []
    keep_rows: List[int] = []
    for i, targets in enumerate(df_holdout["target"].tolist()):
        idxs = [gene2idx[g] for g in targets if g in split_targets and g in gene2idx]
        if idxs:
            keep_rows.append(i)
            true_indices.append(idxs)
    df_keep = df_holdout.iloc[keep_rows].reset_index(drop=True).copy()
    return {
        "df": df_keep,
        "x": df_keep[expr_cols].to_numpy(dtype=np.float32),
        "true_indices": true_indices,
    }


def prepare_data(protocol: dict, device: torch.device) -> dict:
    report_step('Preparing bulk expression data and gene graph')
    drug_to_targets = build_drug_to_targets()
    report_step("Loading the PPI graph and gene universe")
    gene_list, gene2idx, adj_norm, ppi_edges, ppi_edges_set = load_ppi_and_gene_universe(drug_to_targets, device)
    report_step("Loading and filtering bulk expression responses")
    df0, expr_cols = load_expression_dataframe(drug_to_targets)

    holdout_val_targets = set(protocol["holdout_targets"]["holdout_targets_val"])
    holdout_test_targets = set(protocol["holdout_targets"]["holdout_targets_test"])
    all_holdout_targets = holdout_val_targets | holdout_test_targets

    train_cells = set(protocol["cell_split"]["train_cells"])
    holdout_cells = set(protocol["cell_split"]["holdout_cells"])
    inner_train_drugs = set(protocol["drug_split"]["inner_train_drugs"])
    val_drugs = set(protocol["drug_split"]["val_drugs"])
    test_drugs = set(protocol["drug_split"]["test_drugs"])

    holdout_mask = df0["target"].map(lambda genes: bool(set(genes) & all_holdout_targets))
    df_holdout_all = df0[holdout_mask].copy()
    df_seen_all = df0[~holdout_mask].copy()

    df_seen_traincells = df_seen_all[df_seen_all["cell_line"].isin(train_cells)].copy()
    df_seen_holdoutcells = df_seen_all[df_seen_all["cell_line"].isin(holdout_cells)].copy()

    inner_train_df = df_seen_traincells[df_seen_traincells["drug"].isin(inner_train_drugs)].copy()
    val_df_raw = df_seen_traincells[df_seen_traincells["drug"].isin(val_drugs)].copy()
    test_df_raw = df_seen_traincells[df_seen_traincells["drug"].isin(test_drugs)].copy()

    inner_train_targets = sorted({g for genes in inner_train_df["target"] for g in genes if g in gene2idx})
    train_gene2local = {g: i for i, g in enumerate(inner_train_targets)}
    inner_train_target_set = set(inner_train_targets)
    train_gene_global_idx = torch.tensor([gene2idx[g] for g in inner_train_targets], dtype=torch.long, device=device)

    val_subset_mask = val_df_raw["target"].map(lambda genes: set(genes).issubset(inner_train_target_set))
    test_subset_mask = test_df_raw["target"].map(lambda genes: set(genes).issubset(inner_train_target_set))
    val_df = val_df_raw[val_subset_mask].copy()
    test_df = test_df_raw[test_subset_mask].copy()

    df_holdout_val_traincells = df_holdout_all[
        df_holdout_all["cell_line"].isin(train_cells)
        & df_holdout_all["target"].map(lambda genes: bool(set(genes) & holdout_val_targets))
    ].copy()
    df_holdout_test_traincells = df_holdout_all[
        df_holdout_all["cell_line"].isin(train_cells)
        & df_holdout_all["target"].map(lambda genes: bool(set(genes) & holdout_test_targets))
    ].copy()

    report_step("Building bulk train, validation and test arrays")
    inner_split = build_split_arrays(inner_train_df, expr_cols, train_gene2local)
    val_split = build_split_arrays(val_df, expr_cols, train_gene2local)
    test_split = build_split_arrays(test_df, expr_cols, train_gene2local)

    report_step("Building held-out target evaluation arrays")
    holdout_val_bundle = build_holdout_bundle(df_holdout_val_traincells, expr_cols, holdout_val_targets, gene2idx)
    holdout_test_bundle = build_holdout_bundle(df_holdout_test_traincells, expr_cols, holdout_test_targets, gene2idx)

    audit = {
        "df0_rows_method0_mapped": int(len(df0)),
        "df_seen_all_rows": int(len(df_seen_all)),
        "df_holdout_all_rows": int(len(df_holdout_all)),
        "df_seen_traincells_rows": int(len(df_seen_traincells)),
        "df_seen_holdoutcells_rows": int(len(df_seen_holdoutcells)),
        "inner_train_rows": int(len(inner_train_df)),
        "val_rows_raw": int(len(val_df_raw)),
        "test_rows_raw": int(len(test_df_raw)),
        "val_rows_seen_label_clean": int(len(val_df)),
        "test_rows_seen_label_clean": int(len(test_df)),
        "val_rows_removed_unseen_label": int(len(val_df_raw) - len(val_df)),
        "test_rows_removed_unseen_label": int(len(test_df_raw) - len(test_df)),
        "n_train_label_genes": int(len(inner_train_targets)),
        "n_holdout_val_rows_traincells": int(len(holdout_val_bundle["x"])),
        "n_holdout_test_rows_traincells": int(len(holdout_test_bundle["x"])),
        "n_genes_full_universe": int(len(gene_list)),
        "n_train_cells_with_tasks": int(len(inner_split["pool"])),
    }

    audit_path = PROJECT_ROOT / "derived_tables" / "bulk_protocol_audit_v2.json"
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)

    audit_rows = [{"metric": key, "value": value} for key, value in audit.items()]
    pd.DataFrame(audit_rows).to_csv(PROJECT_ROOT / "derived_tables" / "bulk_protocol_audit_v2.tsv", sep="\t", index=False)

    logger.info("Protocol audit saved to derived_tables/bulk_protocol_audit_v2.{json,tsv}")
    for key, value in audit.items():
        logger.info("  %s: %s", key, value)

    return {
        "expr_cols": expr_cols,
        "gene_list": gene_list,
        "gene2idx": gene2idx,
        "adj_norm": adj_norm,
        "ppi_edges": ppi_edges,
        "ppi_edges_set": ppi_edges_set,
        "train_gene_global_idx": train_gene_global_idx,
        "train_label_genes": inner_train_targets,
        "inner_train": inner_split,
        "val": val_split,
        "test": test_split,
        "holdout_val": holdout_val_bundle,
        "holdout_test": holdout_test_bundle,
        "audit": audit,
    }


def pretrain_gnn(gnn: nn.Module, adj_norm: torch.Tensor, ppi_edges: np.ndarray, ppi_edges_set: set, n_genes: int, device: torch.device, epochs: int, lr: float) -> None:
    report_step('Pretraining gene graph encoder')
    opt = torch.optim.Adam(gnn.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = CosineAnnealingLR(opt, T_max=epochs)
    n_pos = len(ppi_edges)
    batch_size = 2048
    report_progress(0, epochs, label='pretrain gnn epochs')
    for epoch in range(1, epochs + 1):
        gnn.train()
        pos_idx = np.random.choice(n_pos, min(batch_size, n_pos), replace=False)
        pos_e = ppi_edges[pos_idx]
        neg_e = []
        attempts = 0
        while len(neg_e) < batch_size and attempts < batch_size * 20:
            i = np.random.randint(0, n_genes)
            j = np.random.randint(0, n_genes)
            if i != j and (i, j) not in ppi_edges_set and (j, i) not in ppi_edges_set:
                neg_e.append((i, j))
            attempts += 1
        neg_e = np.asarray(neg_e[:batch_size], dtype=np.int64)

        emb = gnn(adj_norm)
        emb = F.normalize(emb, dim=1)

        pos_i = torch.tensor(pos_e[:, 0], dtype=torch.long, device=device)
        pos_j = torch.tensor(pos_e[:, 1], dtype=torch.long, device=device)
        neg_i = torch.tensor(neg_e[:, 0], dtype=torch.long, device=device)
        neg_j = torch.tensor(neg_e[:, 1], dtype=torch.long, device=device)

        pos_scores = (emb[pos_i] * emb[pos_j]).sum(dim=1)
        neg_scores = (emb[neg_i] * emb[neg_j]).sum(dim=1)
        scores = torch.cat([pos_scores, neg_scores]) * 5.0
        labels = torch.cat([
            torch.ones(len(pos_i), device=device),
            torch.zeros(len(neg_i), device=device),
        ])
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gnn.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if epoch % 10 == 0 or epoch == epochs:
            logger.info("GNN pretrain epoch %02d/%02d | loss=%.4f", epoch, epochs, loss.item())
        report_progress(epoch, epochs, label='pretrain gnn epochs')


def build_label_mask(y: torch.Tensor) -> torch.Tensor:
    return (y > 0.5).any(dim=0).float().unsqueeze(0)


def masked_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None, eps: float = 1e-8) -> torch.Tensor:
    if mask is None or float(mask.sum()) == 0.0:
        return F.binary_cross_entropy_with_logits(logits, targets)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = loss * mask
    return loss.sum() / mask.sum().clamp(min=eps)


def pairwise_rank_loss(logits: torch.Tensor, targets: torch.Tensor, hard_negatives: int = 32) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    for row_logits, row_targets in zip(logits, targets):
        pos_idx = torch.where(row_targets > 0.5)[0]
        neg_idx = torch.where(row_targets <= 0.5)[0]
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            continue
        neg_scores = row_logits[neg_idx]
        k = min(hard_negatives, neg_scores.numel())
        hard_neg_idx = neg_idx[torch.topk(neg_scores, k=k).indices]
        pos_scores = row_logits[pos_idx]
        chosen_neg = row_logits[hard_neg_idx]
        diff = chosen_neg.unsqueeze(0) - pos_scores.unsqueeze(1)
        losses.append(F.softplus(diff).mean())
    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def positive_softmax_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pos_count = targets.sum(dim=1, keepdim=True)
    valid = pos_count.squeeze(1) > 0
    if not valid.any():
        return logits.new_tensor(0.0)
    target_dist = targets[valid] / pos_count[valid].clamp(min=eps)
    log_probs = F.log_softmax(logits[valid], dim=1)
    return -(target_dist * log_probs).sum(dim=1).mean()


def asymmetric_multilabel_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma_neg: float = 3.0,
    gamma_pos: float = 0.0,
    clip: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    prob_pos = torch.sigmoid(logits)
    prob_neg = 1.0 - prob_pos
    if clip > 0:
        prob_neg = (prob_neg + clip).clamp(max=1.0)
    pos_term = targets * torch.log(prob_pos.clamp(min=eps)) * torch.pow(1.0 - prob_pos, gamma_pos)
    neg_term = (1.0 - targets) * torch.log(prob_neg.clamp(min=eps)) * torch.pow(prob_pos, gamma_neg)
    return -(pos_term + neg_term).mean()


def geometry_preserve_loss(
    gene_repr_full: torch.Tensor,
    projected_gene_repr: torch.Tensor,
    sample_size: int = 384,
) -> torch.Tensor:
    n = gene_repr_full.shape[0]
    take = min(sample_size, n)
    idx = torch.randperm(n, device=gene_repr_full.device)[:take]
    raw = F.normalize(gene_repr_full[idx].detach(), dim=1)
    proj = F.normalize(projected_gene_repr[idx], dim=1)
    raw_sim = raw @ raw.t()
    proj_sim = proj @ proj.t()
    return F.mse_loss(proj_sim, raw_sim)


def rank_loss_weight(variant: str, epoch: int) -> float:
    if variant == "rankpair_v1":
        return 0.25
    if variant == "rankpair_lite_v2":
        return 0.10
    if variant == "rankpair_warm_v2":
        return 0.15 if epoch >= 5 else 0.0
    if variant in {
        "resmlp_rankpair_v1",
        "resmlp_rankpair_geom_v1",
        "resmlp_linearlabel_rankpair_v1",
        "resmlp_linearlabel_rankpair_geom_v1",
    }:
        return 0.25
    if variant in {"ranklist_v1", "resmlp_ranklist_v1"}:
        return 0.15
    if variant == "resmlp_linearlabel_ranklist_v1":
        return 0.15
    if variant in {"asl_rank_v1", "resmlp_asl_rank_v1"}:
        return 0.10
    return 0.0


def geometry_loss_weight(variant: str) -> float:
    if variant in {"resmlp_rankpair_geom_v1", "resmlp_linearlabel_rankpair_geom_v1"}:
        return 0.20
    return 0.0


def compute_query_loss(logits: torch.Tensor, targets: torch.Tensor, variant: str, epoch: int) -> torch.Tensor:
    if variant in {"asl_rank_v1", "resmlp_asl_rank_v1"}:
        base = asymmetric_multilabel_loss(logits, targets, gamma_neg=3.0, gamma_pos=0.0, clip=0.05)
        return (
            base
            + rank_loss_weight(variant, epoch) * pairwise_rank_loss(logits, targets, hard_negatives=48)
            + 0.10 * positive_softmax_loss(logits, targets)
        )

    bce = F.binary_cross_entropy_with_logits(logits, targets)
    rank_weight = rank_loss_weight(variant, epoch)
    listwise_weight = 0.0
    if variant in {"ranklist_v1", "resmlp_ranklist_v1", "resmlp_linearlabel_ranklist_v1"}:
        listwise_weight = 0.20
    if rank_weight <= 0.0 and listwise_weight <= 0.0:
        return bce
    return (
        bce
        + rank_weight * pairwise_rank_loss(logits, targets, hard_negatives=32)
        + listwise_weight * positive_softmax_loss(logits, targets)
    )


def compute_seen_head_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return (
        F.binary_cross_entropy_with_logits(logits, targets)
        + 0.15 * pairwise_rank_loss(logits, targets, hard_negatives=32)
        + 0.20 * positive_softmax_loss(logits, targets)
    )


def compute_unseen_head_loss(logits: torch.Tensor, targets_full: torch.Tensor) -> torch.Tensor:
    return (
        F.binary_cross_entropy_with_logits(logits, targets_full)
        + 0.25 * pairwise_rank_loss(logits, targets_full, hard_negatives=64)
    )


def scatter_local_targets_to_full(y_local: torch.Tensor, train_gene_global_idx: torch.Tensor, n_genes: int) -> torch.Tensor:
    y_full = torch.zeros((y_local.shape[0], n_genes), dtype=y_local.dtype, device=y_local.device)
    y_full[:, train_gene_global_idx] = y_local
    return y_full


def dualhead_geometry_loss(model: DualHeadMetaTwoTowerGNN, gene_repr_full: torch.Tensor, sample_size: int = 384) -> torch.Tensor:
    projected_full = model.label_forward(gene_repr_full, head="unseen")
    return geometry_preserve_loss(gene_repr_full, projected_full, sample_size=sample_size)


def compute_seen_selection_score(metrics: dict) -> float:
    return (
        0.45 * metrics["Hit@10"]
        + 0.20 * metrics["AUROC"]
        + 0.20 * metrics["NDCG@10"]
        + 0.15 * metrics["MeanAP"]
    )


def compute_unseen_selection_score(holdout_metrics: dict, seen_proxy_metrics: dict) -> float:
    return 0.85 * holdout_metrics["top5pct_success"] + 0.15 * seen_proxy_metrics["Hit@10"]


def sample_task_from_pool(split: dict, cell: str, support_size: int, query_size: int, device: torch.device):
    idxs = split["pool"].get(cell)
    if idxs is None or len(idxs) < support_size + query_size:
        return None
    chosen = np.random.choice(idxs, size=support_size + query_size, replace=False).astype(np.int64)
    s_idx = chosen[:support_size]
    q_idx = chosen[support_size:]
    x_s = torch.from_numpy(split["x"][s_idx]).to(device)
    y_s = torch.from_numpy(split["y"][s_idx].astype(np.float32)).to(device)
    x_q = torch.from_numpy(split["x"][q_idx]).to(device)
    y_q = torch.from_numpy(split["y"][q_idx].astype(np.float32)).to(device)
    return x_s, y_s, x_q, y_q


@torch.no_grad()
def evaluate_seen_split(model: MetaTwoTowerGNN, split: dict, gene_repr_subset: torch.Tensor, batch_size: int, device: torch.device) -> dict:
    if len(split["x"]) == 0:
        return {"Hit@10": 0.0, "AUROC": 0.5, "NDCG@10": 0.0, "MeanAP": 0.0, "n_samples": 0}

    model.eval()
    weights = F.normalize(model.label(gene_repr_subset), dim=1)
    scale = model.logit_scale.exp().clamp(max=100.0)
    preds = []
    labels = []

    x = split["x"]
    y = split["y"].astype(np.float32)
    for start in range(0, len(x), batch_size):
        report_progress(start, len(x), label='evaluate seen split rows')
        xb = torch.from_numpy(x[start:start + batch_size]).to(device)
        zb = F.normalize(model.sample(xb), dim=1)
        logits = (zb @ weights.t()) * scale
        preds.append(logits.cpu().numpy())
        labels.append(y[start:start + batch_size])
    report_progress(len(x), len(x), label='evaluate seen split rows')

    pred_mat = np.concatenate(preds, axis=0)
    label_mat = np.concatenate(labels, axis=0)
    n_samples = int(pred_mat.shape[0])

    topk_idx = np.argpartition(-pred_mat, kth=9, axis=1)[:, :10]
    hits = (label_mat[np.arange(n_samples)[:, None], topk_idx].sum(axis=1) > 0).mean()

    valid_auc = (label_mat.sum(axis=1) > 0) & (label_mat.sum(axis=1) < label_mat.shape[1])
    if valid_auc.any():
        auroc = float(roc_auc_score(label_mat[valid_auc], pred_mat[valid_auc], average="samples"))
    else:
        auroc = 0.5

    meanap = float(average_precision_score(label_mat, pred_mat, average="samples"))
    ndcg = float(ndcg_score(label_mat, pred_mat, k=10))

    return {
        "Hit@10": float(hits),
        "AUROC": auroc,
        "NDCG@10": ndcg,
        "MeanAP": meanap,
        "n_samples": n_samples,
    }


@torch.no_grad()
def evaluate_holdout_top5(model: MetaTwoTowerGNN, bundle: dict, full_gene_repr: torch.Tensor, batch_size: int, device: torch.device, k_percent: float = 0.05) -> dict:
    if len(bundle["x"]) == 0:
        return {"top5pct_success": 0.0, "n_samples": 0}

    model.eval()
    weights = F.normalize(model.label(full_gene_repr), dim=1)
    scale = model.logit_scale.exp().clamp(max=100.0)
    k = max(1, int(weights.shape[0] * k_percent))
    successes = 0
    total = 0

    for start in range(0, len(bundle["x"]), batch_size):
        report_progress(start, len(bundle["x"]), label='evaluate holdout top5 rows')
        end = min(start + batch_size, len(bundle["x"]))
        xb = torch.from_numpy(bundle["x"][start:end]).to(device)
        zb = F.normalize(model.sample(xb), dim=1)
        logits = (zb @ weights.t()) * scale
        topk_idx = torch.topk(logits, k=k, dim=1).indices.cpu().numpy()
        for row_topk, true_idx in zip(topk_idx, bundle["true_indices"][start:end]):
            row_topk_set = set(row_topk.tolist())
            if any(t in row_topk_set for t in true_idx):
                successes += 1
            total += 1
    report_progress(len(bundle["x"]), len(bundle["x"]), label='evaluate holdout top5 rows')

    return {"top5pct_success": successes / total if total > 0 else 0.0, "n_samples": int(total)}


@torch.no_grad()
def evaluate_seen_split_dual(
    model: DualHeadMetaTwoTowerGNN,
    split: dict,
    gene_repr_subset: torch.Tensor,
    batch_size: int,
    device: torch.device,
    head: str,
) -> dict:
    if len(split["x"]) == 0:
        return {"Hit@10": 0.0, "AUROC": 0.5, "NDCG@10": 0.0, "MeanAP": 0.0, "n_samples": 0}

    model.eval()
    weights = F.normalize(model.label_forward(gene_repr_subset, head=head), dim=1)
    scale = getattr(model, f"logit_scale_{head}").exp().clamp(max=100.0)
    preds = []
    labels = []
    x = split["x"]
    y = split["y"].astype(np.float32)

    for start in range(0, len(x), batch_size):
        report_progress(start, len(x), label='evaluate seen split dual rows')
        xb = torch.from_numpy(x[start:start + batch_size]).to(device)
        zb = F.normalize(model.sample_forward(xb, head=head), dim=1)
        logits = (zb @ weights.t()) * scale
        preds.append(logits.cpu().numpy())
        labels.append(y[start:start + batch_size])
    report_progress(len(x), len(x), label='evaluate seen split dual rows')

    pred_mat = np.concatenate(preds, axis=0)
    label_mat = np.concatenate(labels, axis=0)
    n_samples = int(pred_mat.shape[0])

    topk_idx = np.argpartition(-pred_mat, kth=9, axis=1)[:, :10]
    hits = (label_mat[np.arange(n_samples)[:, None], topk_idx].sum(axis=1) > 0).mean()
    valid_auc = (label_mat.sum(axis=1) > 0) & (label_mat.sum(axis=1) < label_mat.shape[1])
    if valid_auc.any():
        auroc = float(roc_auc_score(label_mat[valid_auc], pred_mat[valid_auc], average="samples"))
    else:
        auroc = 0.5

    meanap = float(average_precision_score(label_mat, pred_mat, average="samples"))
    ndcg = float(ndcg_score(label_mat, pred_mat, k=10))
    return {
        "Hit@10": float(hits),
        "AUROC": auroc,
        "NDCG@10": ndcg,
        "MeanAP": meanap,
        "n_samples": n_samples,
    }


@torch.no_grad()
def evaluate_holdout_top5_dual(
    model: DualHeadMetaTwoTowerGNN,
    bundle: dict,
    full_gene_repr: torch.Tensor,
    batch_size: int,
    device: torch.device,
    head: str = "unseen",
    k_percent: float = 0.05,
) -> dict:
    if len(bundle["x"]) == 0:
        return {"top5pct_success": 0.0, "n_samples": 0}

    model.eval()
    weights = F.normalize(model.label_forward(full_gene_repr, head=head), dim=1)
    scale = getattr(model, f"logit_scale_{head}").exp().clamp(max=100.0)
    k = max(1, int(weights.shape[0] * k_percent))
    successes = 0
    total = 0

    for start in range(0, len(bundle["x"]), batch_size):
        report_progress(start, len(bundle["x"]), label='evaluate holdout top5 dual rows')
        end = min(start + batch_size, len(bundle["x"]))
        xb = torch.from_numpy(bundle["x"][start:end]).to(device)
        zb = F.normalize(model.sample_forward(xb, head=head), dim=1)
        logits = (zb @ weights.t()) * scale
        topk_idx = torch.topk(logits, k=k, dim=1).indices.cpu().numpy()
        for row_topk, true_idx in zip(topk_idx, bundle["true_indices"][start:end]):
            row_topk_set = set(row_topk.tolist())
            if any(t in row_topk_set for t in true_idx):
                successes += 1
            total += 1
    report_progress(len(bundle["x"]), len(bundle["x"]), label='evaluate holdout top5 dual rows')

    return {"top5pct_success": successes / total if total > 0 else 0.0, "n_samples": int(total)}


def compute_model_selection_score(seen_metrics: dict, holdout_metrics: dict) -> float:
    return (
        0.40 * seen_metrics["Hit@10"]
        + 0.15 * seen_metrics["AUROC"]
        + 0.15 * seen_metrics["NDCG@10"]
        + 0.10 * seen_metrics["MeanAP"]
        + 0.20 * holdout_metrics["top5pct_success"]
    )


def export_gene_artifacts(model: MetaTwoTowerGNN, data: dict) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        gene_repr = model.gnn(data["adj_norm"])
        raw_repr = gene_repr.cpu().numpy()
        label_repr = F.normalize(model.label(gene_repr), dim=1).cpu().numpy()
    return raw_repr, label_repr


def export_gene_artifacts_dual(model: DualHeadMetaTwoTowerGNN, data: dict, head: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        gene_repr = model.gnn(data["adj_norm"])
        raw_repr = gene_repr.cpu().numpy()
        label_repr = F.normalize(model.label_forward(gene_repr, head=head), dim=1).cpu().numpy()
    return raw_repr, label_repr


def choose_gnn(variant: str, n_genes: int) -> nn.Module:
    if variant == "gnn_residual_v2":
        return GeneGNNResidual(n_genes=n_genes)
    return GeneGNN(n_genes=n_genes)


def choose_sample_variant(variant: str) -> str:
    if variant.startswith("resmlp_"):
        return "residual"
    return "default"


def choose_label_variant(variant: str) -> str:
    if "linearlabel" in variant:
        return "linear"
    if variant.startswith("resmlp_"):
        return "residual"
    return "default"


def is_dualhead_variant(variant: str) -> bool:
    return variant in {"dualhead_seenunseen_v1", "dualbranch_seenunseen_v2"}


def maybe_load_single_head_checkpoint(
    model: DualHeadMetaTwoTowerGNN,
    ckpt_path: str,
    branch: str,
    device: torch.device,
    load_gnn: bool = False,
) -> None:
    if not ckpt_path:
        return
    ckpt = torch.load(ckpt_path, map_location=device)
    src_state = ckpt["model_state_dict"]
    dst_state = model.state_dict()
    mapped_state = {}
    skipped_shape = []
    for key, value in src_state.items():
        if load_gnn and key.startswith("gnn."):
            dst_key = key
        elif key.startswith("sample."):
            dst_key = f"sample_{branch}.{key[len('sample.'):]}"
        elif key.startswith("label."):
            dst_key = f"label_{branch}.{key[len('label.'):]}"
        elif key == "logit_scale":
            dst_key = f"logit_scale_{branch}"
        else:
            continue

        if dst_key in dst_state and tuple(dst_state[dst_key].shape) == tuple(value.shape):
            mapped_state[dst_key] = value
        else:
            skipped_shape.append((key, dst_key, tuple(value.shape), tuple(dst_state[dst_key].shape) if dst_key in dst_state else None))

    missing, unexpected = model.load_state_dict(mapped_state, strict=False)
    logger.info(
        "Loaded %s branch checkpoint from %s | load_gnn=%s | mapped=%d | skipped_shape=%d | missing=%d | unexpected=%d",
        branch,
        ckpt_path,
        load_gnn,
        len(mapped_state),
        len(skipped_shape),
        len(missing),
        len(unexpected),
    )
    for src_key, dst_key, src_shape, dst_shape in skipped_shape[:8]:
        logger.info(
            "  Skipped incompatible tensor %s -> %s | src_shape=%s dst_shape=%s",
            src_key,
            dst_key,
            src_shape,
            dst_shape,
        )


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def train_dualhead_variant(args, data: dict, device: torch.device) -> dict:
    report_step('Preparing bulk dual-head training')
    set_seed(args.seed)
    variant = args.variant
    n_genes = len(data["gene_list"])
    train_gene_global_idx = data["train_gene_global_idx"]

    gnn = choose_gnn("baseline_clean_v6", n_genes).to(device)
    pretrain_gnn(
        gnn=gnn,
        adj_norm=data["adj_norm"],
        ppi_edges=data["ppi_edges"],
        ppi_edges_set=data["ppi_edges_set"],
        n_genes=n_genes,
        device=device,
        epochs=args.gnn_pretrain_epochs,
        lr=args.gnn_lr_pretrain,
    )

    model = DualHeadMetaTwoTowerGNN(
        gnn=gnn,
        n_expr_genes=len(data["expr_cols"]),
        seen_sample_variant=args.seen_sample_variant,
        unseen_sample_variant=args.unseen_sample_variant,
        seen_label_variant="residual",
        unseen_label_variant="linear",
    ).to(device)

    gnn_loaded = False
    if args.init_seen_ckpt:
        maybe_load_single_head_checkpoint(
            model=model,
            ckpt_path=args.init_seen_ckpt,
            branch="seen",
            device=device,
            load_gnn=args.gnn_init_source == "seen",
        )
        gnn_loaded = gnn_loaded or args.gnn_init_source == "seen"
    if args.init_unseen_ckpt:
        maybe_load_single_head_checkpoint(
            model=model,
            ckpt_path=args.init_unseen_ckpt,
            branch="unseen",
            device=device,
            load_gnn=(args.gnn_init_source == "unseen") and (not gnn_loaded),
        )

    if args.freeze_gnn:
        freeze_module(model.gnn)
    if args.freeze_seen_branch:
        freeze_module(model.sample_seen)
        freeze_module(model.label_seen)
        model.logit_scale_seen.requires_grad = False
    if args.freeze_unseen_branch:
        freeze_module(model.sample_unseen)
        freeze_module(model.label_unseen)
        model.logit_scale_unseen.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Variant %s | params=%s", variant, f"{total_params:,}")

    optimizer_groups = []
    if any(p.requires_grad for p in model.gnn.parameters()):
        optimizer_groups.append({"params": [p for p in model.gnn.parameters() if p.requires_grad], "lr": args.gnn_lr, "weight_decay": 1e-6})
    if any(p.requires_grad for p in model.sample_seen.parameters()):
        optimizer_groups.append({"params": [p for p in model.sample_seen.parameters() if p.requires_grad], "lr": args.other_lr, "weight_decay": 1e-5})
    if any(p.requires_grad for p in model.sample_unseen.parameters()):
        optimizer_groups.append({"params": [p for p in model.sample_unseen.parameters() if p.requires_grad], "lr": args.other_lr, "weight_decay": 1e-5})
    if any(p.requires_grad for p in model.label_seen.parameters()):
        optimizer_groups.append({"params": [p for p in model.label_seen.parameters() if p.requires_grad], "lr": args.other_lr, "weight_decay": 1e-5})
    if any(p.requires_grad for p in model.label_unseen.parameters()):
        optimizer_groups.append({"params": [p for p in model.label_unseen.parameters() if p.requires_grad], "lr": args.other_lr, "weight_decay": 1e-5})
    scale_params = []
    if model.logit_scale_seen.requires_grad:
        scale_params.append(model.logit_scale_seen)
    if model.logit_scale_unseen.requires_grad:
        scale_params.append(model.logit_scale_unseen)
    if scale_params:
        optimizer_groups.append({"params": scale_params, "lr": args.other_lr, "weight_decay": 0.0})
    if not optimizer_groups:
        raise RuntimeError("No trainable parameters remain after applying freeze flags.")

    optimizer = torch.optim.Adam(optimizer_groups)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    train_cells = [cell for cell, idxs in data["inner_train"]["pool"].items() if len(idxs) >= args.support_size + args.query_size]
    if not train_cells:
        raise RuntimeError("No train cells have enough samples for MAML tasks.")

    best_seen_score = -1.0
    best_unseen_score = -1.0
    best_seen_epoch = 0
    best_unseen_epoch = 0
    best_seen_path = PROJECT_ROOT / "checkpoints" / f"best_seen_{variant}.pth"
    best_unseen_path = PROJECT_ROOT / "checkpoints" / f"best_unseen_{variant}.pth"
    history_rows = []

    train_start = time.time()
    report_progress(0, args.n_epochs, label='dualhead variant epochs')
    report_metrics(vars(args), label="Training settings")
    for epoch in range(1, args.n_epochs + 1):
        model.train()
        meta_seen_total = 0.0
        meta_unseen_total = 0.0
        geom_loss_total = 0.0
        valid_steps = 0
        epoch_start = time.time()

        for _ in range(args.steps_per_epoch):
            report_progress(_, args.steps_per_epoch, label=f"Epoch {epoch} training episodes")
            sampled_cells = random.sample(train_cells, min(args.meta_batch, len(train_cells)))
            optimizer.zero_grad()

            gene_repr_full = model.gnn(data["adj_norm"])
            gene_repr_train = gene_repr_full[train_gene_global_idx]
            gene_repr_full_detached = gene_repr_full.detach()
            gene_repr_train_detached = gene_repr_train.detach()
            base_params = model.get_parameters_dict()
            seen_fast_keys = [
                k for k, v in base_params.items()
                if v.requires_grad and (k.startswith("sample_seen.") or k.startswith("label_seen.") or k == "logit_scale_seen")
            ]
            unseen_fast_keys = [
                k for k, v in base_params.items()
                if v.requires_grad and (k.startswith("sample_unseen.") or k.startswith("label_unseen.") or k == "logit_scale_unseen")
            ]

            batch_seen_loss = torch.tensor(0.0, device=device)
            batch_unseen_loss = torch.tensor(0.0, device=device)
            valid_tasks = 0

            for cell in sampled_cells:
                task = sample_task_from_pool(data["inner_train"], cell, args.support_size, args.query_size, device)
                if task is None:
                    continue
                x_s, y_s, x_q, y_q = task
                y_s_full = scatter_local_targets_to_full(y_s, train_gene_global_idx, n_genes)
                y_q_full = scatter_local_targets_to_full(y_q, train_gene_global_idx, n_genes)

                adapted_seen = dict(base_params)
                seen_label_mask = build_label_mask(y_s)
                if seen_fast_keys:
                    for _ in range(args.inner_steps):
                        logits_s_seen = model.functional_logits(x_s, gene_repr_train_detached, adapted_seen, head="seen")
                        loss_s_seen = masked_bce_with_logits(logits_s_seen, y_s, mask=seen_label_mask)
                        grads_seen = torch.autograd.grad(loss_s_seen, [adapted_seen[k] for k in seen_fast_keys], create_graph=False)
                        for key, grad in zip(seen_fast_keys, grads_seen):
                            adapted_seen[key] = adapted_seen[key] - args.inner_lr * grad

                adapted_unseen = dict(base_params)
                if unseen_fast_keys:
                    for _ in range(args.inner_steps):
                        logits_s_unseen = model.functional_logits(x_s, gene_repr_full_detached, adapted_unseen, head="unseen")
                        loss_s_unseen = compute_unseen_head_loss(logits_s_unseen, y_s_full)
                        grads_unseen = torch.autograd.grad(loss_s_unseen, [adapted_unseen[k] for k in unseen_fast_keys], create_graph=False)
                        for key, grad in zip(unseen_fast_keys, grads_unseen):
                            adapted_unseen[key] = adapted_unseen[key] - args.inner_lr * grad

                logits_q_seen = model.functional_logits(x_q, gene_repr_train, adapted_seen, head="seen")
                logits_q_unseen = model.functional_logits(x_q, gene_repr_full, adapted_unseen, head="unseen")
                batch_seen_loss = batch_seen_loss + compute_seen_head_loss(logits_q_seen, y_q)
                batch_unseen_loss = batch_unseen_loss + compute_unseen_head_loss(logits_q_unseen, y_q_full)
                valid_tasks += 1

            if valid_tasks == 0:
                continue

            batch_seen_loss = batch_seen_loss / valid_tasks
            batch_unseen_loss = batch_unseen_loss / valid_tasks
            batch_geom_loss = dualhead_geometry_loss(model, gene_repr_full, sample_size=384)
            total_loss = batch_seen_loss + args.unseen_loss_weight * batch_unseen_loss + args.unseen_geom_weight * batch_geom_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            meta_seen_total += batch_seen_loss.item()
            meta_unseen_total += batch_unseen_loss.item()
            geom_loss_total += batch_geom_loss.item()
            valid_steps += 1
        report_progress(args.steps_per_epoch, args.steps_per_epoch, label=f"Epoch {epoch} training episodes")

        scheduler.step()
        avg_seen_loss = meta_seen_total / max(valid_steps, 1)
        avg_unseen_loss = meta_unseen_total / max(valid_steps, 1)
        avg_geom_loss = geom_loss_total / max(valid_steps, 1)

        if epoch % args.eval_every == 0 or epoch == args.n_epochs:
            with torch.no_grad():
                gene_repr_eval = model.gnn(data["adj_norm"])
                gene_repr_train_eval = gene_repr_eval[train_gene_global_idx]

            val_seen_head = evaluate_seen_split_dual(model, data["val"], gene_repr_train_eval, args.eval_batch_size, device, head="seen")
            val_unseen_proxy = evaluate_seen_split_dual(model, data["val"], gene_repr_train_eval, args.eval_batch_size, device, head="unseen")
            val_holdout = evaluate_holdout_top5_dual(model, data["holdout_val"], gene_repr_eval, args.holdout_batch_size, device, head="unseen")
            seen_score = compute_seen_selection_score(val_seen_head)
            unseen_score = compute_unseen_selection_score(val_holdout, val_unseen_proxy)

            history_rows.append(
                {
                    "variant": variant,
                    "epoch": epoch,
                    "meta_seen_loss": avg_seen_loss,
                    "meta_unseen_loss": avg_unseen_loss,
                    "geom_loss": avg_geom_loss,
                    "val_seen_hit10": val_seen_head["Hit@10"],
                    "val_seen_auroc": val_seen_head["AUROC"],
                    "val_seen_ndcg10": val_seen_head["NDCG@10"],
                    "val_seen_map": val_seen_head["MeanAP"],
                    "val_unseen_proxy_hit10": val_unseen_proxy["Hit@10"],
                    "val_unseen_proxy_auroc": val_unseen_proxy["AUROC"],
                    "val_holdout_top5pct": val_holdout["top5pct_success"],
                    "seen_score": seen_score,
                    "unseen_score": unseen_score,
                    "epoch_time_s": time.time() - epoch_start,
                }
            )
            report_progress(epoch, args.n_epochs, label='dualhead variant epochs')
            report_metrics(
                {key: history_rows[-1][key] for key in (
                    "meta_seen_loss", "meta_unseen_loss", "val_seen_hit10",
                    "val_seen_map", "val_holdout_top5pct",
                )},
                label=f"Epoch {epoch}/{args.n_epochs}",
            )
            logger.info(
                "Epoch %02d/%02d | seen_loss=%.4f | unseen_loss=%.4f | geom=%.4f | seen_hit10=%.4f | seen_auc=%.4f | unseen_proxy_hit10=%.4f | holdout=%.4f | seen_score=%.4f | unseen_score=%.4f",
                epoch,
                args.n_epochs,
                avg_seen_loss,
                avg_unseen_loss,
                avg_geom_loss,
                val_seen_head["Hit@10"],
                val_seen_head["AUROC"],
                val_unseen_proxy["Hit@10"],
                val_holdout["top5pct_success"],
                seen_score,
                unseen_score,
            )

            if seen_score > best_seen_score:
                best_seen_score = seen_score
                best_seen_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "variant": variant,
                        "head": "seen",
                        "model_state_dict": model.state_dict(),
                        "seen_score": seen_score,
                    },
                    best_seen_path,
                )
                logger.info("  New best seen checkpoint: epoch=%d score=%.4f path=%s", epoch, seen_score, best_seen_path)

            if unseen_score > best_unseen_score:
                best_unseen_score = unseen_score
                best_unseen_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "variant": variant,
                        "head": "unseen",
                        "model_state_dict": model.state_dict(),
                        "unseen_score": unseen_score,
                    },
                    best_unseen_path,
                )
                logger.info("  New best unseen checkpoint: epoch=%d score=%.4f path=%s", epoch, unseen_score, best_unseen_path)

    total_time = time.time() - train_start
    logger.info("Training finished in %.1f s", total_time)

    fixed_gene_names = [DATE_GENE_FIX.get(g, g) for g in data["gene_list"]]

    seen_ckpt = torch.load(best_seen_path, map_location=device)
    model.load_state_dict(seen_ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        gene_repr_seen_final = model.gnn(data["adj_norm"])
        gene_repr_seen_train = gene_repr_seen_final[train_gene_global_idx]
    seen_val_metrics = evaluate_seen_split_dual(model, data["val"], gene_repr_seen_train, args.eval_batch_size, device, head="seen")
    seen_test_metrics = evaluate_seen_split_dual(model, data["test"], gene_repr_seen_train, args.eval_batch_size, device, head="seen")
    seen_label_path = PROJECT_ROOT / "artifacts" / f"gene_repr_labelproj_seen__{variant}.csv"
    if not args.skip_artifact_export:
        seen_raw_repr, seen_label_repr = export_gene_artifacts_dual(model, data, head="seen")
        pd.DataFrame(seen_raw_repr, index=fixed_gene_names).to_csv(PROJECT_ROOT / "artifacts" / f"gene_repr_gnn__{variant}__seenbest.csv")
        pd.DataFrame(seen_label_repr, index=fixed_gene_names).to_csv(seen_label_path)

    unseen_ckpt = torch.load(best_unseen_path, map_location=device)
    model.load_state_dict(unseen_ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        gene_repr_unseen_final = model.gnn(data["adj_norm"])
        gene_repr_unseen_train = gene_repr_unseen_final[train_gene_global_idx]
    unseen_proxy_val = evaluate_seen_split_dual(model, data["val"], gene_repr_unseen_train, args.eval_batch_size, device, head="unseen")
    unseen_proxy_test = evaluate_seen_split_dual(model, data["test"], gene_repr_unseen_train, args.eval_batch_size, device, head="unseen")
    unseen_holdout_val = evaluate_holdout_top5_dual(model, data["holdout_val"], gene_repr_unseen_final, args.holdout_batch_size, device, head="unseen")
    unseen_holdout_test = evaluate_holdout_top5_dual(model, data["holdout_test"], gene_repr_unseen_final, args.holdout_batch_size, device, head="unseen")
    unseen_label_path = PROJECT_ROOT / "artifacts" / f"gene_repr_labelproj_unseen__{variant}.csv"
    if not args.skip_artifact_export:
        unseen_raw_repr, unseen_label_repr = export_gene_artifacts_dual(model, data, head="unseen")
        pd.DataFrame(unseen_raw_repr, index=fixed_gene_names).to_csv(PROJECT_ROOT / "artifacts" / f"gene_repr_gnn__{variant}__unseenbest.csv")
        pd.DataFrame(unseen_label_repr, index=fixed_gene_names).to_csv(unseen_label_path)

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(PROJECT_ROOT / "derived_tables" / f"history_{variant}.tsv", sep="\t", index=False)

    results = {
        "variant": variant,
        "best_seen_epoch": best_seen_epoch,
        "best_seen_score": best_seen_score,
        "best_unseen_epoch": best_unseen_epoch,
        "best_unseen_score": best_unseen_score,
        "seen_head": {
            "val_metrics": seen_val_metrics,
            "test_metrics": seen_test_metrics,
            "checkpoint_path": str(best_seen_path),
            "label_embedding_path": str(seen_label_path) if not args.skip_artifact_export else "",
        },
        "unseen_head": {
            "val_proxy_metrics": unseen_proxy_val,
            "test_proxy_metrics": unseen_proxy_test,
            "val_holdout_top5pct": unseen_holdout_val["top5pct_success"],
            "val_holdout_n_samples": unseen_holdout_val["n_samples"],
            "test_holdout_top5pct": unseen_holdout_test["top5pct_success"],
            "test_holdout_n_samples": unseen_holdout_test["n_samples"],
            "checkpoint_path": str(best_unseen_path),
            "label_embedding_path": str(unseen_label_path) if not args.skip_artifact_export else "",
        },
        "config": {
            "n_epochs": args.n_epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "meta_batch": args.meta_batch,
            "support_size": args.support_size,
            "query_size": args.query_size,
            "inner_steps": args.inner_steps,
            "inner_lr": args.inner_lr,
            "gnn_lr": args.gnn_lr,
            "other_lr": args.other_lr,
            "gnn_pretrain_epochs": args.gnn_pretrain_epochs,
            "eval_every": args.eval_every,
            "seed": args.seed,
            "seen_sample_variant": args.seen_sample_variant,
            "unseen_sample_variant": args.unseen_sample_variant,
            "seen_label_variant": "residual",
            "unseen_label_variant": "linear",
            "unseen_loss_weight": args.unseen_loss_weight,
            "unseen_geom_weight": args.unseen_geom_weight,
            "init_seen_ckpt": args.init_seen_ckpt,
            "init_unseen_ckpt": args.init_unseen_ckpt,
            "gnn_init_source": args.gnn_init_source,
            "freeze_gnn": args.freeze_gnn,
            "freeze_seen_branch": args.freeze_seen_branch,
            "freeze_unseen_branch": args.freeze_unseen_branch,
            "skip_artifact_export": args.skip_artifact_export,
        },
        "protocol_audit": data["audit"],
        "total_train_time_s": total_time,
    }
    with open(PROJECT_ROOT / "derived_tables" / f"results_{variant}.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def train_variant(args, data: dict, device: torch.device) -> dict:
    report_step('Preparing bulk training')
    if is_dualhead_variant(args.variant):
        return train_dualhead_variant(args, data, device)

    set_seed(args.seed)
    variant = args.variant
    n_genes = len(data["gene_list"])

    gnn = choose_gnn(variant, n_genes).to(device)
    pretrain_gnn(
        gnn=gnn,
        adj_norm=data["adj_norm"],
        ppi_edges=data["ppi_edges"],
        ppi_edges_set=data["ppi_edges_set"],
        n_genes=n_genes,
        device=device,
        epochs=args.gnn_pretrain_epochs,
        lr=args.gnn_lr_pretrain,
    )

    model = MetaTwoTowerGNN(
        gnn=gnn,
        n_expr_genes=len(data["expr_cols"]),
        sample_variant=choose_sample_variant(variant),
        label_variant=choose_label_variant(variant),
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Variant %s | params=%s", variant, f"{total_params:,}")

    optimizer = torch.optim.Adam(
        [
            {"params": model.gnn.parameters(), "lr": args.gnn_lr, "weight_decay": 1e-6},
            {"params": model.sample.parameters(), "lr": args.other_lr, "weight_decay": 1e-5},
            {"params": model.label.parameters(), "lr": args.other_lr, "weight_decay": 1e-5},
            {"params": [model.logit_scale], "lr": args.other_lr, "weight_decay": 0.0},
        ]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    train_cells = [cell for cell, idxs in data["inner_train"]["pool"].items() if len(idxs) >= args.support_size + args.query_size]
    if not train_cells:
        raise RuntimeError("No train cells have enough samples for MAML tasks.")

    best_val_score = -1.0
    best_epoch = 0
    best_path = PROJECT_ROOT / "checkpoints" / f"best_{variant}.pth"
    history_rows = []

    train_start = time.time()
    report_progress(0, args.n_epochs, label='variant epochs')
    report_metrics(vars(args), label="Training settings")
    for epoch in range(1, args.n_epochs + 1):
        model.train()
        meta_loss_total = 0.0
        geom_loss_total = 0.0
        valid_steps = 0
        epoch_start = time.time()

        for _ in range(args.steps_per_epoch):
            report_progress(_, args.steps_per_epoch, label=f"Epoch {epoch} training episodes")
            sampled_cells = random.sample(train_cells, min(args.meta_batch, len(train_cells)))
            optimizer.zero_grad()

            gene_repr_full = model.gnn(data["adj_norm"])
            gene_repr_train = gene_repr_full[data["train_gene_global_idx"]]
            gene_repr_train_detached = gene_repr_train.detach()
            base_params = model.get_parameters_dict()
            fast_keys = [k for k in base_params if k.startswith("sample.") or k.startswith("label.")]

            batch_meta_loss = torch.tensor(0.0, device=device)
            valid_tasks = 0
            for cell in sampled_cells:
                task = sample_task_from_pool(data["inner_train"], cell, args.support_size, args.query_size, device)
                if task is None:
                    continue
                x_s, y_s, x_q, y_q = task
                adapted = dict(base_params)
                label_mask = build_label_mask(y_s)

                for _ in range(args.inner_steps):
                    logits_s = model.functional_logits(x_s, gene_repr_train_detached, adapted)
                    loss_s = masked_bce_with_logits(logits_s, y_s, mask=label_mask)
                    grads = torch.autograd.grad(loss_s, [adapted[k] for k in fast_keys], create_graph=False)
                    for key, grad in zip(fast_keys, grads):
                        adapted[key] = adapted[key] - args.inner_lr * grad

                logits_q = model.functional_logits(x_q, gene_repr_train, adapted)
                loss_q = compute_query_loss(logits_q, y_q, variant=variant, epoch=epoch)
                batch_meta_loss = batch_meta_loss + loss_q
                valid_tasks += 1

            if valid_tasks == 0:
                continue

            batch_meta_loss = batch_meta_loss / valid_tasks
            batch_geom_loss = torch.tensor(0.0, device=device)
            geom_weight = geometry_loss_weight(variant)
            if geom_weight > 0.0:
                projected_full = model.label(gene_repr_full)
                batch_geom_loss = geometry_preserve_loss(gene_repr_full, projected_full, sample_size=384)

            total_loss = batch_meta_loss + geom_weight * batch_geom_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            meta_loss_total += batch_meta_loss.item()
            geom_loss_total += batch_geom_loss.item()
            valid_steps += 1
        report_progress(args.steps_per_epoch, args.steps_per_epoch, label=f"Epoch {epoch} training episodes")

        scheduler.step()
        avg_meta_loss = meta_loss_total / max(valid_steps, 1)
        avg_geom_loss = geom_loss_total / max(valid_steps, 1)

        if epoch % args.eval_every == 0 or epoch == args.n_epochs:
            with torch.no_grad():
                gene_repr_eval = model.gnn(data["adj_norm"])
                gene_repr_train_eval = gene_repr_eval[data["train_gene_global_idx"]]

            val_seen = evaluate_seen_split(model, data["val"], gene_repr_train_eval, args.eval_batch_size, device)
            val_holdout = evaluate_holdout_top5(model, data["holdout_val"], gene_repr_eval, args.holdout_batch_size, device)
            val_score = compute_model_selection_score(val_seen, val_holdout)

            row = {
                "variant": variant,
                "epoch": epoch,
                "meta_loss": avg_meta_loss,
                "geom_loss": avg_geom_loss,
                "val_hit10": val_seen["Hit@10"],
                "val_auroc": val_seen["AUROC"],
                "val_ndcg10": val_seen["NDCG@10"],
                "val_meanap": val_seen["MeanAP"],
                "val_holdout_top5pct": val_holdout["top5pct_success"],
                "val_score": val_score,
                "val_seen_n": val_seen["n_samples"],
                "val_holdout_n": val_holdout["n_samples"],
                "epoch_time_s": time.time() - epoch_start,
            }
            history_rows.append(row)
            report_progress(epoch, args.n_epochs, label='variant epochs')
            report_metrics(
                {key: history_rows[-1][key] for key in (
                    "meta_loss", "val_hit10", "val_meanap", "val_holdout_top5pct",
                )},
                label=f"Epoch {epoch}/{args.n_epochs}",
            )
            logger.info(
                "Epoch %02d/%02d | loss=%.4f | geom=%.4f | val_hit10=%.4f | val_auroc=%.4f | val_ndcg10=%.4f | val_map=%.4f | val_holdout=%.4f | val_score=%.4f",
                epoch,
                args.n_epochs,
                avg_meta_loss,
                avg_geom_loss,
                val_seen["Hit@10"],
                val_seen["AUROC"],
                val_seen["NDCG@10"],
                val_seen["MeanAP"],
                val_holdout["top5pct_success"],
                val_score,
            )

            if val_score > best_val_score:
                best_val_score = val_score
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "variant": variant,
                        "model_state_dict": model.state_dict(),
                        "val_score": val_score,
                    },
                    best_path,
                )
                logger.info("  New best checkpoint: epoch=%d score=%.4f path=%s", epoch, val_score, best_path)

    total_time = time.time() - train_start
    logger.info("Training finished in %.1f s", total_time)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        gene_repr_final = model.gnn(data["adj_norm"])
        gene_repr_train_final = gene_repr_final[data["train_gene_global_idx"]]

    val_seen = evaluate_seen_split(model, data["val"], gene_repr_train_final, args.eval_batch_size, device)
    val_holdout = evaluate_holdout_top5(model, data["holdout_val"], gene_repr_final, args.holdout_batch_size, device)
    test_seen = evaluate_seen_split(model, data["test"], gene_repr_train_final, args.eval_batch_size, device)
    test_holdout = evaluate_holdout_top5(model, data["holdout_test"], gene_repr_final, args.holdout_batch_size, device)

    raw_repr, label_repr = export_gene_artifacts(model, data)
    fixed_gene_names = [DATE_GENE_FIX.get(g, g) for g in data["gene_list"]]
    pd.DataFrame(raw_repr, index=fixed_gene_names).to_csv(PROJECT_ROOT / "artifacts" / f"gene_repr_gnn__{variant}.csv")
    pd.DataFrame(label_repr, index=fixed_gene_names).to_csv(PROJECT_ROOT / "artifacts" / f"gene_repr_labelproj__{variant}.csv")

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(PROJECT_ROOT / "derived_tables" / f"history_{variant}.tsv", sep="\t", index=False)

    results = {
        "variant": variant,
        "best_epoch": best_epoch,
        "best_val_score": best_val_score,
        "val_metrics": {
            **val_seen,
            "holdout_top5pct": val_holdout["top5pct_success"],
            "holdout_n_samples": val_holdout["n_samples"],
        },
        "test_metrics": {
            **test_seen,
            "holdout_top5pct": test_holdout["top5pct_success"],
            "holdout_n_samples": test_holdout["n_samples"],
        },
        "config": {
            "n_epochs": args.n_epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "meta_batch": args.meta_batch,
            "support_size": args.support_size,
            "query_size": args.query_size,
            "inner_steps": args.inner_steps,
            "inner_lr": args.inner_lr,
            "gnn_lr": args.gnn_lr,
            "other_lr": args.other_lr,
            "gnn_pretrain_epochs": args.gnn_pretrain_epochs,
            "eval_every": args.eval_every,
            "seed": args.seed,
            "sample_variant": choose_sample_variant(variant),
            "label_variant": choose_label_variant(variant),
            "geometry_loss_weight": geometry_loss_weight(variant),
        },
        "protocol_audit": data["audit"],
        "total_train_time_s": total_time,
    }
    with open(PROJECT_ROOT / "derived_tables" / f"results_{variant}.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


VARIANTS = [
    "baseline_clean_v6",
    "rankpair_v1",
    "rankpair_lite_v2",
    "rankpair_warm_v2",
    "gnn_residual_v2",
    "ranklist_v1",
    "asl_rank_v1",
    "resmlp_ranklist_v1",
    "resmlp_asl_rank_v1",
    "resmlp_rankpair_v1",
    "resmlp_rankpair_geom_v1",
    "resmlp_linearlabel_rankpair_v1",
    "resmlp_linearlabel_rankpair_geom_v1",
    "resmlp_linearlabel_ranklist_v1",
    "dualhead_seenunseen_v1",
    "dualbranch_seenunseen_v2",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Clean MeTar bulk training script")
    parser.add_argument("--variant", type=str, required=True, choices=VARIANTS)
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_epochs", type=int, default=24)
    parser.add_argument("--steps_per_epoch", type=int, default=80)
    parser.add_argument("--meta_batch", type=int, default=8)
    parser.add_argument("--support_size", type=int, default=32)
    parser.add_argument("--query_size", type=int, default=64)
    parser.add_argument("--inner_steps", type=int, default=3)
    parser.add_argument("--inner_lr", type=float, default=0.01)
    parser.add_argument("--gnn_lr", type=float, default=1e-4)
    parser.add_argument("--gnn_lr_pretrain", type=float, default=1e-3)
    parser.add_argument("--other_lr", type=float, default=1e-3)
    parser.add_argument("--gnn_pretrain_epochs", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--holdout_batch_size", type=int, default=48)
    parser.add_argument("--unseen_loss_weight", type=float, default=1.0)
    parser.add_argument("--unseen_geom_weight", type=float, default=0.10)
    parser.add_argument("--seen_sample_variant", type=str, default="residual", choices=["default", "residual"])
    parser.add_argument("--unseen_sample_variant", type=str, default="residual", choices=["default", "residual"])
    parser.add_argument("--init_seen_ckpt", type=str, default="")
    parser.add_argument("--init_unseen_ckpt", type=str, default="")
    parser.add_argument("--gnn_init_source", type=str, default="none", choices=["none", "seen", "unseen"])
    parser.add_argument("--freeze_gnn", action="store_true")
    parser.add_argument("--freeze_seen_branch", action="store_true")
    parser.add_argument("--freeze_unseen_branch", action="store_true")
    parser.add_argument("--skip_artifact_export", action="store_true")
    return parser.parse_args()


def configure_logging(variant: str) -> None:
    logger.setLevel(logging.INFO)
    logger.handlers = []
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(PROJECT_ROOT / "logs" / f"train_{variant}.log", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)


def main():
    ensure_dirs()
    args = parse_args()
    configure_logging(args.variant)
    set_seed(args.seed)
    device = torch.device(resolve_device(args.device))
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Variant: %s", args.variant)
    logger.info("Device: %s", device)
    logger.info("Manifest: %s", MANIFEST_PATH)

    protocol = load_protocol(MANIFEST_PATH)
    data = prepare_data(protocol, device)
    results = train_variant(args, data, device)
    logger.info("Final result summary:")
    if "test_metrics" in results:
        logger.info(json.dumps(results["test_metrics"], indent=2))
    else:
        logger.info(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
