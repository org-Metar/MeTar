from __future__ import annotations

import ast
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


SEED = 2026
VAL_FRAC = 0.2
PROJ_DIM = 256
N_RES = 1
DROPOUT = 0.45
INIT_SCALE = 10.0
BATCH_SIZE = 16
TOPK_FEAT = 2000
LABEL_ADAPTER_MODE = "linear"
LABEL_BOTTLENECK = 64
L_FULL = 16583
THRESHOLDS = (0.01, 0.05, 0.10)

ROUND_ROOT = Path(
    "/home/lwn/claude/manuscript/write/"
    "claude_round259_crosscell_fullspace_round252query_then_sc_proteomics_20260522"
)
PROTEOMICS_DATA = Path("/home/lwn/claude/pt/data/proteomicsdb_data.csv")
LABEL_REPR = ROUND_ROOT / "source_data" / "round259_target_embedding_aligned_to_proteomics_gene_order.csv"
CHECKPOINT = ROUND_ROOT / "checkpoints" / "round261_proteomics_round259_embedding.pth"
REFERENCE_PER_DRUG = ROUND_ROOT / "results" / "round261_proteomics_round259_embedding_per_drug.tsv"
OUT_DIR = Path(
    "/home/lwn/claude/manuscript/write/"
    "claude_round505_fig6_holdout_numeric_percentile_20260706/source_data"
)
OUT_TSV = OUT_DIR / "Figure6_R505_holdout_drug_numeric_percentile_source.tsv"
OUT_AUDIT = OUT_DIR / "Figure6_R505_holdout_numeric_percentile_audit.json"


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class ProteoDataset(Dataset):
    def __init__(self, x: np.ndarray, y_seen: np.ndarray, y_hold: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y_seen = torch.tensor(y_seen, dtype=torch.float32)
        self.y_hold = torch.tensor(y_hold, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward_embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        for block in self.res_blocks:
            x = block(x)
        return F.normalize(self.head(x), dim=1)


class LabelAdapter(nn.Module):
    def __init__(self, dim: int, mode: str = "linear", bottleneck: int = 64):
        super().__init__()
        self.mode = mode
        if mode == "identity":
            self.core = None
        elif mode == "linear":
            self.core = nn.Linear(dim, dim, bias=False)
        elif mode == "residual_mlp":
            self.down = nn.Linear(dim, bottleneck, bias=False)
            self.up = nn.Linear(bottleneck, dim, bias=False)
            self.gate = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        else:
            raise ValueError(f"Unknown label adapter mode: {mode}")

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        if self.mode == "identity":
            return F.normalize(w, dim=1)
        if self.mode == "linear":
            return F.normalize(self.core(w), dim=1)
        scale = torch.sigmoid(self.gate)
        delta = self.up(F.gelu(self.down(w)))
        return F.normalize(w + scale * delta, dim=1)


def split_by_drug(df, all_drugs, drug_targets, test_size=0.2, seed=0):
    rng = np.random.RandomState(seed)
    drugs = all_drugs.copy()
    rng.shuffle(drugs)
    n_test = max(1, int(len(drugs) * test_size))
    test_drugs = drugs[:n_test]
    train_drugs = drugs[n_test:]
    train_targets = set()
    for drug in train_drugs:
        train_targets |= drug_targets[drug]
    valid_test = [drug for drug in test_drugs if drug_targets[drug] & train_targets]
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
    for drug in raw_train:
        train_targets |= drug_targets[drug]
    val_drugs = np.array([drug for drug in raw_val if drug_targets[drug] & train_targets])
    if len(val_drugs) == 0:
        val_drugs = np.array(raw_val[:1])
    train_drugs2 = np.array([drug for drug in drugs if drug not in set(val_drugs.tolist())])
    return train_drugs2, val_drugs


def load_locked_data():
    df = pd.read_csv(PROTEOMICS_DATA, index_col=0)
    df["target"] = df["target"].apply(ast.literal_eval)
    df = df[df["target"].apply(len) > 0].copy()
    df["drug"] = df["drug"].astype(str).str.strip().str.lower()
    df = df.reset_index(drop=True)

    df_w = pd.read_csv(LABEL_REPR, index_col=0)
    df_w = df_w[~df_w.index.duplicated(keep="first")]
    all_embed_genes = list(df_w.index)
    gene_to_global = {gene: i for i, gene in enumerate(all_embed_genes)}
    w_all_np = df_w.loc[all_embed_genes].values.astype(np.float32)
    w_all = F.normalize(torch.tensor(w_all_np, dtype=torch.float32), dim=1)

    def filter_targets(targets: List) -> List[str]:
        seen = set()
        out = []
        for target in targets:
            target = str(target)
            if target in gene_to_global and target not in seen:
                out.append(target)
                seen.add(target)
        return out

    df["target_filt"] = df["target"].apply(filter_targets)
    df = df[df["target_filt"].apply(len) > 0].copy().reset_index(drop=True)

    drug_targets: Dict[str, set] = {}
    for drug, group in df.groupby("drug"):
        merged = set()
        for targets in group["target_filt"]:
            merged |= set(targets)
        drug_targets[drug] = merged

    all_drugs = np.array(sorted(drug_targets.keys()))
    all_targets = sorted({target for drug in all_drugs for target in drug_targets[drug]})
    random.seed(42)
    holdout_targets_list = random.sample(all_targets, min(50, len(all_targets)))
    holdout_targets_set = set(holdout_targets_list)
    holdout_pos = {gene: i for i, gene in enumerate(holdout_targets_list)}
    holdout_pos_to_global = np.array([gene_to_global[gene] for gene in holdout_targets_list], dtype=np.int64)

    seen_targets = sorted(set(all_targets) - holdout_targets_set)
    seen_gene2local = {gene: i for i, gene in enumerate(seen_targets)}
    seen_global_idx = np.array([gene_to_global[gene] for gene in seen_targets], dtype=np.int64)
    w_seen = F.normalize(w_all[torch.tensor(seen_global_idx)], dim=1)

    def to_seen_mh(targets):
        values = np.zeros(len(seen_targets), dtype=np.float32)
        for target in targets:
            if target in seen_gene2local:
                values[seen_gene2local[target]] = 1.0
        return values

    def to_hold_mh(targets):
        values = np.zeros(len(holdout_targets_list), dtype=np.float32)
        for target in targets:
            if target in holdout_pos:
                values[holdout_pos[target]] = 1.0
        return values

    y_seen_full = np.stack([to_seen_mh(targets) for targets in df["target_filt"]]).astype(np.float32)
    y_hold_full = np.stack([to_hold_mh(targets) for targets in df["target_filt"]]).astype(np.float32)

    feature_cols = [col for col in df.columns if col not in ["drug", "target", "target_filt"]]
    x_raw = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32)

    _, test_df, train_drugs_full, _ = split_by_drug(df, all_drugs, drug_targets, test_size=0.2, seed=0)
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
    x_test = (x_test_raw - mu) / sd

    y_test = y_seen_full[test_idx]
    yh_test = y_hold_full[test_idx]

    return {
        "test_df": test_df.reset_index(drop=True),
        "x_test": x_test,
        "y_test": y_test,
        "yh_test": yh_test,
        "w_all": w_all,
        "w_seen": w_seen,
        "holdout_pos_to_global": holdout_pos_to_global,
        "n_train_drugs": len(train_drugs),
        "n_val_drugs": len(val_drugs),
        "n_test_drugs": len(test_df),
    }


@torch.no_grad()
def derive_percentiles() -> pd.DataFrame:
    data = load_locked_data()
    test_loader = DataLoader(
        ProteoDataset(data["x_test"], data["y_test"], data["yh_test"]),
        batch_size=max(1, len(data["x_test"])),
        shuffle=False,
        drop_last=False,
    )
    w_all = data["w_all"]
    model = DeepProteoMLP(
        in_dim=data["x_test"].shape[1],
        proj_dim=PROJ_DIM,
        d=w_all.shape[1],
        n_res=N_RES,
        dropout=DROPOUT,
        init_scale=INIT_SCALE,
    )
    adapter = LabelAdapter(dim=w_all.shape[1], mode=LABEL_ADAPTER_MODE, bottleneck=LABEL_BOTTLENECK)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    adapter.load_state_dict(checkpoint["label_adapter_state_dict"])
    model.eval()
    adapter.eval()

    rows = []
    offset = 0
    w_all_adapt = adapter(w_all)
    for xb, _, y_hold_b in test_loader:
        logits = (model.forward_embed(xb) @ w_all_adapt.t()) * model.logit_scale.exp().clamp(max=100.0)
        logits_np = logits.cpu().numpy()
        for i, scores in enumerate(logits_np):
            src = data["test_df"].iloc[offset + i]
            hold_pos = torch.where(y_hold_b[i] > 0.5)[0].numpy()
            if len(hold_pos) == 0:
                continue
            true_global = data["holdout_pos_to_global"][hold_pos]
            order = np.argsort(-scores)
            inverse_rank = np.empty_like(order)
            inverse_rank[order] = np.arange(1, len(order) + 1)
            ranks = sorted(int(inverse_rank[idx]) for idx in true_global)
            best_rank = min(ranks)
            best_percentile = best_rank / float(len(order))
            threshold_hits = {
                f"holdout_top{int(threshold * 100)}pct": float(best_rank <= max(1, int(len(order) * threshold)))
                for threshold in THRESHOLDS
            }
            if threshold_hits["holdout_top1pct"] >= 0.5:
                band = "<=1%"
            elif threshold_hits["holdout_top5pct"] >= 0.5:
                band = "1-5%"
            elif threshold_hits["holdout_top10pct"] >= 0.5:
                band = "5-10%"
            else:
                band = ">10%"
            rows.append(
                {
                    "drug": src["drug"],
                    "n_holdout_pos": int(y_hold_b[i].sum().item()),
                    "best_rank": best_rank,
                    "best_percentile": best_percentile,
                    "best_percentile_pct": 100.0 * best_percentile,
                    "all_positive_ranks": ";".join(str(rank) for rank in ranks),
                    "threshold_interval": band,
                    **threshold_hits,
                }
            )
        offset += len(logits_np)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    numeric = derive_percentiles()
    reference = pd.read_csv(REFERENCE_PER_DRUG, sep="\t")
    reference = reference[reference["n_holdout_pos"] > 0].copy()
    merged = numeric.merge(
        reference[["drug", "n_holdout_pos", "holdout_top1pct", "holdout_top5pct", "holdout_top10pct"]],
        on="drug",
        suffixes=("", "_reference"),
        how="inner",
    )
    if len(merged) != 9:
        raise AssertionError(f"Expected 9 holdout drugs, found {len(merged)}")
    for col in ["n_holdout_pos", "holdout_top1pct", "holdout_top5pct", "holdout_top10pct"]:
        ref_col = f"{col}_reference"
        if not np.allclose(merged[col].astype(float), merged[ref_col].astype(float), equal_nan=True):
            bad = merged[["drug", col, ref_col]].to_dict(orient="records")
            raise AssertionError(f"Derived {col} does not match Round261 reference: {bad}")

    numeric = numeric.sort_values(["best_percentile", "drug"], ascending=[True, True])
    numeric.to_csv(OUT_TSV, sep="\t", index=False)
    audit = {
        "source": "Round261 fixed checkpoint inference",
        "checkpoint": str(CHECKPOINT),
        "proteomics_data": str(PROTEOMICS_DATA),
        "label_repr": str(LABEL_REPR),
        "n_holdout_drugs": int(len(numeric)),
        "n_train_drugs": int(load_locked_data()["n_train_drugs"]),
        "n_val_drugs": int(load_locked_data()["n_val_drugs"]),
        "n_test_drugs": int(load_locked_data()["n_test_drugs"]),
        "rank_universe": L_FULL,
        "topk_rule": "1-based best rank <= floor(universe * threshold), matching the Round261 torch.topk implementation",
        "reference_match": True,
    }
    OUT_AUDIT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(numeric.to_string(index=False))
    print(f"Saved {OUT_TSV}")


if __name__ == "__main__":
    main()
