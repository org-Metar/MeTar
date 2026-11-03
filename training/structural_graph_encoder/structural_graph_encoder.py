#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
LOCAL_WORKSPACE = ROOT.parent
SERVER_WORKSPACE = Path("/home/lwn/claude/manuscript/write")
WORKSPACE = SERVER_WORKSPACE if (SERVER_WORKSPACE / "claude_round93_landmark_target_relation_bridge_pc3_20260507").exists() else LOCAL_WORKSPACE
SOURCE = ROOT / "source_data"
QA = ROOT / "qa"
CHECKPOINTS = ROOT / "checkpoints"
LOGS = ROOT / "logs"

SEED = 20260507
CELL_LINE = "PC3"
HIT_KS = [1, 5, 10, 20, 30, 50]
META_COLUMNS = ["split", "profile_id", "drug", "cell_line", "n_profiles", "true_targets"]
META_SET = set(META_COLUMNS)

RESPONSE_PATH = WORKSPACE / "claude_round93_landmark_target_relation_bridge_pc3_20260507" / "source_data" / "pc3_landmark_response_978.tsv.gz"
ROUND92_SOURCE = WORKSPACE / "claude_round92_metar_fixed_graph_prior_pc3_20260507" / "source_data"
SCORE_LABEL_CANDIDATES = [
    ROUND92_SOURCE / "pc3_metar_drugcell_scores_668.tsv.gz",
    ROUND92_SOURCE / "pc3_metar_drugcell_scores_668_formal_export.tsv.gz",
    ROUND92_SOURCE / "pc3_metar_graph_prior_scores_668.tsv.gz",
]
PROCESS_PATH = WORKSPACE / "claude_round91_pertkge_inductive_single_cellline_20260507" / "cell_PC3" / "pertkge" / "knowledge_graph" / "process.txt"
ROUND174_SUMMARY_PATH = WORKSPACE / "claude_round174_joint_heterobridge_pc3_single_seed_20260508" / "source_data" / "pc3_metric_summary.tsv"
ROUND175_SUMMARY_PATH = WORKSPACE / "claude_round175_graph_signal_models_pc3_single_seed_20260508" / "source_data" / "pc3_metric_summary.tsv"
ROUND90_PROFILE_PATH = WORKSPACE / "claude_round90_pertkge_single_cellline_convergence_20260506" / "remote_metrics" / "metric_summary_PC3.tsv"
ROUND100_DRUGMEAN_PATH = WORKSPACE / "claude_round100_pertkge_prediction_time_effect_required_20260508" / "cell_PC3" / "metric_summary_drug_mean.tsv"

RELATION_TYPES = ["PPI", "Translates", "Transcribes", "TF_Regulates", "RBP_Regulates", "miRNA_Regulates", "lncRNA_Regulates", "LMI"]
MAX_HOPS = 3
FRONTIER_CAP = 512
EDGE_CAP_PER_REL = 96

MLP_HIDDEN = 128
GRAPH_HIDDEN = 64
TARGET_DIM = 64
RGCN_BASIS_DIM = 32
ATTN_HEADS = 2
DROPOUT = 0.15
MAX_EPOCHS = 120
MIN_EPOCHS_BEFORE_STOP = 40
PATIENCE = 16
BATCH_SIZE = 256
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-5
DNBCE_WEIGHT = 0.20
TARGET_BIAS_L2 = 1.0e-4
RELATION_REG_L2 = 1.0e-4
SCORE_L2 = 1.0e-5

VARIANTS = [
    "opt1_strong_base_tiny_graph",
    "opt2_scalar_relation_residual",
    "opt3_sample_gated_relation",
    "opt4_target_conditioned_relation",
    "opt5_consensus_graph_residual",
]


def ensure_dirs() -> None:
    for path in [SOURCE, QA, CHECKPOINTS, LOGS]:
        path.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    print(f"[round176] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_targets(value) -> tuple[str, ...]:
    text = str(value).strip()
    if text.startswith(("(", "[")):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return tuple(str(v).strip() for v in parsed if str(v).strip() and str(v).strip() not in {"0", "-666", "nan", "None"})
        except Exception:
            pass
    return tuple(g.strip() for g in text.split("|") if g.strip() and g.strip() not in {"0", "-666", "nan", "None"})


def zscore_by_train(x: np.ndarray, split: pd.Series) -> np.ndarray:
    out = x.astype(np.float32, copy=True)
    train_mask = split.astype(str).eq("inner_train").to_numpy()
    mu = out[train_mask].mean(axis=0, keepdims=True)
    sd = out[train_mask].std(axis=0, keepdims=True)
    sd[sd < 1.0e-6] = 1.0
    return (out - mu) / sd


def true_target_matrix(rows: pd.DataFrame, labels: list[str]) -> np.ndarray:
    idx = {g: i for i, g in enumerate(labels)}
    y = np.zeros((len(rows), len(labels)), dtype=np.float32)
    for row_i, value in enumerate(rows["true_targets"].astype(str)):
        for gene in split_targets(value):
            j = idx.get(gene)
            if j is not None:
                y[row_i, j] = 1.0
    return y


def load_response() -> tuple[list[str], list[str], pd.DataFrame, np.ndarray, np.ndarray]:
    log("loading PC3 landmark response and 668-label space")
    response = pd.read_csv(RESPONSE_PATH, sep="\t", compression="gzip")
    response = response[response["cell_line"].astype(str).eq(CELL_LINE)].copy().reset_index(drop=True)
    landmarks = [c for c in response.columns if c not in META_SET]
    if len(landmarks) != 978:
        raise RuntimeError(f"Expected 978 landmark response columns, observed {len(landmarks)}")

    score_label_path = next((path for path in SCORE_LABEL_CANDIDATES if path.exists()), None)
    if score_label_path is None:
        raise FileNotFoundError(f"No 668-label score table found in candidates: {[str(p) for p in SCORE_LABEL_CANDIDATES]}")
    score_head = pd.read_csv(score_label_path, sep="\t", compression="gzip", nrows=1)
    labels = [c for c in score_head.columns if c not in META_SET]
    if len(labels) != 668:
        raise RuntimeError(f"Expected 668 target labels, observed {len(labels)}")

    rows = response[META_COLUMNS].copy()
    x = zscore_by_train(response[landmarks].to_numpy(dtype=np.float32), rows["split"])
    y = true_target_matrix(rows, labels)
    if not (y.sum(axis=1) > 0).all():
        raise RuntimeError("Non-evaluable rows found after loading PC3 response")
    return landmarks, labels, rows, x, y


def parse_process_graph() -> tuple[dict[str, dict[str, set[str]]], pd.DataFrame, pd.DataFrame, dict]:
    log("parsing PertKGE process graph")
    adjacency: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    relation_counts = Counter()
    prefix_counts = Counter()
    forbidden_has_binding = 0
    cid_edges = 0
    with PROCESS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            h, rel, t = parts
            relation_counts[rel] += 1
            hp = h.split(":", 1)[0] if ":" in h else h
            tp = t.split(":", 1)[0] if ":" in t else t
            prefix_counts[(rel, hp, tp)] += 1
            if rel == "HAS_BINDING_TO":
                forbidden_has_binding += 1
            if h.startswith("CID:") or t.startswith("CID:"):
                cid_edges += 1
            if rel not in RELATION_TYPES:
                continue
            adjacency[h][rel].add(t)
            if rel == "PPI":
                adjacency[t][rel].add(h)
            else:
                adjacency[t][f"{rel}__rev"].add(h)
    rel_df = pd.DataFrame(
        [{"relation": rel, "n_edges": int(n), "used_by_round176": rel in RELATION_TYPES} for rel, n in sorted(relation_counts.items())]
    )
    prefix_df = pd.DataFrame(
        [{"relation": r, "head_prefix": hp, "tail_prefix": tp, "n_edges": int(n)} for (r, hp, tp), n in prefix_counts.most_common()]
    )
    audit = {
        "forbidden_has_binding_edges_in_process": int(forbidden_has_binding),
        "cid_edges_in_process": int(cid_edges),
        "n_nodes_with_out_edges": int(len(adjacency)),
        "n_relation_types_raw": int(len(relation_counts)),
        "relation_types_used": RELATION_TYPES,
    }
    return adjacency, rel_df, prefix_df, audit


def cap_neighbors(items: set[str], cap: int = EDGE_CAP_PER_REL) -> list[str]:
    if len(items) <= cap:
        return sorted(items)
    return sorted(items)[:cap]


def build_typed_channel_matrix(landmarks: list[str], labels: list[str], adjacency: dict[str, dict[str, set[str]]]) -> tuple[np.ndarray, pd.DataFrame]:
    log("building typed landmark-target graph operators")
    target_nodes = {}
    for j, gene in enumerate(labels):
        for prefix in ("Protein", "TF", "RBP"):
            target_nodes[f"{prefix}:{gene}"] = j

    names = []
    matrices = []
    for rel in RELATION_TYPES:
        names.append(rel)
        matrices.append(np.zeros((len(landmarks), len(labels)), dtype=np.float32))
        if rel != "PPI":
            names.append(f"{rel}__rev")
            matrices.append(np.zeros((len(landmarks), len(labels)), dtype=np.float32))
    channel_index = {name: i for i, name in enumerate(names)}
    channel_mat = np.stack(matrices, axis=0)

    label_idx = {g: i for i, g in enumerate(labels)}
    same_gene = np.zeros((len(landmarks), len(labels)), dtype=np.float32)
    for i, gene in enumerate(landmarks):
        if gene in label_idx:
            same_gene[i, label_idx[gene]] = 1.0

    start_templates = ["mRNA:{gene}", "DNA:{gene}", "Readout:{gene}"]
    for li, gene in enumerate(landmarks):
        start_nodes = {template.format(gene=gene) for template in start_templates}
        visited = set(start_nodes)
        queue = deque((node, 0, 1.0, tuple()) for node in start_nodes)
        while queue:
            node, depth, score, rel_trace = queue.popleft()
            if depth >= MAX_HOPS:
                continue
            for rel_name, nbrs in adjacency.get(node, {}).items():
                if rel_name not in channel_index:
                    continue
                capped = cap_neighbors(nbrs)
                if not capped:
                    continue
                step_weight = score * (0.70 ** (depth + 1)) / math.sqrt(max(1, len(capped)))
                next_trace = rel_trace + (rel_name,)
                for nbr in capped:
                    tj = target_nodes.get(nbr)
                    if tj is not None:
                        unique_trace = tuple(dict.fromkeys(next_trace))
                        trace_weight = step_weight / math.sqrt(max(1, len(unique_trace)))
                        for trace_rel in unique_trace:
                            channel_mat[channel_index[trace_rel], li, tj] += trace_weight
                    if nbr not in visited and len(visited) < FRONTIER_CAP:
                        visited.add(nbr)
                        queue.append((nbr, depth + 1, step_weight, next_trace))

    channel_mat = np.concatenate([same_gene[None, :, :], channel_mat], axis=0)
    names = ["same_gene"] + names
    rows = []
    for ci, name in enumerate(names):
        mat = channel_mat[ci]
        rows.append(
            {
                "channel": name,
                "nonzero_edges": int((mat != 0).sum()),
                "covered_landmarks": int((np.abs(mat).sum(axis=1) > 0).sum()),
                "covered_targets": int((np.abs(mat).sum(axis=0) > 0).sum()),
                "mean_abs_nonzero": float(np.abs(mat[mat != 0]).mean()) if (mat != 0).any() else 0.0,
            }
        )
        denom = np.abs(mat).sum(axis=1, keepdims=True)
        denom[denom < 1.0e-6] = 1.0
        channel_mat[ci] = mat / denom
    return channel_mat.astype(np.float32), pd.DataFrame(rows)


def load_or_build_channels(landmarks: list[str], labels: list[str]) -> np.ndarray:
    cache_path = SOURCE / "landmark_target_graph_operators_978x668x16.npz"
    audit_path = SOURCE / "heterograph_node_edge_audit.tsv"
    rel_path = QA / "process_relation_counts.tsv"
    prefix_path = QA / "process_prefix_relation_counts_top200.tsv"
    if cache_path.exists() and audit_path.exists():
        log(f"loading cached graph operators: {cache_path}")
        return np.load(cache_path)["channel"].astype(np.float32)

    prior_cache = WORKSPACE / "claude_round174_joint_heterobridge_pc3_single_seed_20260508" / "source_data" / "landmark_target_relation_channels_978x668x16.npz"
    prior_audit = WORKSPACE / "claude_round174_joint_heterobridge_pc3_single_seed_20260508" / "source_data" / "heterograph_node_edge_audit.tsv"
    if prior_cache.exists() and prior_audit.exists():
        log(f"loading Round174 graph operator cache: {prior_cache}")
        channel = np.load(prior_cache)["channel"].astype(np.float32)
        pd.read_csv(prior_audit, sep="\t").to_csv(audit_path, sep="\t", index=False)
        return channel

    adjacency, rel_df, prefix_df, graph_audit = parse_process_graph()
    channel, audit = build_typed_channel_matrix(landmarks, labels, adjacency)
    np.savez_compressed(cache_path, channel=channel)
    audit.to_csv(audit_path, sep="\t", index=False)
    rel_df.to_csv(rel_path, sep="\t", index=False)
    prefix_df.head(200).to_csv(prefix_path, sep="\t", index=False)
    (QA / "graph_parse_audit.json").write_text(json.dumps(graph_audit, indent=2), encoding="utf-8")
    return channel


def row_weights(rows: pd.DataFrame) -> np.ndarray:
    w = np.log1p(np.minimum(rows["n_profiles"].to_numpy(dtype=np.float32), 50.0))
    w = w / max(float(w.mean()), 1.0e-8)
    return w.astype(np.float32)


def bag_softmax_loss(logits: torch.Tensor, y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(y <= 0, -1.0e9)
    pos_logsum = torch.logsumexp(masked, dim=1)
    all_logsum = torch.logsumexp(logits, dim=1)
    return ((all_logsum - pos_logsum) * weights).mean()


def drug_normalized_bce_loss(logits: torch.Tensor, y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    pos = y > 0
    neg = ~pos
    pos_loss = (loss * pos.float()).sum(dim=1) / torch.clamp(pos.float().sum(dim=1), min=1.0)
    neg_loss = (loss * neg.float()).sum(dim=1) / torch.clamp(neg.float().sum(dim=1), min=1.0)
    return ((pos_loss + neg_loss) * weights).mean()


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = y_true.astype(np.int8)
    n_pos = int(y.sum())
    if n_pos <= 0:
        return 0.0
    order = np.argsort(scores)[::-1]
    hits = y[order] > 0
    cum_hits = np.cumsum(hits)
    ranks = np.arange(1, len(y) + 1)
    return float((cum_hits[hits] / ranks[hits]).sum() / n_pos)


def ndcg_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    y = y_true.astype(np.float64)
    if y.sum() <= 0:
        return 0.0
    kk = min(k, len(y))
    order = np.argsort(scores)[::-1][:kk]
    discounts = 1.0 / np.log2(np.arange(2, kk + 2, dtype=np.float64))
    dcg = float((y[order] * discounts).sum())
    ideal = np.sort(y)[::-1][:kk]
    idcg = float((ideal * discounts).sum())
    return 0.0 if idcg <= 0 else dcg / idcg


def binary_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = y_true.astype(np.int8)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos <= 0 or n_neg <= 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    pos_rank_sum = float(ranks[y > 0].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def compute_metrics(scores: np.ndarray, y: np.ndarray, rows: pd.DataFrame, method: str, variant: str, unit: str) -> tuple[dict, pd.DataFrame]:
    records = []
    for i in range(scores.shape[0]):
        s = np.nan_to_num(scores[i].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        yy = y[i].astype(np.int8)
        order = np.argsort(s)[::-1]
        pos = set(np.where(yy > 0)[0].tolist())
        rec = {
            "method": method,
            "variant": variant,
            "split": rows.iloc[i]["split"],
            "evaluation_unit": unit,
            "drug": rows.iloc[i]["drug"],
            "cell_line": rows.iloc[i]["cell_line"],
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
        "split": rows["split"].iloc[0],
        "evaluation_unit": unit,
        "n_groups": int(len(rows)),
        "n_drugs": int(rows["drug"].nunique()),
        "n_profiles": int(rows["n_profiles"].sum()),
    }
    for metric in [f"Hit@{k}" for k in HIT_KS] + ["AUROC", "NDCG@10", "MeanAP"]:
        summary[metric] = float(per[metric].mean())
    return summary, per


class MLPOnlyReference(torch.nn.Module):
    def __init__(self, n_features: int, n_targets: int):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(n_features, MLP_HIDDEN),
            torch.nn.LayerNorm(MLP_HIDDEN),
            torch.nn.GELU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(MLP_HIDDEN, MLP_HIDDEN),
            torch.nn.LayerNorm(MLP_HIDDEN),
            torch.nn.GELU(),
        )
        self.sample_proj = torch.nn.Linear(MLP_HIDDEN, TARGET_DIM)
        self.target_emb = torch.nn.Parameter(torch.randn(n_targets, TARGET_DIM) * 0.02)
        self.target_bias = torch.nn.Parameter(torch.zeros(n_targets))
        self.base_scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        q = F.normalize(self.sample_proj(h), dim=1)
        t = F.normalize(self.target_emb, dim=1)
        return torch.matmul(q, t.t()) * torch.exp(torch.clamp(self.base_scale, -2.0, 3.0)) + self.target_bias[None, :]

    def regularization(self) -> torch.Tensor:
        return TARGET_BIAS_L2 * torch.square(self.target_bias).sum() + SCORE_L2 * torch.square(self.target_emb).sum()

    def diagnostics(self) -> pd.DataFrame:
        return pd.DataFrame([{"name": "target_bias_l2", "value": float(torch.square(self.target_bias.detach()).sum().cpu())}])


class GraphSignalBase(torch.nn.Module):
    def __init__(self, channel: np.ndarray, n_features: int, n_targets: int):
        super().__init__()
        self.n_targets = n_targets
        self.register_buffer("channel", torch.tensor(channel, dtype=torch.float32))
        self.sample_encoder = torch.nn.Sequential(
            torch.nn.Linear(n_features, GRAPH_HIDDEN),
            torch.nn.LayerNorm(GRAPH_HIDDEN),
            torch.nn.GELU(),
            torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(GRAPH_HIDDEN, GRAPH_HIDDEN),
            torch.nn.LayerNorm(GRAPH_HIDDEN),
            torch.nn.GELU(),
        )
        self.sample_proj = torch.nn.Linear(GRAPH_HIDDEN, TARGET_DIM)
        self.target_emb = torch.nn.Parameter(torch.randn(n_targets, TARGET_DIM) * 0.02)
        self.target_bias = torch.nn.Parameter(torch.zeros(n_targets))
        self.base_scale = torch.nn.Parameter(torch.tensor(0.5))
        self.graph_scale = torch.nn.Parameter(torch.tensor(-0.5))

    def base_score(self, x: torch.Tensor) -> torch.Tensor:
        h = self.sample_encoder(x)
        q = F.normalize(self.sample_proj(h), dim=1)
        t = F.normalize(self.target_emb, dim=1)
        return torch.matmul(q, t.t()) * torch.exp(torch.clamp(self.base_scale, -2.0, 3.0))

    def relation_signal(self, x: torch.Tensor) -> torch.Tensor:
        feats = torch.stack([x, torch.relu(x), torch.relu(-x), torch.abs(x)], dim=-1)
        rel = torch.einsum("blf,clt->btcf", feats, self.channel)
        mu = rel.mean(dim=(1, 2), keepdim=True)
        sd = torch.clamp(rel.std(dim=(1, 2), keepdim=True), min=1.0e-6)
        return (rel - mu) / sd

    def common_regularization(self) -> torch.Tensor:
        return TARGET_BIAS_L2 * torch.square(self.target_bias).sum() + SCORE_L2 * torch.square(self.target_emb).sum()


class RGCNTargetReadout(GraphSignalBase):
    def __init__(self, channel: np.ndarray, n_features: int, n_targets: int):
        super().__init__(channel, n_features, n_targets)
        n_channels = channel.shape[0]
        self.relation_gate = torch.nn.Parameter(torch.zeros(n_channels))
        self.basis = torch.nn.Sequential(
            torch.nn.Linear(n_channels * 4, RGCN_BASIS_DIM),
            torch.nn.LayerNorm(RGCN_BASIS_DIM),
            torch.nn.GELU(),
            torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(RGCN_BASIS_DIM, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_score(x)
        rel = self.relation_signal(x)
        gate = torch.softmax(self.relation_gate, dim=0)[None, None, :, None]
        rel = (rel * gate).reshape(x.shape[0] * self.n_targets, -1)
        graph_score = self.basis(rel).reshape(x.shape[0], self.n_targets)
        return base + torch.exp(torch.clamp(self.graph_scale, -4.0, 2.0)) * graph_score + self.target_bias[None, :]

    def regularization(self) -> torch.Tensor:
        return self.common_regularization() + RELATION_REG_L2 * torch.square(self.relation_gate).sum()

    def diagnostics(self) -> pd.DataFrame:
        gate = torch.softmax(self.relation_gate.detach(), dim=0).cpu().numpy()
        return pd.DataFrame({"name": [f"relation_gate_{i}" for i in range(len(gate))], "value": gate})


class HGTLiteTargetAttention(GraphSignalBase):
    def __init__(self, channel: np.ndarray, n_features: int, n_targets: int):
        super().__init__(channel, n_features, n_targets)
        n_channels = channel.shape[0]
        self.sample_context = torch.nn.Linear(GRAPH_HIDDEN, TARGET_DIM)
        self.rel_key = torch.nn.Parameter(torch.randn(n_channels, TARGET_DIM) * 0.02)
        self.rel_bias = torch.nn.Parameter(torch.zeros(n_channels))
        self.value_mlp = torch.nn.Sequential(
            torch.nn.Linear(4, TARGET_DIM),
            torch.nn.GELU(),
            torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(TARGET_DIM, TARGET_DIM),
        )
        self.readout = torch.nn.Linear(TARGET_DIM, 1)
        self.n_heads = ATTN_HEADS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.sample_encoder(x)
        q_sample = self.sample_proj(h)
        q = F.normalize(q_sample, dim=1)
        t = F.normalize(self.target_emb, dim=1)
        base = torch.matmul(q, t.t()) * torch.exp(torch.clamp(self.base_scale, -2.0, 3.0))

        rel = self.relation_signal(x)
        values = self.value_mlp(rel)
        query = F.normalize(t[None, :, :] + 0.25 * self.sample_context(h)[:, None, :], dim=-1)
        keys = F.normalize(self.rel_key, dim=1)
        attn_logits = torch.einsum("btd,cd->btc", query, keys) / math.sqrt(TARGET_DIM)
        attn_logits = attn_logits + self.rel_bias[None, None, :]
        attn = torch.softmax(attn_logits, dim=2)
        context = torch.einsum("btc,btcd->btd", attn, values)
        graph_score = self.readout(context).squeeze(-1)
        return base + torch.exp(torch.clamp(self.graph_scale, -4.0, 2.0)) * graph_score + self.target_bias[None, :]

    def regularization(self) -> torch.Tensor:
        return self.common_regularization() + RELATION_REG_L2 * (torch.square(self.rel_key).sum() + torch.square(self.rel_bias).sum())

    def diagnostics(self) -> pd.DataFrame:
        bias = self.rel_bias.detach().cpu().numpy()
        return pd.DataFrame({"name": [f"attention_relation_bias_{i}" for i in range(len(bias))], "value": bias})


class StrongBaseGraphResidual(torch.nn.Module):
    def __init__(self, variant: str, channel: np.ndarray, n_features: int, n_targets: int):
        super().__init__()
        self.variant = variant
        self.n_targets = n_targets
        self.register_buffer("channel", torch.tensor(channel, dtype=torch.float32))
        n_channels = channel.shape[0]
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(n_features, MLP_HIDDEN),
            torch.nn.LayerNorm(MLP_HIDDEN),
            torch.nn.GELU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(MLP_HIDDEN, MLP_HIDDEN),
            torch.nn.LayerNorm(MLP_HIDDEN),
            torch.nn.GELU(),
        )
        self.sample_proj = torch.nn.Linear(MLP_HIDDEN, TARGET_DIM)
        self.target_emb = torch.nn.Parameter(torch.randn(n_targets, TARGET_DIM) * 0.02)
        self.target_bias = torch.nn.Parameter(torch.zeros(n_targets))
        self.base_scale = torch.nn.Parameter(torch.tensor(1.0))
        # Start the graph residual small; Round175 showed that a large graph
        # branch can hurt ranking before it learns useful relation weights.
        self.graph_scale = torch.nn.Parameter(torch.tensor(-2.25))
        self.relation_logits = torch.nn.Parameter(torch.zeros(n_channels))

        self.scalar_weight = torch.nn.Parameter(torch.zeros(n_channels, 4))
        self.scalar_bias = torch.nn.Parameter(torch.zeros(1))
        self.sample_gate = torch.nn.Sequential(
            torch.nn.Linear(MLP_HIDDEN, 32),
            torch.nn.GELU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(32, n_channels),
        )
        self.target_channel = torch.nn.Linear(TARGET_DIM, n_channels)
        self.target_feature = torch.nn.Linear(TARGET_DIM, 4)
        self.consensus_weight = torch.nn.Parameter(torch.zeros(n_channels))
        self.consensus_mlp = torch.nn.Sequential(
            torch.nn.Linear(3, 16),
            torch.nn.GELU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(16, 1),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        q = F.normalize(self.sample_proj(h), dim=1)
        t = F.normalize(self.target_emb, dim=1)
        base = torch.matmul(q, t.t()) * torch.exp(torch.clamp(self.base_scale, -2.0, 3.0))
        return h, t, base

    def relation_signal(self, x: torch.Tensor) -> torch.Tensor:
        feats = torch.stack([x, torch.relu(x), torch.relu(-x), torch.abs(x)], dim=-1)
        rel = torch.einsum("blf,clt->btcf", feats, self.channel)
        mu = rel.mean(dim=(1, 2), keepdim=True)
        sd = torch.clamp(rel.std(dim=(1, 2), keepdim=True), min=1.0e-6)
        return (rel - mu) / sd

    def scalar_residual(self, rel: torch.Tensor) -> torch.Tensor:
        gate = torch.softmax(self.relation_logits, dim=0)
        weight = self.scalar_weight * gate[:, None]
        return torch.einsum("btcf,cf->bt", rel, weight) + self.scalar_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, target, base = self.encode(x)
        rel = self.relation_signal(x)
        graph = self.scalar_residual(rel)

        if self.variant == "opt1_strong_base_tiny_graph":
            # Small scalar residual only: tests whether a protected base model
            # can tolerate a graph signal without overfitting.
            pass
        elif self.variant == "opt2_scalar_relation_residual":
            graph = graph + torch.einsum("btcf,cf->bt", rel, torch.tanh(self.scalar_weight))
        elif self.variant == "opt3_sample_gated_relation":
            sample_gate = torch.sigmoid(self.sample_gate(h))
            graph = torch.einsum("btcf,bc,cf->bt", rel, sample_gate, torch.tanh(self.scalar_weight))
        elif self.variant == "opt4_target_conditioned_relation":
            channel_gate = torch.softmax(self.target_channel(target), dim=1)
            feature_gate = torch.tanh(self.target_feature(target))
            graph = torch.einsum("btcf,tc,tf->bt", rel, channel_gate, feature_gate)
        elif self.variant == "opt5_consensus_graph_residual":
            channel_gate = torch.softmax(self.consensus_weight, dim=0)
            signed = torch.einsum("btcf,c->btf", rel, channel_gate)
            consensus = torch.stack(
                [
                    signed[..., 0],
                    torch.maximum(signed[..., 1], signed[..., 2]),
                    signed[..., 3],
                ],
                dim=-1,
            )
            graph = self.consensus_mlp(consensus).squeeze(-1)
        else:
            raise RuntimeError(f"Unknown Round176 variant: {self.variant}")

        return base + torch.exp(torch.clamp(self.graph_scale, -5.0, 1.0)) * graph + self.target_bias[None, :]

    def regularization(self) -> torch.Tensor:
        reg = TARGET_BIAS_L2 * torch.square(self.target_bias).sum()
        reg = reg + SCORE_L2 * (torch.square(self.target_emb).sum() + torch.square(self.scalar_weight).sum())
        reg = reg + RELATION_REG_L2 * (torch.square(self.relation_logits).sum() + torch.square(self.consensus_weight).sum())
        return reg

    def diagnostics(self) -> pd.DataFrame:
        rows = [{"name": "graph_scale_exp", "value": float(torch.exp(torch.clamp(self.graph_scale.detach(), -5.0, 1.0)).cpu())}]
        gate = torch.softmax(self.relation_logits.detach(), dim=0).cpu().numpy()
        rows += [{"name": f"relation_gate_{i}", "value": float(v)} for i, v in enumerate(gate)]
        return pd.DataFrame(rows)


def make_model(variant: str, channel: np.ndarray | None, n_features: int, n_targets: int) -> torch.nn.Module:
    if channel is None:
        raise RuntimeError(f"{variant} requires graph operators")
    if variant in VARIANTS:
        return StrongBaseGraphResidual(variant, channel, n_features, n_targets)
    raise RuntimeError(f"Unknown variant: {variant}")


def split_data(rows: pd.DataFrame, x: np.ndarray, y: np.ndarray) -> dict[str, dict]:
    out = {}
    for split in ["inner_train", "val", "test"]:
        mask = rows["split"].astype(str).eq(split).to_numpy()
        split_rows = rows[mask].reset_index(drop=True)
        out[split] = {
            "rows": split_rows,
            "x": x[mask],
            "y": y[mask],
            "weights": row_weights(split_rows),
        }
    return out


def better_val(candidate: dict, best: dict | None) -> bool:
    if best is None:
        return True
    for key in ["Hit@10", "Hit@5", "MeanAP", "NDCG@10"]:
        diff = candidate[key] - best[key]
        if abs(diff) > 1.0e-12:
            return diff > 0.0
    return False


def train_variant(variant: str, data: dict[str, dict], channel: np.ndarray | None, device: torch.device) -> tuple[torch.nn.Module, list[dict], dict]:
    n_features = data["inner_train"]["x"].shape[1]
    n_targets = data["inner_train"]["y"].shape[1]
    model = make_model(variant, channel, n_features, n_targets).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_state = None
    best_val = None
    best_epoch = 0
    stale = 0
    history = []
    rng = np.random.default_rng(SEED)

    tensors = {}
    for split, item in data.items():
        tensors[split] = {
            "x": torch.tensor(item["x"], dtype=torch.float32, device=device),
            "y": torch.tensor(item["y"], dtype=torch.float32, device=device),
            "w": torch.tensor(item["weights"], dtype=torch.float32, device=device),
        }

    n_train = tensors["inner_train"]["x"].shape[0]
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = rng.permutation(n_train)
        losses = []
        for start in range(0, n_train, BATCH_SIZE):
            idx_np = order[start:start + BATCH_SIZE]
            idx = torch.tensor(idx_np, dtype=torch.long, device=device)
            xb = tensors["inner_train"]["x"][idx]
            yb = tensors["inner_train"]["y"][idx]
            wb = tensors["inner_train"]["w"][idx]
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = bag_softmax_loss(logits, yb, wb) + DNBCE_WEIGHT * drug_normalized_bce_loss(logits, yb, wb) + model.regularization()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_scores = model(tensors["val"]["x"]).detach().cpu().numpy()
            train_scores = model(tensors["inner_train"]["x"]).detach().cpu().numpy()
        val_summary, _ = compute_metrics(val_scores, data["val"]["y"], data["val"]["rows"], "Round176", variant, "drug-mean")
        train_summary, _ = compute_metrics(train_scores, data["inner_train"]["y"], data["inner_train"]["rows"], "Round176", variant, "drug-mean")
        rec = {
            "variant": variant,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_Hit@10": train_summary["Hit@10"],
            "train_MeanAP": train_summary["MeanAP"],
            **{f"val_{k}": v for k, v in val_summary.items() if k in {"Hit@5", "Hit@10", "Hit@20", "NDCG@10", "MeanAP"}},
        }
        history.append(rec)
        if epoch == 1 or epoch % 10 == 0:
            log(
                f"{variant} epoch={epoch} loss={rec['train_loss']:.4f} "
                f"train_Hit@10={rec['train_Hit@10']:.4f} val_Hit@10={rec['val_Hit@10']:.4f} val_MeanAP={rec['val_MeanAP']:.4f}"
            )
        if better_val(val_summary, best_val):
            best_val = val_summary
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch >= MIN_EPOCHS_BEFORE_STOP and stale >= PATIENCE:
            break

    if best_state is None or best_val is None:
        raise RuntimeError(f"No checkpoint selected for {variant}")
    model.load_state_dict(best_state)
    return model, history, {"variant": variant, "best_epoch": best_epoch, "best_val": best_val}


def predict(model: torch.nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            out.append(model(xb).detach().cpu().numpy())
    return np.vstack(out)


def read_reference_rows() -> pd.DataFrame:
    rows = []
    if ROUND174_SUMMARY_PATH.exists():
        df = pd.read_csv(ROUND174_SUMMARY_PATH, sep="\t")
        keep = df[(df["split"].eq("test")) & (df["variant"].isin(["mlp_only_retrain"]))].copy()
        keep["method"] = "Round174"
        keep["variant"] = "mlp_only_retrain"
        keep["evaluation_unit"] = "drug-mean"
        rows.append(keep)
    if ROUND175_SUMMARY_PATH.exists():
        df = pd.read_csv(ROUND175_SUMMARY_PATH, sep="\t")
        keep = df[(df["method"].eq("Round175")) & (df["split"].eq("test")) & (df["variant"].isin(["hgtlite_target_attention"]))].copy()
        keep["evaluation_unit"] = "drug-mean"
        rows.append(keep)
    if ROUND90_PROFILE_PATH.exists():
        df = pd.read_csv(ROUND90_PROFILE_PATH, sep="\t")
        keep = df[(df["cell_line"].eq("PC3")) & (df["method"].eq("PertKGE"))].copy()
        keep["variant"] = "round90_300epoch_converged"
        keep["split"] = "test"
        keep["evaluation_unit"] = "profile-level"
        keep["n_groups"] = keep["n_profiles"]
        rows.append(keep)
    if ROUND100_DRUGMEAN_PATH.exists():
        df = pd.read_csv(ROUND100_DRUGMEAN_PATH, sep="\t")
        keep = df[(df["cell_line"].eq("PC3")) & (df["method"].eq("PertKGE"))].copy()
        keep["variant"] = "round100_drug_not_trained"
        keep["split"] = "test"
        keep["evaluation_unit"] = "drug-mean"
        keep["n_groups"] = keep["n_profiles"]
        rows.append(keep)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True, sort=False)
    for col in ["best_epoch", "best_true_target_percentile", "setting"]:
        if col in out.columns:
            out = out.drop(columns=[col])
    return out


def markdown_table(df: pd.DataFrame, digits: int = 4) -> str:
    if df.empty:
        return "_No rows._"
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: f"{x:.{digits}f}")
    out = out.fillna("").astype(str)
    lines = ["| " + " | ".join(out.columns) + " |", "| " + " | ".join(["---"] * len(out.columns)) + " |"]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(row[c] for c in out.columns) + " |")
    return "\n".join(lines)


def run_one_variant(variant: str, device_arg: str) -> None:
    ensure_dirs()
    if variant not in VARIANTS:
        raise RuntimeError(f"Unknown variant: {variant}")
    set_seed(SEED)
    use_cuda = torch.cuda.is_available() and device_arg.startswith("cuda")
    device = torch.device(device_arg if use_cuda else "cpu")
    log(f"variant={variant} device={device}")

    landmarks, labels, rows, x, y = load_response()
    channel = None
    channel = load_or_build_channels(landmarks, labels)
    if channel.shape != (16, 978, 668):
        raise RuntimeError(f"Unexpected graph operator shape: {channel.shape}")
    data = split_data(rows, x, y)

    split_counts = {
        split: {
            "n_groups": int(len(item["rows"])),
            "n_drugs": int(item["rows"]["drug"].nunique()),
            "n_profiles": int(item["rows"]["n_profiles"].sum()),
        }
        for split, item in data.items()
    }
    if split_counts["inner_train"]["n_groups"] != 1906 or split_counts["val"]["n_groups"] != 172 or split_counts["test"]["n_groups"] != 465:
        raise RuntimeError(f"Unexpected split counts: {split_counts}")

    model, history, selected = train_variant(variant, data, channel, device)
    pd.DataFrame(history).to_csv(SOURCE / f"training_history_{variant}.tsv", sep="\t", index=False)

    summaries = []
    per_rows = []
    for split in ["inner_train", "val", "test"]:
        scores = predict(model, data[split]["x"], device)
        summ, per = compute_metrics(scores, data[split]["y"], data[split]["rows"], "Round176", variant, "drug-mean")
        summ["best_epoch"] = selected["best_epoch"]
        summaries.append(summ)
        per_rows.append(per)

    summary_df = pd.DataFrame(summaries)
    per_df = pd.concat(per_rows, ignore_index=True)
    summary_df.to_csv(SOURCE / f"pc3_metric_summary_{variant}.tsv", sep="\t", index=False)
    per_df.to_csv(SOURCE / f"pc3_per_drug_metrics_{variant}.tsv", sep="\t", index=False)
    if hasattr(model, "diagnostics"):
        model.diagnostics().to_csv(SOURCE / f"relation_or_attention_weights_{variant}.tsv", sep="\t", index=False)
    torch.save({"variant": variant, "state_dict": model.state_dict(), "selected": selected, "labels": labels, "landmarks": landmarks}, CHECKPOINTS / f"checkpoint_{variant}.pth")

    lock = {
        "round": "round176_hgtlite_structural_5x_pc3",
        "variant": variant,
        "status": "completed",
        "cell_line": CELL_LINE,
        "seed": SEED,
        "candidate_space": "668 train-label target space",
        "evaluation_unit": "drug-mean",
        "selection": "validation Hit@10; tie Hit@5, MeanAP, NDCG@10",
        "split_counts": split_counts,
        "input_paths": {
            "response": str(RESPONSE_PATH),
            "score_label_source": str(next(path for path in SCORE_LABEL_CANDIDATES if path.exists())),
            "process_graph": str(PROCESS_PATH),
        },
        "hashes": {
            "response_sha256": sha256_file(RESPONSE_PATH),
            "score_label_source_sha256": sha256_file(next(path for path in SCORE_LABEL_CANDIDATES if path.exists())),
            "process_sha256": sha256_file(PROCESS_PATH),
        },
        "selected": {"variant": variant, "best_epoch": selected["best_epoch"], "best_val": selected["best_val"]},
        "hyperparameters": {
            "mlp_hidden": MLP_HIDDEN,
            "graph_hidden": GRAPH_HIDDEN,
            "target_dim": TARGET_DIM,
            "rgcn_basis_dim": RGCN_BASIS_DIM,
            "attention_heads": ATTN_HEADS,
            "dropout": DROPOUT,
            "max_epochs": MAX_EPOCHS,
            "min_epochs_before_stop": MIN_EPOCHS_BEFORE_STOP,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "dnbce_weight": DNBCE_WEIGHT,
        },
        "forbidden": [
            "No HAS_BINDING_TO drug-target edges in graph construction",
            "No CID drug nodes in graph construction",
            "No validation/test labels used for training or graph construction",
            "No frozen MeTar score fusion as primary route",
        ],
    }
    (ROOT / f"method_lock_{variant}.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")


def write_leakage_audit() -> None:
    audit_path = SOURCE / "heterograph_node_edge_audit.tsv"
    lines = [
        "# Round176 leakage and graph audit",
        "",
        "- Graph source: PertKGE process graph.",
        "- `HAS_BINDING_TO` and `CID:` drug nodes are not used by the graph operator parser.",
        "- Train labels enter only the supervised loss.",
        "- Validation/test labels are used only for checkpoint selection/evaluation.",
        "- Candidate space is fixed to 668 labels from the Round92 score table header.",
        "- Main Round176 evaluation unit is drug-mean with 465 PC3 test groups.",
        "",
    ]
    if audit_path.exists():
        audit = pd.read_csv(audit_path, sep="\t")
        lines += ["## Graph operator coverage", "", markdown_table(audit)]
    (QA / "leakage_and_graph_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_optimization_lessons() -> None:
    lines = [
        "# Round176 structural optimization lessons",
        "",
        "Starting point: Round175 `hgtlite_target_attention` was the better graph-prior branch by test Hit@10, but it underperformed the Round174 MLP baseline. The main lesson was that the graph branch should be a controlled residual on top of a strong sample encoder, not a competing weak encoder.",
        "",
        "## opt1_strong_base_tiny_graph",
        "",
        "- Change: restore the stronger Round174-style MLP base and add a very small graph residual.",
        "- Lesson tested: graph prior must first avoid hurting a strong expression encoder.",
        "",
        "## opt2_scalar_relation_residual",
        "",
        "- Change: use a direct scalar relation residual over typed graph operators.",
        "- Lesson tested: a low-capacity typed residual may capture useful prior without attention overfitting.",
        "",
        "## opt3_sample_gated_relation",
        "",
        "- Change: condition relation weights on each drug response sample.",
        "- Lesson tested: different drug states may need different relation types, but sample gating can overfit if the signal is weak.",
        "",
        "## opt4_target_conditioned_relation",
        "",
        "- Change: condition relation and feature weights on the candidate target embedding.",
        "- Lesson tested: target-specific relation routing may better model input-gene to target-gene structure.",
        "",
        "## opt5_consensus_graph_residual",
        "",
        "- Change: reduce graph evidence to signed/strong/absolute consensus statistics before a small readout.",
        "- Lesson tested: compressing graph evidence may regularize noisy heterogeneous relations.",
    ]
    (QA / "optimization_lessons.md").write_text("\n".join(lines) + "\n", encoding="ascii")


def summarize_all() -> None:
    ensure_dirs()
    frames = []
    for variant in VARIANTS:
        path = SOURCE / f"pc3_metric_summary_{variant}.tsv"
        if path.exists():
            frames.append(pd.read_csv(path, sep="\t"))
    if not frames:
        raise RuntimeError("No Round176 variant summary files found")
    combined = pd.concat(frames, ignore_index=True)
    ref = read_reference_rows()
    if not ref.empty:
        combined = pd.concat([ref, combined], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=["method", "variant", "split", "evaluation_unit"], keep="last")
    combined.to_csv(SOURCE / "pc3_metric_summary.tsv", sep="\t", index=False)

    per_frames = []
    for variant in VARIANTS:
        path = SOURCE / f"pc3_per_drug_metrics_{variant}.tsv"
        if path.exists():
            per_frames.append(pd.read_csv(path, sep="\t"))
    if per_frames:
        pd.concat(per_frames, ignore_index=True).to_csv(SOURCE / "pc3_per_drug_metrics.tsv", sep="\t", index=False)

    histories = []
    for variant in VARIANTS:
        path = SOURCE / f"training_history_{variant}.tsv"
        if path.exists():
            histories.append(pd.read_csv(path, sep="\t"))
    if histories:
        pd.concat(histories, ignore_index=True).to_csv(SOURCE / "training_history.tsv", sep="\t", index=False)

    weight_frames = []
    for variant in VARIANTS:
        path = SOURCE / f"relation_or_attention_weights_{variant}.tsv"
        if path.exists():
            df = pd.read_csv(path, sep="\t")
            df.insert(0, "variant", variant)
            weight_frames.append(df)
    if weight_frames:
        pd.concat(weight_frames, ignore_index=True).to_csv(SOURCE / "relation_or_attention_weights.tsv", sep="\t", index=False)

    val = combined[(combined["method"].eq("Round176")) & (combined["split"].eq("val"))].copy()
    best = None
    for _, row in val.iterrows():
        rec = row.to_dict()
        if best is None or better_val(rec, best):
            best = rec

    test_cols = ["method", "variant", "evaluation_unit", "n_groups", "n_profiles", "Hit@1", "Hit@5", "Hit@10", "Hit@20", "Hit@50", "AUROC", "NDCG@10", "MeanAP"]
    test = combined[combined["split"].eq("test")].copy()
    lines = [
        "# Round176 PC3 HGT-lite structural optimization result report",
        "",
        "This run performs five structural optimizations of the best Round175 graph-prior branch on PC3 single seed. No manuscript or figure files were modified.",
        "",
    ]
    if best is not None:
        lines += [
            "## Selected by validation",
            "",
            f"- Variant: `{best['variant']}`",
            f"- Validation Hit@10: `{best['Hit@10']:.4f}`",
            f"- Validation MeanAP: `{best['MeanAP']:.4f}`",
            "",
        ]
    lines += [
        "## Test comparison",
        "",
        markdown_table(test[[c for c in test_cols if c in test.columns]]),
        "",
        "## Interpretation guardrail",
        "",
        "Round176 main results use drug-mean n=465. Round90 PertKGE 300epoch is profile-level n=3829 context and should not be treated as a strict same-denominator conclusion.",
    ]
    (ROOT / "round176_result_report.md").write_text("\n".join(lines) + "\n", encoding="ascii")
    write_leakage_audit()
    write_optimization_lessons()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=[*VARIANTS, "summarize"], required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.variant == "summarize":
        summarize_all()
    else:
        run_one_variant(args.variant, args.device)


if __name__ == "__main__":
    main()
