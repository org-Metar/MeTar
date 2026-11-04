from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from metar.runtime import report_metrics, report_progress, report_step, resolve_device
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
METAR_ROOT = ROOT.parent
ROUND176_SCRIPT_DIR = METAR_ROOT / "claude_round176_hgtlite_structural_5x_pc3_20260508" / "scripts"
ROUND91_PC3 = METAR_ROOT / "claude_round91_pertkge_inductive_single_cellline_20260507" / "cell_PC3"
PERTKGE_TI = ROUND91_PC3 / "pertkge" / "target_inference"
PERTKGE_KG = ROUND91_PC3 / "pertkge" / "knowledge_graph"

SOURCE = ROOT / "source_data"
QA = ROOT / "qa"
CHECKPOINTS = ROOT / "checkpoints"
LOGS = ROOT / "logs"

ROUND204_BASELINE = SOURCE / "round204_baseline_clean_test_comparison.tsv"
ROUND196_SUMMARY = METAR_ROOT / "claude_round196_regularized_distmult_readout_pc3_20260512" / "source_data" / "pc3_clean_test_comparison.tsv"
ROUND195_SUMMARY = METAR_ROOT / "claude_round195_distmult_no_binding_graph_pretrain_readout_pc3_20260512" / "source_data" / "pc3_clean_test_comparison.tsv"
ROUND185_SUMMARY = METAR_ROOT / "claude_round185_xonly_finetune_target_scorer_pc3_20260511" / "source_data" / "pc3_metric_summary.tsv"

SEED = 20260507
CELL_LINE = "PC3"
METHOD_NAME = "Round211"
ROUND210_PRETRAIN_ROOT = METAR_ROOT / "claude_round210_dim256_checkpoint_sweep_20260515"

GRAPH_EDGE_FILES = {
    "cause": PERTKGE_TI / "cause.txt",
    "effect": PERTKGE_TI / "effect.txt",
    "process": PERTKGE_KG / "process.txt",
}

GRAPH_VARIANTS = [
    {
        "graph_id": "process_only",
        "edge_files": {
            "process": PERTKGE_KG / "process.txt",
        },
        "description": "Process graph only. Effect graph and HAS_BINDING_TO cause edges are not used.",
    },
]

CHECKPOINT_EPOCHS = [5, 10, 15, 20, 25, 30, 35, 40]

READOUT_BATCH_SIZE = 256
READOUT_MIN_EPOCHS = 12
READOUT_PATIENCE = 8
READOUT_WEIGHT_DECAY = 3.0e-4
DNBCE_WEIGHT = 0.20
TARGET_BIAS_L2 = 5.0e-4
TARGET_DELTA_L2 = 5.0e-3
SCORER_L2 = 2.0e-4

READOUT_CONFIGS = [
    {
        "readout_id": "rbase_hit10_control",
        "hidden": 192,
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
    },
    {
        "readout_id": "rwide256_hit10",
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
    },
    {
        "readout_id": "rwide384_hit10",
        "hidden": 384,
        "extra_layers": 0,
        "dropout": 0.35,
        "lr": 4.0e-4,
        "weight_decay": 3.0e-4,
        "target_bias_l2": 5.0e-4,
        "target_delta_l2": 5.0e-3,
        "scorer_l2": 2.0e-4,
        "selection": ["Hit@10", "MeanAP", "Hit@5", "NDCG@10"],
        "patience": 8,
        "max_epochs": 60,
    },
    {
        "readout_id": "rdeep256_hit10",
        "hidden": 256,
        "extra_layers": 1,
        "dropout": 0.35,
        "lr": 4.0e-4,
        "weight_decay": 3.0e-4,
        "target_bias_l2": 5.0e-4,
        "target_delta_l2": 5.0e-3,
        "scorer_l2": 2.0e-4,
        "selection": ["Hit@10", "MeanAP", "Hit@5", "NDCG@10"],
        "patience": 9,
        "max_epochs": 70,
    },
    {
        "readout_id": "rwide256_lowreg_hit10",
        "hidden": 256,
        "extra_layers": 0,
        "dropout": 0.25,
        "lr": 5.0e-4,
        "weight_decay": 1.0e-4,
        "target_bias_l2": 2.0e-4,
        "target_delta_l2": 2.0e-3,
        "scorer_l2": 1.0e-4,
        "selection": ["Hit@10", "MeanAP", "Hit@5", "NDCG@10"],
        "patience": 9,
        "max_epochs": 70,
    },
    {
        "readout_id": "rwide384_lowreg_hit10",
        "hidden": 384,
        "extra_layers": 0,
        "dropout": 0.25,
        "lr": 4.0e-4,
        "weight_decay": 1.0e-4,
        "target_bias_l2": 2.0e-4,
        "target_delta_l2": 2.0e-3,
        "scorer_l2": 1.0e-4,
        "selection": ["Hit@10", "MeanAP", "Hit@5", "NDCG@10"],
        "patience": 9,
        "max_epochs": 70,
    },
]


FLOW_CONFIGS = [
    {
        "flow_id": "r210_a01_reladd2_d256_f32_t020_edge050",
        "decoder_kind": "hgcl",
        "hgcl_aggregator": "rel_add",
        "dim": 256,
        "hidden": 0,
        "n_layers": 2,
        "dropout": 0.15,
        "feature_dropout": 0.10,
        "edge_dropout": 0.20,
        "node_sample_batch": 4096,
        "fanout": 32,
        "temperature": 0.20,
        "edge_loss_weight": 0.50,
        "n_neg": 8,
        "kg_epochs": 40,
        "kg_lr": 7.0e-4,
        "kg_weight_decay": 1.0e-4,
        "kg_val_fraction": 0.02,
        "hypothesis": "rerun the balanced Round205 A01 relation-add HGCL setting at 256 dimensions with full early checkpoint coverage",
        "next_plan": "if 256 improves MeanAP or Hit@10, keep this family and tune regularization; if it degrades, the lower dimension is likely underrepresenting graph semantics",
    },
    {
        "flow_id": "r210_a04_reladd2_d256_f24_t025_edge050",
        "decoder_kind": "hgcl",
        "hgcl_aggregator": "rel_add",
        "dim": 256,
        "hidden": 0,
        "n_layers": 2,
        "dropout": 0.20,
        "feature_dropout": 0.10,
        "edge_dropout": 0.25,
        "node_sample_batch": 3072,
        "fanout": 24,
        "temperature": 0.25,
        "edge_loss_weight": 0.50,
        "n_neg": 8,
        "kg_epochs": 40,
        "kg_lr": 6.0e-4,
        "kg_weight_decay": 1.5e-4,
        "kg_val_fraction": 0.02,
        "hypothesis": "rerun the Round205 A04 softened-temperature relation-add HGCL setting at 256 dimensions with full early checkpoint coverage",
        "next_plan": "if this beats the fanout-32 256 run, the softer contrastive geometry remains preferable at lower dimension; otherwise return to the balanced fanout-32 setting",
    },
    {
        "flow_id": "r210_a02_reladd2_d256_f16_t020_edge050",
        "decoder_kind": "hgcl",
        "hgcl_aggregator": "rel_add",
        "dim": 256,
        "hidden": 0,
        "n_layers": 2,
        "dropout": 0.15,
        "feature_dropout": 0.10,
        "edge_dropout": 0.20,
        "node_sample_batch": 4096,
        "fanout": 16,
        "temperature": 0.20,
        "edge_loss_weight": 0.50,
        "n_neg": 8,
        "kg_epochs": 40,
        "kg_lr": 7.0e-4,
        "kg_weight_decay": 1.0e-4,
        "kg_val_fraction": 0.02,
        "hypothesis": "rerun the Round205 A02 low-fanout relation-add HGCL setting at 256 dimensions to test whether a smaller graph representation benefits from tighter neighborhoods",
        "next_plan": "if low fanout improves 256 Hit@10 or MeanAP, tune fanout between 12 and 24; if it underfits, use A01/A04 as the 256 backbone",
    },
]


def ensure_dirs() -> None:
    for path in [SOURCE, QA, CHECKPOINTS, LOGS]:
        path.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    print(f"[round211] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


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


def load_round176_module():
    sys.path.insert(0, str(ROUND176_SCRIPT_DIR))
    old_argv = sys.argv[:]
    try:
        # The Round176 script parses sys.argv at import time. Keep this import
        # insulated from this script's command-line flags.
        sys.argv = [old_argv[0]]
        import run_round176_hgtlite_structural_5x_pc3 as r176  # noqa: PLC0415
    finally:
        sys.argv = old_argv

    return r176


def read_triple_files(edge_files: Optional[Dict[str, Path]] = None) -> Tuple[pd.DataFrame, dict]:
    if edge_files is None:
        edge_files = GRAPH_EDGE_FILES
    records = []
    audit = {}
    for source_name, path in edge_files.items():
        if not path.exists():
            raise FileNotFoundError(path)
        kept = 0
        excluded = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    continue
                h, r, t = parts
                if r == "HAS_BINDING_TO":
                    excluded += 1
                    continue
                records.append((h, r, t, source_name))
                kept += 1
        audit[f"{source_name}_edges"] = int(kept)
        audit[f"{source_name}_excluded_HAS_BINDING_TO"] = int(excluded)
        audit[f"{source_name}_sha256"] = sha256_file(path)
    triples = pd.DataFrame(records, columns=["head", "relation", "tail", "source"])
    return triples, audit


def build_vocab(
    triples: pd.DataFrame,
    extra_entities: List[str],
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int], dict]:
    base_entities = set(triples["head"]).union(set(triples["tail"]))
    requested_extra = sorted(set(extra_entities))
    added_extra = sorted(set(requested_extra).difference(base_entities))
    entities = sorted(base_entities.union(requested_extra))
    relations = sorted(triples["relation"].unique())
    ent2id = {entity: idx for idx, entity in enumerate(entities)}
    rel2id = {relation: idx for idx, relation in enumerate(relations)}
    encoded = pd.DataFrame(
        {
            "head_id": triples["head"].map(ent2id).astype(np.int64),
            "rel_id": triples["relation"].map(rel2id).astype(np.int64),
            "tail_id": triples["tail"].map(ent2id).astype(np.int64),
            "source": triples["source"].astype(str),
            "relation": triples["relation"].astype(str),
        }
    )
    entity_types = [entity.split(":", 1)[0] if ":" in entity else "OTHER" for entity in entities]
    type_to_ids: Dict[str, np.ndarray] = {}
    for etype in sorted(set(entity_types)):
        ids = np.array([i for i, x in enumerate(entity_types) if x == etype], dtype=np.int64)
        type_to_ids[etype] = ids
    audit = {
        "n_entities": int(len(ent2id)),
        "n_relations": int(len(rel2id)),
        "n_triples": int(len(encoded)),
        "n_extra_entities_requested": int(len(requested_extra)),
        "n_extra_entities_added": int(len(added_extra)),
        "extra_entities_added": added_extra,
        "relation_counts": {k: int(v) for k, v in Counter(triples["relation"]).items()},
        "source_counts": {k: int(v) for k, v in Counter(triples["source"]).items()},
        "entity_type_counts": {k: int(len(v)) for k, v in type_to_ids.items()},
    }
    return encoded, ent2id, rel2id, {"audit": audit, "entity_types": entity_types, "type_to_ids": type_to_ids}


def build_canonical_target_entities(labels: List[str], triples: pd.DataFrame) -> Tuple[List[str], dict]:
    """Map target labels to graph-owned protein-like entities before adding fallbacks."""
    graph_entities = set(triples["head"]).union(set(triples["tail"]))
    target_entities: List[str] = []
    rows = []
    type_counts: Counter[str] = Counter()
    fallback_genes = []
    for gene in labels:
        selected = None
        selected_type = "fallback_Protein"
        for prefix in ["Protein", "TF", "RBP", "mRNA"]:
            entity = f"{prefix}:{gene}"
            if entity in graph_entities:
                selected = entity
                selected_type = prefix
                break
        if selected is None:
            selected = f"Protein:{gene}"
            fallback_genes.append(gene)
        target_entities.append(selected)
        type_counts[selected_type] += 1
        rows.append(
            {
                "gene": gene,
                "target_entity": selected,
                "target_entity_type": selected_type,
                "protein_entity_in_graph": f"Protein:{gene}" in graph_entities,
                "tf_entity_in_graph": f"TF:{gene}" in graph_entities,
                "rbp_entity_in_graph": f"RBP:{gene}" in graph_entities,
                "mrna_entity_in_graph": f"mRNA:{gene}" in graph_entities,
            }
        )
    return target_entities, {
        "rows": rows,
        "audit": {
            "target_entity_policy": "prefer graph-owned Protein, then TF, RBP, mRNA; fallback to Protein only when no graph entity exists",
            "target_entity_type_counts": {k: int(v) for k, v in type_counts.items()},
            "fallback_protein_genes": fallback_genes,
            "n_fallback_protein": int(len(fallback_genes)),
        },
    }


class KGDecoder(torch.nn.Module):
    def __init__(self, n_entities: int, n_relations: int, flow: dict) -> None:
        super().__init__()
        dim = int(flow["dim"])
        hidden = int(flow["hidden"])
        dropout = float(flow["dropout"])
        decoder_kind = str(flow["decoder_kind"])
        self.entity = torch.nn.Embedding(n_entities, dim)
        self.relation = torch.nn.Embedding(n_relations, dim)
        self.decoder_kind = decoder_kind
        self.embedding_dropout = torch.nn.Dropout(float(flow.get("embedding_dropout", 0.0)))
        bound = 6.0 / math.sqrt(dim)
        torch.nn.init.uniform_(self.entity.weight, -bound, bound)
        torch.nn.init.uniform_(self.relation.weight, -bound, bound)
        if decoder_kind == "distmult_linear":
            self.decoder = None
        elif decoder_kind == "concat_mlp":
            input_dim = 5 * dim
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden),
                torch.nn.LayerNorm(hidden),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden, hidden),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden, 1),
            )
        elif decoder_kind == "residual_distmult_mlp":
            input_dim = 5 * dim
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden),
                torch.nn.LayerNorm(hidden),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden, hidden),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden, 1),
            )
            self.residual_scale = torch.nn.Parameter(torch.tensor(float(flow.get("residual_scale_init", 0.1))))
        elif decoder_kind == "relation_transform_mlp":
            rank = int(flow.get("relation_rank", 16))
            self.relation_rank = rank
            self.rel_left = torch.nn.Embedding(n_relations, rank * dim)
            self.rel_right = torch.nn.Embedding(n_relations, rank * dim)
            torch.nn.init.normal_(self.rel_left.weight, mean=0.0, std=0.02)
            torch.nn.init.normal_(self.rel_right.weight, mean=0.0, std=0.02)
            input_dim = 5 * dim
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden),
                torch.nn.LayerNorm(hidden),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden, hidden),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden, 1),
            )
        elif decoder_kind == "complex":
            if dim % 2 != 0:
                raise ValueError("ComplEx decoder requires an even embedding dimension")
            self.decoder = None
        elif decoder_kind == "transe_distance_mlp":
            input_dim = 4 * dim
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden),
                torch.nn.LayerNorm(hidden),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden, hidden),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden, 1),
            )
            self.distance_scale = torch.nn.Parameter(torch.tensor(float(flow.get("distance_scale_init", 1.0))))
        elif decoder_kind == "neighbor_gcn_mlp":
            input_dim = 5 * dim
            self.neighbor_proj = torch.nn.Linear(2 * dim, dim)
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden),
                torch.nn.LayerNorm(hidden),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden, hidden),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden, 1),
            )
        elif decoder_kind == "hgcl":
            self.decoder = None
            self.hgcl_aggregator = str(flow.get("hgcl_aggregator", "rel_add"))
            self.n_layers = int(flow.get("n_layers", 1))
            self.edge_chunk = int(flow.get("edge_chunk", 262144))
            self.sample_msg_proj = torch.nn.Linear(dim, dim)
            self.sample_msg_norm = torch.nn.LayerNorm(dim)
            self.msg_proj = torch.nn.ModuleList([torch.nn.Linear(dim, dim) for _ in range(self.n_layers)])
            self.msg_norm = torch.nn.ModuleList([torch.nn.LayerNorm(dim) for _ in range(self.n_layers)])
            self.proj_head = torch.nn.Sequential(
                torch.nn.Linear(dim, dim),
                torch.nn.GELU(),
                torch.nn.Linear(dim, dim),
            )
            self.edge_head = torch.nn.Linear(dim, dim, bias=False)
        else:
            raise ValueError(f"Unknown decoder_kind: {decoder_kind}")

    def contextual_entity_weight(self) -> torch.Tensor:
        return self.entity.weight

    def set_neighbor_index(self, neighbor_index: torch.Tensor, neighbor_mask: torch.Tensor) -> None:
        self.register_buffer("neighbor_index", neighbor_index)
        self.register_buffer("neighbor_mask", neighbor_mask)

    def set_hgcl_graph(self, src: torch.Tensor, dst: torch.Tensor, rel: torch.Tensor) -> None:
        self.register_buffer("hgcl_src", src)
        self.register_buffer("hgcl_dst", dst)
        self.register_buffer("hgcl_rel", rel)

    def encode_hgcl(self, edge_dropout: float = 0.0, feature_dropout: float = 0.0) -> torch.Tensor:
        h = self.entity.weight
        if self.training and feature_dropout > 0.0:
            h = F.dropout(h, p=feature_dropout, training=True)
        if not hasattr(self, "hgcl_src"):
            return self.embedding_dropout(h)
        src = self.hgcl_src
        dst = self.hgcl_dst
        rel = self.hgcl_rel
        if self.training and edge_dropout > 0.0:
            keep = torch.rand(src.shape[0], device=src.device) >= edge_dropout
            if keep.sum() > 0:
                src = src[keep]
                dst = dst[keep]
                rel = rel[keep]
        for layer in range(self.n_layers):
            agg = torch.zeros_like(h)
            deg = torch.zeros(h.shape[0], dtype=h.dtype, device=h.device)
            for start in range(0, src.shape[0], self.edge_chunk):
                end = min(start + self.edge_chunk, src.shape[0])
                ss = src[start:end]
                dd = dst[start:end]
                rr = rel[start:end]
                rel_vec = self.relation(rr)
                if self.hgcl_aggregator == "rel_mul":
                    msg = h[ss] * torch.sigmoid(rel_vec)
                else:
                    msg = h[ss] + rel_vec
                msg = self.msg_proj[layer](msg)
                agg.index_add_(0, dd, msg)
                deg.index_add_(0, dd, torch.ones(dd.shape[0], dtype=h.dtype, device=h.device))
            agg = agg / deg.clamp_min(1.0).unsqueeze(1)
            h = self.msg_norm[layer](h + F.dropout(torch.tanh(agg), p=float(self.embedding_dropout.p), training=self.training))
        return h

    def encode_hgcl_sampled(
        self,
        nodes: torch.Tensor,
        neigh: torch.Tensor,
        rel: torch.Tensor,
        mask: torch.Tensor,
        feature_dropout: float = 0.0,
    ) -> torch.Tensor:
        base = self.entity(nodes)
        if self.training and feature_dropout > 0.0:
            base = F.dropout(base, p=feature_dropout, training=True)
        neigh_emb = self.entity(neigh)
        rel_emb = self.relation(rel)
        if self.hgcl_aggregator == "rel_mul":
            msg = neigh_emb * torch.sigmoid(rel_emb)
        else:
            msg = neigh_emb + rel_emb
        msg = msg * mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0).unsqueeze(1)
        agg = msg.sum(dim=1) / denom
        out = self.sample_msg_norm(base + F.dropout(torch.tanh(self.sample_msg_proj(agg)), p=float(self.embedding_dropout.p), training=self.training))
        if self.n_layers > 1:
            # A second lightweight residual pass gives the two-layer variants extra capacity
            # without materializing a full two-hop computation graph.
            out = self.sample_msg_norm(out + F.dropout(torch.tanh(self.sample_msg_proj(agg)), p=float(self.embedding_dropout.p), training=self.training))
        return out

    def contextualize(self, ids: torch.Tensor) -> torch.Tensor:
        if self.decoder_kind == "hgcl":
            return self.embedding_dropout(self.entity(ids))
        base = self.entity(ids)
        if self.decoder_kind != "neighbor_gcn_mlp":
            return self.embedding_dropout(base)
        if not hasattr(self, "neighbor_index"):
            return self.embedding_dropout(base)
        neigh_ids = self.neighbor_index[ids]
        neigh_mask = self.neighbor_mask[ids].unsqueeze(-1)
        neigh = self.entity(neigh_ids) * neigh_mask
        denom = neigh_mask.sum(dim=1).clamp_min(1.0)
        neigh_mean = neigh.sum(dim=1) / denom
        ctx = self.neighbor_proj(torch.cat([base, neigh_mean], dim=1))
        return self.embedding_dropout(F.layer_norm(base + torch.tanh(ctx), base.shape[1:]))

    def export_entity_weight(self) -> torch.Tensor:
        if self.decoder_kind == "hgcl":
            return self.entity.weight
        if self.decoder_kind != "neighbor_gcn_mlp" or not hasattr(self, "neighbor_index"):
            return self.entity.weight
        ids = torch.arange(self.entity.weight.shape[0], dtype=torch.long, device=self.entity.weight.device)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            out = self.contextualize(ids)
        if was_training:
            self.train()
        return out

    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.decoder_kind == "hgcl":
            he = self.embedding_dropout(self.entity(h))
            re = self.embedding_dropout(self.relation(r))
            te = self.embedding_dropout(self.entity(t))
            if self.hgcl_aggregator == "rel_mul":
                q = self.edge_head(he * torch.sigmoid(re))
            else:
                q = self.edge_head(he + re)
            return F.cosine_similarity(q, te, dim=1) * math.sqrt(float(he.shape[1]))
        he = self.contextualize(h)
        re = self.embedding_dropout(self.relation(r))
        te = self.contextualize(t)
        if self.decoder_kind == "distmult_linear":
            return (he * re * te).sum(dim=1)
        if self.decoder_kind == "concat_mlp":
            features = torch.cat([he, re, te, torch.abs(he - te), he * te], dim=1)
            return self.decoder(features).squeeze(1)
        if self.decoder_kind == "residual_distmult_mlp":
            base = (he * re * te).sum(dim=1)
            features = torch.cat([he, re, te, torch.abs(he - te), he * te], dim=1)
            residual = self.decoder(features).squeeze(1)
            return base + self.residual_scale * residual
        if self.decoder_kind == "relation_transform_mlp":
            bsz, dim = he.shape
            left = self.rel_left(r).reshape(bsz, self.relation_rank, dim)
            right = self.rel_right(r).reshape(bsz, self.relation_rank, dim)
            coeff = (he[:, None, :] * left).sum(dim=2)
            delta = (coeff[:, :, None] * right).sum(dim=1) / math.sqrt(float(self.relation_rank))
            h_rel = he + delta
            features = torch.cat([h_rel, re, te, torch.abs(h_rel - te), h_rel * te], dim=1)
            return self.decoder(features).squeeze(1)
        if self.decoder_kind == "complex":
            h_re, h_im = torch.chunk(he, 2, dim=1)
            r_re, r_im = torch.chunk(re, 2, dim=1)
            t_re, t_im = torch.chunk(te, 2, dim=1)
            return (
                h_re * r_re * t_re
                + h_im * r_re * t_im
                + h_re * r_im * t_im
                - h_im * r_im * t_re
            ).sum(dim=1)
        if self.decoder_kind == "transe_distance_mlp":
            translated = he + re
            delta = translated - te
            base = -torch.exp(torch.clamp(self.distance_scale, -2.0, 2.0)) * torch.linalg.vector_norm(delta, ord=1, dim=1) / he.shape[1]
            features = torch.cat([translated, te, torch.abs(delta), delta], dim=1)
            return base + self.decoder(features).squeeze(1)
        if self.decoder_kind == "neighbor_gcn_mlp":
            features = torch.cat([he, re, te, torch.abs(he - te), he * te], dim=1)
            return self.decoder(features).squeeze(1)
        raise ValueError(f"Unknown decoder_kind: {self.decoder_kind}")

    def embedding_regularization(self) -> torch.Tensor:
        return torch.square(self.entity.weight).mean() + torch.square(self.relation.weight).mean()


def sample_corruptions(
    head: np.ndarray,
    tail: np.ndarray,
    n_entities: int,
    n_neg: int,
    rng: np.random.Generator,
    typed_negative: bool,
    entity_types: List[str],
    type_to_ids: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    n_pos = len(head)
    corrupt_head = rng.random((n_pos, n_neg)) < 0.5
    neg_h = np.repeat(head[:, None], n_neg, axis=1)
    neg_t = np.repeat(tail[:, None], n_neg, axis=1)
    if not typed_negative:
        rand_h = rng.integers(0, n_entities, size=(n_pos, n_neg), dtype=np.int64)
        rand_t = rng.integers(0, n_entities, size=(n_pos, n_neg), dtype=np.int64)
        neg_h[corrupt_head] = rand_h[corrupt_head]
        neg_t[~corrupt_head] = rand_t[~corrupt_head]
    else:
        flat_ch = corrupt_head.reshape(-1)
        flat_h = neg_h.reshape(-1)
        flat_t = neg_t.reshape(-1)
        for corrupt, original, target_flat in [
            (flat_ch, np.repeat(head[:, None], n_neg, axis=1).reshape(-1), flat_h),
            (~flat_ch, np.repeat(tail[:, None], n_neg, axis=1).reshape(-1), flat_t),
        ]:
            idx_all = np.where(corrupt)[0]
            if len(idx_all) == 0:
                continue
            by_type: Dict[str, List[int]] = {}
            for idx in idx_all:
                by_type.setdefault(entity_types[int(original[idx])], []).append(int(idx))
            for etype, positions in by_type.items():
                pool = type_to_ids.get(etype)
                if pool is None or len(pool) <= 1:
                    vals = rng.integers(0, n_entities, size=len(positions), dtype=np.int64)
                else:
                    vals = rng.choice(pool, size=len(positions), replace=True).astype(np.int64)
                target_flat[np.array(positions, dtype=np.int64)] = vals
        neg_h = flat_h.reshape(n_pos, n_neg)
        neg_t = flat_t.reshape(n_pos, n_neg)
    mask_h = neg_h == head[:, None]
    while mask_h.any():
        neg_h[mask_h] = rng.integers(0, n_entities, size=int(mask_h.sum()), dtype=np.int64)
        mask_h = neg_h == head[:, None]
    mask_t = neg_t == tail[:, None]
    while mask_t.any():
        neg_t[mask_t] = rng.integers(0, n_entities, size=int(mask_t.sum()), dtype=np.int64)
        mask_t = neg_t == tail[:, None]
    return neg_h.reshape(-1), neg_t.reshape(-1)


def split_kg_edges(encoded: pd.DataFrame, val_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if val_fraction <= 0.0:
        return encoded.reset_index(drop=True), encoded.iloc[0:0].copy().reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    train_parts = []
    val_parts = []
    for _rel, group in encoded.groupby("rel_id", sort=False):
        idx = group.index.to_numpy()
        n_val = max(1, int(round(len(idx) * val_fraction))) if len(idx) >= 50 else 0
        if n_val == 0:
            train_parts.append(group)
            continue
        val_idx = set(rng.choice(idx, size=n_val, replace=False).tolist())
        mask = group.index.to_series().isin(val_idx)
        val_parts.append(group[mask])
        train_parts.append(group[~mask])
    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True) if val_parts else encoded.iloc[0:0].copy()
    return train, val.reset_index(drop=True)


def build_neighbor_index(edge_df: pd.DataFrame, n_entities: int, max_neighbors: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(SEED)
    adj: List[List[int]] = [[] for _ in range(n_entities)]
    heads = edge_df["head_id"].to_numpy(np.int64)
    tails = edge_df["tail_id"].to_numpy(np.int64)
    for h_id, t_id in zip(heads, tails):
        h_int = int(h_id)
        t_int = int(t_id)
        adj[h_int].append(t_int)
        adj[t_int].append(h_int)
    index = np.zeros((n_entities, max_neighbors), dtype=np.int64)
    mask = np.zeros((n_entities, max_neighbors), dtype=np.float32)
    for ent_id, neigh in enumerate(adj):
        if not neigh:
            continue
        if len(neigh) > max_neighbors:
            chosen = rng.choice(np.array(neigh, dtype=np.int64), size=max_neighbors, replace=False)
        else:
            chosen = np.array(neigh, dtype=np.int64)
        index[ent_id, : len(chosen)] = chosen
        mask[ent_id, : len(chosen)] = 1.0
    return (
        torch.tensor(index, dtype=torch.long, device=device),
        torch.tensor(mask, dtype=torch.float32, device=device),
    )


def build_directed_edge_tensors(edge_df: pd.DataFrame, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    heads = torch.tensor(edge_df["head_id"].to_numpy(np.int64), dtype=torch.long, device=device)
    tails = torch.tensor(edge_df["tail_id"].to_numpy(np.int64), dtype=torch.long, device=device)
    rels = torch.tensor(edge_df["rel_id"].to_numpy(np.int64), dtype=torch.long, device=device)
    src = torch.cat([heads, tails], dim=0)
    dst = torch.cat([tails, heads], dim=0)
    rel = torch.cat([rels, rels], dim=0)
    return src, dst, rel


def build_sampled_neighbor_tables(edge_df: pd.DataFrame, n_entities: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    neigh: List[List[int]] = [[] for _ in range(n_entities)]
    rels: List[List[int]] = [[] for _ in range(n_entities)]
    for row in edge_df[["head_id", "rel_id", "tail_id"]].itertuples(index=False):
        h_id = int(row.head_id)
        r_id = int(row.rel_id)
        t_id = int(row.tail_id)
        neigh[h_id].append(t_id)
        rels[h_id].append(r_id)
        neigh[t_id].append(h_id)
        rels[t_id].append(r_id)
    neigh_np = [np.asarray(x if x else [i], dtype=np.int64) for i, x in enumerate(neigh)]
    rel_np = [np.asarray(x if x else [0], dtype=np.int64) for x in rels]
    return neigh_np, rel_np


def sample_node_batch(
    n_entities: int,
    neigh_table: List[np.ndarray],
    rel_table: List[np.ndarray],
    batch_size: int,
    fanout: int,
    rng: np.random.Generator,
    device: torch.device,
    fixed_nodes: Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if fixed_nodes is None:
        nodes = rng.choice(np.arange(n_entities, dtype=np.int64), size=min(batch_size, n_entities), replace=False)
    else:
        nodes = fixed_nodes.astype(np.int64, copy=False)
    neigh = np.zeros((len(nodes), fanout), dtype=np.int64)
    rels = np.zeros((len(nodes), fanout), dtype=np.int64)
    mask = np.zeros((len(nodes), fanout), dtype=np.float32)
    for i, node in enumerate(nodes):
        pool = neigh_table[int(node)]
        rpool = rel_table[int(node)]
        if len(pool) == 0:
            continue
        replace = len(pool) < fanout
        idx = rng.choice(np.arange(len(pool)), size=fanout, replace=replace)
        neigh[i] = pool[idx]
        rels[i] = rpool[idx]
        mask[i, : min(fanout, len(pool))] = 1.0 if not replace else 1.0
    return (
        torch.tensor(nodes, dtype=torch.long, device=device),
        torch.tensor(neigh, dtype=torch.long, device=device),
        torch.tensor(rels, dtype=torch.long, device=device),
        torch.tensor(mask, dtype=torch.float32, device=device),
    )


def node_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float, batch_size: int = 2048) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    n = z1.shape[0]
    losses = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        logits12 = z1[start:end] @ z2[start:end].T / temperature
        labels = torch.arange(0, end - start, dtype=torch.long, device=z1.device)
        logits21 = z2[start:end] @ z1[start:end].T / temperature
        losses.append(0.5 * (F.cross_entropy(logits12, labels) + F.cross_entropy(logits21, labels)))
    return torch.stack(losses).mean()


def edge_contrastive_loss(
    model: KGDecoder,
    h: torch.Tensor,
    r: torch.Tensor,
    t: torch.Tensor,
    n_entities: int,
    n_neg: int,
    rng: np.random.Generator,
    vocab_meta: dict,
    device: torch.device,
) -> torch.Tensor:
    hp = h.detach().cpu().numpy().astype(np.int64)
    tp = t.detach().cpu().numpy().astype(np.int64)
    nh, nt = sample_corruptions(
        hp,
        tp,
        n_entities,
        n_neg,
        rng,
        True,
        vocab_meta["entity_types"],
        vocab_meta["type_to_ids"],
    )
    pos = model.score(h, r, t)
    neg = model.score(
        torch.tensor(np.repeat(hp, n_neg), dtype=torch.long, device=device),
        torch.tensor(np.repeat(r.detach().cpu().numpy().astype(np.int64), n_neg), dtype=torch.long, device=device),
        torch.tensor(nt, dtype=torch.long, device=device),
    ).reshape(len(hp), n_neg)
    logits = torch.cat([pos[:, None], neg], dim=1)
    labels = torch.zeros(len(hp), dtype=torch.long, device=device)
    return F.cross_entropy(logits, labels)


def evaluate_kg_link_metrics(
    model: KGDecoder,
    edge_df: pd.DataFrame,
    n_entities: int,
    n_neg: int,
    rng: np.random.Generator,
    vocab_meta: dict,
    device: torch.device,
    max_edges: int = 20000,
) -> dict:
    if edge_df.empty:
        return {"kg_val_loss": np.nan, "kg_val_pos_mean": np.nan, "kg_val_neg_mean": np.nan, "kg_val_pair_acc": np.nan}
    sample_df = edge_df
    if len(edge_df) > max_edges:
        sample_df = edge_df.sample(n=max_edges, random_state=SEED)
    hp = sample_df["head_id"].to_numpy(np.int64)
    rp = sample_df["rel_id"].to_numpy(np.int64)
    tp = sample_df["tail_id"].to_numpy(np.int64)
    nh, nt = sample_corruptions(
        hp,
        tp,
        n_entities,
        n_neg,
        rng,
        True,
        vocab_meta["entity_types"],
        vocab_meta["type_to_ids"],
    )
    model.eval()
    with torch.no_grad():
        pos = model.score(
            torch.tensor(hp, dtype=torch.long, device=device),
            torch.tensor(rp, dtype=torch.long, device=device),
            torch.tensor(tp, dtype=torch.long, device=device),
        )
        neg = model.score(
            torch.tensor(np.repeat(hp, n_neg), dtype=torch.long, device=device),
            torch.tensor(np.repeat(rp, n_neg), dtype=torch.long, device=device),
            torch.tensor(nt, dtype=torch.long, device=device),
        ).reshape(len(hp), n_neg)
        logits = torch.cat([pos, neg.reshape(-1)])
        labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg.reshape(-1))])
        loss = F.binary_cross_entropy_with_logits(logits, labels).item()
        pair_acc = (pos[:, None] > neg).float().mean().item()
    return {
        "kg_val_loss": float(loss),
        "kg_val_pos_mean": float(pos.mean().detach().cpu()),
        "kg_val_neg_mean": float(neg.mean().detach().cpu()),
        "kg_val_pair_acc": float(pair_acc),
    }


def train_kg_decoder(
    flow: dict,
    encoded: pd.DataFrame,
    ent2id: Dict[str, int],
    rel2id: Dict[str, int],
    vocab_meta: dict,
    device: torch.device,
    graph_id: str,
    checkpoint_callback: Optional[Callable[[Path], None]] = None,
) -> Tuple[List[Path], pd.DataFrame]:
    report_step('Preparing gene-prior graph training')
    set_seed()
    flow_id = flow["flow_id"]
    graph_dir = CHECKPOINTS / graph_id / flow_id
    graph_dir.mkdir(parents=True, exist_ok=True)
    train_encoded, val_encoded = split_kg_edges(encoded, float(flow.get("kg_val_fraction", 0.0)))
    model = KGDecoder(len(ent2id), len(rel2id), flow).to(device)
    if str(flow["decoder_kind"]) == "neighbor_gcn_mlp":
        neighbor_index, neighbor_mask = build_neighbor_index(
            train_encoded,
            len(ent2id),
            int(flow.get("max_neighbors", 12)),
            device,
        )
        model.set_neighbor_index(neighbor_index, neighbor_mask)
    if str(flow["decoder_kind"]) == "hgcl":
        neigh_table, rel_table = build_sampled_neighbor_tables(train_encoded, len(ent2id))
        src, dst, rel = build_directed_edge_tensors(train_encoded, device)
        model.set_hgcl_graph(src, dst, rel)
    opt = torch.optim.AdamW(model.parameters(), lr=float(flow["kg_lr"]), weight_decay=float(flow["kg_weight_decay"]))
    rng = np.random.default_rng(SEED)
    heads = train_encoded["head_id"].to_numpy(np.int64)
    rels = train_encoded["rel_id"].to_numpy(np.int64)
    tails = train_encoded["tail_id"].to_numpy(np.int64)
    rel_counts = Counter(train_encoded["rel_id"].to_numpy(np.int64).tolist())
    rel_weight_np = np.ones(len(rel2id), dtype=np.float32)
    if flow.get("relation_weighted", False):
        counts = np.array([rel_counts[i] for i in range(len(rel2id))], dtype=np.float32)
        raw = 1.0 / np.sqrt(np.maximum(counts, 1.0))
        rel_weight_np = raw / raw.mean()
    rel_weight = torch.tensor(rel_weight_np, dtype=torch.float32, device=device)
    n_edges = len(train_encoded)
    batch_size = int(flow.get("kg_batch_size", 8192))
    n_neg = int(flow["n_neg"])
    history = []
    checkpoint_paths: List[Path] = []
    log(f"{graph_id}/{flow_id} KG pretrain start: kind={flow['decoder_kind']} dim={flow['dim']} hidden={flow['hidden']} neg={n_neg} typed={flow.get('typed_negative', True)} train_edges={len(train_encoded)} val_edges={len(val_encoded)}")
    report_progress(0, int(flow["kg_epochs"]), label='kg decoder epochs')
    report_metrics(flow, label="Training settings")
    for epoch in range(1, int(flow["kg_epochs"]) + 1):
        model.train()
        losses = []
        if str(flow["decoder_kind"]) == "hgcl":
            opt.zero_grad(set_to_none=True)
            nodes1, neigh1, rel1, mask1 = sample_node_batch(
                len(ent2id),
                neigh_table,
                rel_table,
                int(flow.get("node_sample_batch", 4096)),
                int(flow.get("fanout", 24)),
                rng,
                device,
            )
            # The second view uses the same seed nodes but independently sampled typed neighbors.
            _nodes2, neigh2, rel2, mask2 = sample_node_batch(
                len(ent2id),
                neigh_table,
                rel_table,
                len(nodes1),
                int(flow.get("fanout", 24)),
                rng,
                device,
                fixed_nodes=nodes1.detach().cpu().numpy(),
            )
            nodes2 = nodes1
            z1 = model.proj_head(model.encode_hgcl_sampled(nodes1, neigh1, rel1, mask1, feature_dropout=float(flow.get("feature_dropout", 0.1))))
            z2 = model.proj_head(model.encode_hgcl_sampled(nodes2, neigh2, rel2, mask2, feature_dropout=float(flow.get("feature_dropout", 0.1))))
            loss = node_contrastive_loss(z1, z2, float(flow.get("temperature", 0.2)), batch_size=int(flow.get("node_batch_size", 2048)))
            edge_weight = float(flow.get("edge_loss_weight", 0.0))
            if edge_weight > 0.0:
                idx = rng.choice(n_edges, size=min(batch_size, n_edges), replace=False)
                edge_loss = edge_contrastive_loss(
                    model,
                    torch.tensor(heads[idx], dtype=torch.long, device=device),
                    torch.tensor(rels[idx], dtype=torch.long, device=device),
                    torch.tensor(tails[idx], dtype=torch.long, device=device),
                    len(ent2id),
                    n_neg,
                    rng,
                    vocab_meta,
                    device,
                )
                loss = loss + edge_weight * edge_loss
            loss = loss + 1.0e-6 * model.embedding_regularization()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        else:
            order = rng.permutation(n_edges)
            for start in range(0, n_edges, batch_size):
                report_progress(start, n_edges, label=f"Epoch {epoch} training rows")
                idx = order[start:start + batch_size]
                hp = heads[idx]
                rp = rels[idx]
                tp = tails[idx]
                nh, nt = sample_corruptions(
                    hp,
                    tp,
                    len(ent2id),
                    n_neg,
                    rng,
                bool(flow.get("typed_negative", True)),
                    vocab_meta["entity_types"],
                    vocab_meta["type_to_ids"],
                )
                h_all = np.concatenate([hp, np.repeat(hp, n_neg)])
                r_all = np.concatenate([rp, np.repeat(rp, n_neg)])
                t_all = np.concatenate([tp, nt])
                y_all = np.concatenate([np.ones(len(hp), dtype=np.float32), np.zeros(len(nt), dtype=np.float32)])
                perm = rng.permutation(len(y_all))
                h_t = torch.tensor(h_all[perm], dtype=torch.long, device=device)
                r_t = torch.tensor(r_all[perm], dtype=torch.long, device=device)
                t_t = torch.tensor(t_all[perm], dtype=torch.long, device=device)
                y_t = torch.tensor(y_all[perm], dtype=torch.float32, device=device)
                opt.zero_grad(set_to_none=True)
                logits = model.score(h_t, r_t, t_t)
                loss_vec = F.binary_cross_entropy_with_logits(logits, y_t, reduction="none")
                loss_vec = loss_vec * rel_weight[r_t]
                loss = loss_vec.mean()
                margin_weight = float(flow.get("margin_weight", 0.0))
                if margin_weight > 0.0:
                    pos_logits = model.score(
                        torch.tensor(hp, dtype=torch.long, device=device),
                        torch.tensor(rp, dtype=torch.long, device=device),
                        torch.tensor(tp, dtype=torch.long, device=device),
                    )
                    neg_logits = model.score(
                        torch.tensor(np.repeat(hp, n_neg), dtype=torch.long, device=device),
                        torch.tensor(np.repeat(rp, n_neg), dtype=torch.long, device=device),
                        torch.tensor(nt, dtype=torch.long, device=device),
                    ).reshape(len(hp), n_neg)
                    margin = float(flow.get("margin", 1.0))
                    margin_loss = F.relu(margin - pos_logits[:, None] + neg_logits).mean()
                    loss = loss + margin_weight * margin_loss
                loss = loss + 1.0e-6 * model.embedding_regularization()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                losses.append(float(loss.detach().cpu()))
            report_progress(n_edges, n_edges, label=f"Epoch {epoch} training rows")
        epoch_loss = float(np.mean(losses))
        rec = {"graph_id": graph_id, "flow_id": flow_id, "epoch": epoch, "kg_loss": epoch_loss}
        if epoch == 1 or epoch % 5 == 0 or epoch == int(flow["kg_epochs"]):
            rec.update(evaluate_kg_link_metrics(model, val_encoded, len(ent2id), min(n_neg, 4), rng, vocab_meta, device))
            log(f"{graph_id}/{flow_id} KG epoch={epoch} loss={epoch_loss:.5f} val_loss={rec.get('kg_val_loss', np.nan):.5f} pair_acc={rec.get('kg_val_pair_acc', np.nan):.4f}")
        else:
            log(f"{graph_id}/{flow_id} KG epoch={epoch} loss={epoch_loss:.5f}")
        history.append(rec)
        report_progress(epoch, int(flow["kg_epochs"]), label='kg decoder epochs')
        report_metrics(
            {key: rec[key] for key in ("kg_loss", "kg_val_loss", "kg_val_pair_acc") if key in rec},
            label=f"Epoch {epoch}/{int(flow['kg_epochs'])}",
        )
        if epoch in CHECKPOINT_EPOCHS:
            ckpt_flow = dict(flow)
            ckpt_flow["flow_id"] = f"{flow_id}_{graph_id}_kg{epoch:03d}"
            ckpt_flow["base_flow_id"] = flow_id
            ckpt_flow["graph_id"] = graph_id
            ckpt_flow["kg_checkpoint_epoch"] = int(epoch)
            ckpt = {
                "flow": ckpt_flow,
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "export_entity_weight": model.export_entity_weight().detach().cpu().clone(),
                "ent2id": ent2id,
                "rel2id": rel2id,
                "kg_dim": int(flow["dim"]),
                "graph_id": graph_id,
                "kg_checkpoint_epoch": int(epoch),
                "target_entities": vocab_meta.get("target_entities"),
                "target_entity_audit": vocab_meta.get("audit", {}),
            }
            ckpt_path = graph_dir / f"{flow_id}_kg_checkpoint_epoch{epoch:03d}.pth"
            torch.save(ckpt, ckpt_path)
            checkpoint_paths.append(ckpt_path)
            log(f"{graph_id}/{flow_id} saved KG checkpoint epoch={epoch}")
            if checkpoint_callback is not None:
                checkpoint_callback(ckpt_path)
    pd.DataFrame(history).to_csv(SOURCE / f"kg_history_{graph_id}_{flow_id}.tsv", sep="\t", index=False)
    return checkpoint_paths, pd.DataFrame(history)


class DiagReadout(torch.nn.Module):
    def __init__(self, target_init: np.ndarray, readout_cfg: dict) -> None:
        super().__init__()
        dim = int(target_init.shape[1])
        self.dim = dim
        self.n_targets = int(target_init.shape[0])
        self.readout_cfg = dict(readout_cfg)
        self.register_buffer("target_anchor", F.normalize(torch.tensor(target_init, dtype=torch.float32), dim=1))
        hidden = int(readout_cfg.get("hidden", max(128, min(192, dim * 3))))
        extra_layers = int(readout_cfg.get("extra_layers", 0))
        dropout = float(readout_cfg.get("dropout", 0.35))
        layers: List[torch.nn.Module] = [
            torch.nn.Linear(978, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        ]
        for _ in range(max(0, extra_layers)):
            layers.extend(
                [
                    torch.nn.Linear(hidden, hidden),
                    torch.nn.LayerNorm(hidden),
                    torch.nn.GELU(),
                    torch.nn.Dropout(dropout),
                ]
            )
        layers.extend([torch.nn.Linear(hidden, dim), torch.nn.LayerNorm(dim)])
        self.query_encoder = torch.nn.Sequential(*layers)
        self.query_norm = torch.nn.LayerNorm(dim)
        self.diag = torch.nn.Parameter(torch.ones(dim))
        self.target_delta = torch.nn.Parameter(torch.zeros_like(self.target_anchor))
        self.score_scale = torch.nn.Parameter(torch.tensor(0.7))
        self.target_bias = torch.nn.Parameter(torch.zeros(self.n_targets))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = F.normalize(self.query_norm(self.query_encoder(x)), dim=1)
        target = F.normalize(self.target_anchor + self.target_delta, dim=1)
        score = (q * self.diag[None, :]) @ target.T
        score = score * torch.exp(torch.clamp(self.score_scale, -2.0, 2.0))
        return score + self.target_bias[None, :]

    def regularization(self) -> torch.Tensor:
        reg = float(self.readout_cfg.get("target_bias_l2", TARGET_BIAS_L2)) * torch.square(self.target_bias).sum()
        reg = reg + float(self.readout_cfg.get("scorer_l2", SCORER_L2)) * torch.square(self.diag - 1.0).sum()
        reg = reg + float(self.readout_cfg.get("target_delta_l2", TARGET_DELTA_L2)) * torch.square(self.target_delta).sum()
        return reg


def split_data(rows: pd.DataFrame, x: np.ndarray, y: np.ndarray, r176) -> dict:
    report_step('Building bulk train, validation and test splits')
    out = {}
    for split in ["inner_train", "val", "test"]:
        mask = rows["split"].astype(str).eq(split).to_numpy()
        split_rows = rows[mask].reset_index(drop=True)
        out[split] = {
            "rows": split_rows,
            "x": x[mask],
            "y": y[mask],
            "weights": r176.row_weights(split_rows),
        }
    return out


def predict(model: torch.nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            report_progress(start, len(x), label='predict rows')
            xb = torch.tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            out.append(model(xb).detach().cpu().numpy())
        report_progress(len(x), len(x), label='predict rows')
    return np.vstack(out)


def better_val(candidate: dict, best: Optional[dict], selection: Optional[List[str]] = None) -> bool:
    if best is None:
        return True
    for key in (selection or ["Hit@10", "MeanAP", "Hit@5", "NDCG@10"]):
        diff = candidate[key] - best[key]
        if abs(diff) > 1.0e-12:
            return diff > 0.0
    return False


def target_embeddings_from_ckpt(ckpt: dict, labels: List[str], output_flow_id: Optional[str] = None) -> Tuple[np.ndarray, dict]:
    ent_tensor = ckpt.get("export_entity_weight", ckpt["model_state"]["entity.weight"])
    ent = ent_tensor.detach().cpu().numpy().astype(np.float32)
    ent2id = ckpt["ent2id"]
    dim = int(ckpt["kg_dim"])
    target = np.zeros((len(labels), dim), dtype=np.float32)
    target_entities = ckpt.get("target_entities")
    if target_entities is None:
        raise RuntimeError(f"KG checkpoint lacks canonical target_entities; audit={ckpt.get('vocab_meta', {}).get('audit', {})}")
    if len(target_entities) != len(labels):
        raise RuntimeError(f"target_entities length mismatch: {len(target_entities)} vs {len(labels)}")
    missing = []
    rows = []
    for j, gene in enumerate(labels):
        entity = target_entities[j]
        idx = ent2id.get(entity)
        if idx is None:
            missing.append(f"{gene}->{entity}")
        else:
            target[j] = ent[int(idx)]
            rows.append({"gene": gene, "target_entity": entity, "target_entity_type": entity.split(":", 1)[0] if ":" in entity else "OTHER"})
    if missing:
        raise RuntimeError(f"Missing canonical target embeddings: {missing[:10]}")
    audit_flow_id = output_flow_id or ckpt["flow"]["flow_id"]
    pd.DataFrame(rows).to_csv(SOURCE / f"benchmark3_readout_target_entities_{audit_flow_id}.tsv", sep="\t", index=False)
    return target, {
        "target_hits": len(labels),
        "kg_dim": dim,
        "flow": ckpt["flow"],
        "target_entity_type_counts": {k: int(v) for k, v in Counter(row["target_entity_type"] for row in rows).items()},
    }


def train_readout(flow: dict, ckpt: dict, data: dict, labels: List[str], r176, device: torch.device, readout_cfg: Optional[dict] = None) -> Tuple[dict, pd.DataFrame]:
    report_step('Preparing bulk readout training')
    readout_cfg = dict(readout_cfg or READOUT_CONFIGS[0])
    base_flow_id = str(flow["flow_id"])
    flow = dict(flow)
    flow_id = f"{base_flow_id}_{readout_cfg['readout_id']}"
    flow["flow_id"] = flow_id
    flow["base_checkpoint_flow_id"] = base_flow_id
    flow["readout_id"] = readout_cfg["readout_id"]
    target, target_audit = target_embeddings_from_ckpt(ckpt, labels, flow_id)
    model = DiagReadout(target, readout_cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(readout_cfg.get("lr", 5.0e-4)),
        weight_decay=float(readout_cfg.get("weight_decay", READOUT_WEIGHT_DECAY)),
    )
    tensors = {
        split: {
            "x": torch.tensor(item["x"], dtype=torch.float32, device=device),
            "y": torch.tensor(item["y"], dtype=torch.float32, device=device),
            "w": torch.tensor(item["weights"], dtype=torch.float32, device=device),
        }
        for split, item in data.items()
    }
    rng = np.random.default_rng(SEED)
    n_train = tensors["inner_train"]["x"].shape[0]
    best_state = None
    best_val = None
    best_epoch = 0
    stale = 0
    history = []
    selection = list(readout_cfg.get("selection", ["Hit@10", "MeanAP", "Hit@5", "NDCG@10"]))
    patience = int(readout_cfg.get("patience", READOUT_PATIENCE))
    max_epochs = int(readout_cfg.get("max_epochs", 60))
    min_epochs = int(readout_cfg.get("min_epochs", READOUT_MIN_EPOCHS))
    report_progress(0, max_epochs, label='readout epochs')
    report_metrics(readout_cfg, label="Training settings")
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(n_train)
        losses = []
        for start in range(0, n_train, READOUT_BATCH_SIZE):
            report_progress(start, n_train, label=f"Epoch {epoch} training rows")
            idx = torch.tensor(order[start:start + READOUT_BATCH_SIZE], dtype=torch.long, device=device)
            xb = tensors["inner_train"]["x"][idx]
            yb = tensors["inner_train"]["y"][idx]
            wb = tensors["inner_train"]["w"][idx]
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = r176.bag_softmax_loss(logits, yb, wb)
            loss = loss + DNBCE_WEIGHT * r176.drug_normalized_bce_loss(logits, yb, wb)
            loss = loss + model.regularization()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        report_progress(n_train, n_train, label=f"Epoch {epoch} training rows")
        report_step(f"Epoch {epoch}: evaluating existing training and validation metrics")
        train_scores = predict(model, data["inner_train"]["x"], device)
        val_scores = predict(model, data["val"]["x"], device)
        train_summary, _ = r176.compute_metrics(train_scores, data["inner_train"]["y"], data["inner_train"]["rows"], METHOD_NAME, flow_id, "drug-mean")
        val_summary, _ = r176.compute_metrics(val_scores, data["val"]["y"], data["val"]["rows"], METHOD_NAME, flow_id, "drug-mean")
        rec = {
            "flow_id": flow_id,
            "base_checkpoint_flow_id": base_flow_id,
            "readout_id": readout_cfg["readout_id"],
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_Hit@10": train_summary["Hit@10"],
            "train_MeanAP": train_summary["MeanAP"],
            **{f"val_{k}": v for k, v in val_summary.items() if k in {"Hit@5", "Hit@10", "NDCG@10", "MeanAP"}},
        }
        history.append(rec)
        report_progress(epoch, max_epochs, label='readout epochs')
        report_metrics(
            {key: rec[key] for key in ("train_loss", "val_Hit@10", "val_MeanAP")},
            label=f"Epoch {epoch}/{max_epochs}",
        )
        if epoch == 1 or epoch % 5 == 0:
            log(f"{flow_id} readout epoch={epoch} train_Hit@10={rec['train_Hit@10']:.4f} val_Hit@10={rec['val_Hit@10']:.4f} val_MeanAP={rec['val_MeanAP']:.4f}")
        if better_val(val_summary, best_val, selection):
            best_val = val_summary
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch >= min_epochs and stale >= patience:
            log(f"{flow_id} readout early stop epoch={epoch}; best_epoch={best_epoch}")
            break
    if best_state is None or best_val is None:
        raise RuntimeError(f"No readout checkpoint selected for {flow_id}")
    report_metrics(
        {"epochs_run": len(history), "best_epoch": best_epoch, "Hit@10": best_val["Hit@10"], "MeanAP": best_val["MeanAP"]},
        label="Selected readout validation metrics",
    )
    model.load_state_dict(best_state)
    history_df = pd.DataFrame(history)
    history_df.to_csv(SOURCE / f"readout_history_{flow_id}.tsv", sep="\t", index=False)
    summaries = []
    per_rows = []
    for split in ["inner_train", "val", "test"]:
        report_step(f"Evaluating selected readout: {split}")
        scores = predict(model, data[split]["x"], device)
        summary, per = r176.compute_metrics(scores, data[split]["y"], data[split]["rows"], METHOD_NAME, flow_id, "drug-mean")
        summary["best_epoch"] = best_epoch
        summary["kg_dim"] = int(flow["dim"])
        summary["graph_id"] = str(flow.get("graph_id", ""))
        summary["kg_checkpoint_epoch"] = int(flow.get("kg_checkpoint_epoch", -1))
        summary["base_checkpoint_flow_id"] = base_flow_id
        summary["readout_id"] = readout_cfg["readout_id"]
        summary["readout_selection"] = ",".join(selection)
        summary["readout_dropout"] = float(readout_cfg.get("dropout", 0.35))
        summary["readout_hidden"] = int(readout_cfg.get("hidden", max(128, min(192, int(flow["dim"]) * 3))))
        summary["readout_extra_layers"] = int(readout_cfg.get("extra_layers", 0))
        summary["readout_lr"] = float(readout_cfg.get("lr", 5.0e-4))
        summary["readout_weight_decay"] = float(readout_cfg.get("weight_decay", READOUT_WEIGHT_DECAY))
        summary["target_delta_l2"] = float(readout_cfg.get("target_delta_l2", TARGET_DELTA_L2))
        summary["target_bias_l2"] = float(readout_cfg.get("target_bias_l2", TARGET_BIAS_L2))
        summary["scorer_l2"] = float(readout_cfg.get("scorer_l2", SCORER_L2))
        summaries.append(summary)
        per_rows.append(per)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(SOURCE / f"metric_summary_{flow_id}.tsv", sep="\t", index=False)
    pd.concat(per_rows, ignore_index=True).to_csv(SOURCE / f"per_drug_metrics_{flow_id}.tsv", sep="\t", index=False)
    (CHECKPOINTS / flow_id).mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "flow": flow,
            "state_dict": model.state_dict(),
            "selected": {"best_epoch": best_epoch, "best_val": best_val},
            "readout_cfg": readout_cfg,
            "target_audit": target_audit,
            "labels": labels,
        },
        CHECKPOINTS / flow_id / "readout_checkpoint.pth",
    )
    return summary_df[summary_df["split"].eq("test")].iloc[0].to_dict(), history_df


def analyze_flow(flow: dict, test_summary: dict, history_df: pd.DataFrame, previous_best: Optional[dict]) -> dict:
    best_epoch = int(test_summary["best_epoch"])
    hist_best = history_df[history_df["epoch"].eq(best_epoch)]
    if hist_best.empty:
        hist_best = history_df.tail(1)
    train_hit = float(hist_best.iloc[0]["train_Hit@10"])
    val_hit = float(hist_best.iloc[0]["val_Hit@10"])
    val_meanap = float(hist_best.iloc[0]["val_MeanAP"])
    val_hit5 = float(hist_best.iloc[0]["val_Hit@5"])
    val_ndcg10 = float(hist_best.iloc[0]["val_NDCG@10"])
    overfit_gap = train_hit - val_hit
    hit10 = float(test_summary["Hit@10"])
    meanap = float(test_summary["MeanAP"])
    if previous_best is None:
        result_note = "first multiplicative KG decoder baseline"
    else:
        dh = hit10 - float(previous_best["Hit@10"])
        dm = meanap - float(previous_best["MeanAP"])
        if dh > 0.003 or dm > 0.003:
            result_note = f"improved over previous best (delta Hit@10={dh:.4f}, delta MeanAP={dm:.4f})"
        elif overfit_gap > 0.55:
            result_note = "readout still overfits; next flow should reduce capacity or strengthen negatives"
        else:
            result_note = f"no clear improvement (delta Hit@10={dh:.4f}, delta MeanAP={dm:.4f})"
    return {
        "flow_id": flow["flow_id"],
        "hypothesis": flow["hypothesis"],
        "Hit@10": hit10,
        "MeanAP": meanap,
        "AUROC": float(test_summary["AUROC"]),
        "NDCG@10": float(test_summary["NDCG@10"]),
        "best_epoch": best_epoch,
        "base_flow_id": str(flow.get("base_flow_id", flow["flow_id"])),
        "train_Hit@10_at_best_epoch": train_hit,
        "val_Hit@10_at_best_epoch": val_hit,
        "val_MeanAP_at_best_epoch": val_meanap,
        "val_Hit@5_at_best_epoch": val_hit5,
        "val_NDCG@10_at_best_epoch": val_ndcg10,
        "overfit_gap_at_best_epoch": overfit_gap,
        "analysis": result_note,
        "next_model_plan": flow["next_plan"],
    }


def read_existing_summary(path: Path, method_filter: Optional[str] = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t")
    if "split" in df.columns:
        df = df[df["split"].astype(str).eq("test")].copy()
    if method_filter is not None and "method" in df.columns:
        df = df[df["method"].astype(str).eq(method_filter)].copy()
    return df


def summarize_all() -> None:
    frames = []
    for path in sorted(SOURCE.glob("metric_summary_*.tsv")):
        frames.append(read_existing_summary(path))
    baseline = read_existing_summary(ROUND204_BASELINE)
    if not baseline.empty:
        baseline["baseline_source"] = "Round204_fixed_target_entity"
        frames.append(baseline)
    frames.append(read_existing_summary(ROUND196_SUMMARY, "Round196"))
    frames.append(read_existing_summary(ROUND195_SUMMARY, "Round195"))
    frames.append(read_existing_summary(ROUND185_SUMMARY))
    combined = pd.concat([df for df in frames if not df.empty], ignore_index=True, sort=False)
    combined.to_csv(SOURCE / "combined_metric_summary.tsv", sep="\t", index=False)
    clean = combined.copy()
    clean["n_eval_units"] = clean.get("n_groups", clean.get("n_drugs", np.nan))
    cols = [
        "method", "variant", "evaluation_unit", "n_eval_units", "n_drugs", "n_profiles", "kg_dim",
        "Hit@1", "Hit@5", "Hit@10", "Hit@20", "Hit@30", "Hit@50", "AUROC", "NDCG@10", "MeanAP", "best_epoch",
    ]
    extra_cols = [
        "graph_id", "kg_checkpoint_epoch", "base_checkpoint_flow_id", "readout_id", "readout_selection",
        "readout_hidden", "readout_extra_layers", "readout_dropout", "readout_lr", "readout_weight_decay",
        "target_delta_l2", "target_bias_l2", "scorer_l2",
    ]
    clean = clean[[c for c in [*extra_cols, *cols] if c in clean.columns]].copy()
    clean.to_csv(SOURCE / "clean_test_comparison.tsv", sep="\t", index=False)


def select_flows(flow_ids: str) -> List[dict]:
    if not flow_ids:
        return FLOW_CONFIGS
    requested = [item.strip() for item in flow_ids.split(",") if item.strip()]
    by_id = {flow["flow_id"]: flow for flow in FLOW_CONFIGS}
    missing = [flow_id for flow_id in requested if flow_id not in by_id]
    if missing:
        raise ValueError(f"Unknown flow_ids: {missing}")
    return [by_id[flow_id] for flow_id in requested]


def select_readout_configs(readout_ids: str) -> List[dict]:
    if not readout_ids:
        return READOUT_CONFIGS
    requested = [item.strip() for item in readout_ids.split(",") if item.strip()]
    by_id = {cfg["readout_id"]: cfg for cfg in READOUT_CONFIGS}
    missing = [rid for rid in requested if rid not in by_id]
    if missing:
        raise ValueError(f"Unknown readout_ids: {missing}")
    return [by_id[rid] for rid in requested]


def discover_round210_checkpoints(args: argparse.Namespace) -> List[Path]:
    if args.checkpoints:
        return [Path(item).resolve() for item in args.checkpoints.split(",") if item.strip()]
    root = ROUND210_PRETRAIN_ROOT / "checkpoints" / "process_only"
    patterns = [p.strip() for p in args.checkpoint_patterns.split(",") if p.strip()]
    out: List[Path] = []
    for pattern in patterns:
        out.extend(root.glob(pattern))
    selected_epochs = {int(x) for x in args.checkpoint_epochs.split(",") if x.strip()} if args.checkpoint_epochs else None
    selected_base = {x.strip() for x in args.base_flow_ids.split(",") if x.strip()} if args.base_flow_ids else None
    filtered = []
    for path in sorted(set(out)):
        ckpt = torch.load(path, map_location="cpu")
        flow = ckpt["flow"]
        if int(ckpt.get("kg_dim", -1)) != 256:
            continue
        if selected_epochs is not None and int(flow.get("kg_checkpoint_epoch", -1)) not in selected_epochs:
            continue
        if selected_base is not None and str(flow.get("base_flow_id", "") or str(flow["flow_id"]).split("_process_only_kg", 1)[0]) not in selected_base:
            continue
        filtered.append(path)
    return filtered


def wait_for_one_readout(workers: List[subprocess.Popen]) -> None:
    while workers:
        for proc in list(workers):
            ret = proc.poll()
            if ret is None:
                continue
            workers.remove(proc)
            if ret != 0:
                raise RuntimeError(f"Readout worker failed with exit code {ret}: pid={proc.pid}")
            log(f"readout worker finished pid={proc.pid}")
            return
        time.sleep(5)


def wait_for_all_readouts(workers: List[subprocess.Popen]) -> None:
    while workers:
        wait_for_one_readout(workers)


def launch_readout_worker(ckpt_path: Path, readout_cfg: dict, args: argparse.Namespace, workers: List[subprocess.Popen]) -> None:
    while len(workers) >= int(args.max_readout_workers):
        wait_for_one_readout(workers)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    flow_id = f"{ckpt['flow']['flow_id']}_{readout_cfg['readout_id']}"
    log_path = LOGS / f"readout_{flow_id}.log"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--readout-only",
        "--checkpoint",
        str(ckpt_path),
        "--readout-id",
        str(readout_cfg["readout_id"]),
        "--device",
        str(args.device),
    ]
    with log_path.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
    workers.append(proc)
    log(f"launched readout worker pid={proc.pid} flow={flow_id} log={log_path}")


def run_readout_only(args: argparse.Namespace) -> None:
    ensure_dirs()
    set_seed()
    if not args.checkpoint:
        raise ValueError("--checkpoint is required with --readout-only")
    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    device = torch.device(resolve_device(args.device))
    r176 = load_round176_module()
    _landmarks, labels, rows, x, y = r176.load_response()
    data = split_data(rows, x, y, r176)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    flow = ckpt["flow"]
    cfgs = select_readout_configs(args.readout_id)
    if len(cfgs) != 1:
        raise ValueError("--readout-only requires exactly one --readout-id")
    readout_cfg = cfgs[0]
    log(f"==== readout-only {flow['flow_id']} readout={readout_cfg['readout_id']} from {ckpt_path} ====")
    train_readout(flow, ckpt, data, labels, r176, device, readout_cfg)
    log(f"readout-only done {flow['flow_id']} readout={readout_cfg['readout_id']}")


def run_readout_sweep(args: argparse.Namespace) -> None:
    ensure_dirs()
    set_seed()
    r176 = load_round176_module()
    _landmarks, labels, rows, x, y = r176.load_response()
    data = split_data(rows, x, y, r176)
    device = torch.device(resolve_device(args.device))
    ckpt_paths = discover_round210_checkpoints(args)
    if not ckpt_paths:
        raise RuntimeError("No Round210 KG checkpoints found for readout sweep")
    readout_cfgs = select_readout_configs(args.readout_ids)
    log(f"readout-sweep checkpoints={len(ckpt_paths)} readouts={[cfg['readout_id'] for cfg in readout_cfgs]}")
    if args.dry_run:
        for ckpt_path in ckpt_paths:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            log(f"dry-run checkpoint={ckpt_path} flow={ckpt['flow']['flow_id']} kg_dim={ckpt.get('kg_dim')} epoch={ckpt.get('kg_checkpoint_epoch')}")
        return
    workers: List[subprocess.Popen] = []
    if int(args.max_readout_workers) > 1:
        for ckpt_path in ckpt_paths:
            for readout_cfg in readout_cfgs:
                launch_readout_worker(ckpt_path, readout_cfg, args, workers)
        wait_for_all_readouts(workers)
    else:
        for ckpt_path in ckpt_paths:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            for readout_cfg in readout_cfgs:
                log(f"==== inline sweep {ckpt['flow']['flow_id']} readout={readout_cfg['readout_id']} ====")
                train_readout(ckpt["flow"], ckpt, data, labels, r176, device, readout_cfg)
    summarize_all()
    analysis_df = collect_checkpoint_analyses()
    if not analysis_df.empty:
        analysis_df.to_csv(SOURCE / "round211_dim256_readout_capacity_analysis.tsv", sep="\t", index=False)
    lock = {
        "round": "round211_dim256_readout_capacity_sweep_pc3",
        "seed": SEED,
        "cell_line": CELL_LINE,
        "pretrained_kg_source": str(ROUND210_PRETRAIN_ROOT),
        "checkpoint_paths": [str(p) for p in ckpt_paths],
        "readout_configs": readout_cfgs,
        "max_readout_workers": int(args.max_readout_workers),
        "evaluation_unit": "drug-mean",
        "candidate_space": 668,
        "fixed": "Round210 256-dim KG pretraining is reused; no KG retraining in Round211",
    }
    (ROOT / "method_lock.json").write_text(json.dumps(lock, indent=2, default=str), encoding="utf-8")


def collect_checkpoint_analyses() -> pd.DataFrame:
    rows = []
    for path in sorted(SOURCE.glob("metric_summary_*.tsv")):
        df = read_existing_summary(path)
        if df.empty:
            continue
        rec = df.iloc[0].to_dict()
        flow_id = str(rec.get("variant", path.stem.replace("metric_summary_", "")))
        history_path = SOURCE / f"readout_history_{flow_id}.tsv"
        hist = pd.read_csv(history_path, sep="\t") if history_path.exists() else pd.DataFrame()
        best_epoch = int(rec.get("best_epoch", -1)) if not pd.isna(rec.get("best_epoch", np.nan)) else -1
        hist_best = hist[hist["epoch"].eq(best_epoch)] if not hist.empty and best_epoch >= 0 else pd.DataFrame()
        if hist_best.empty and not hist.empty:
            hist_best = hist.tail(1)
        train_hit = float(hist_best.iloc[0].get("train_Hit@10", np.nan)) if not hist_best.empty else np.nan
        val_hit = float(hist_best.iloc[0].get("val_Hit@10", np.nan)) if not hist_best.empty else np.nan
        val_meanap = float(hist_best.iloc[0].get("val_MeanAP", np.nan)) if not hist_best.empty else np.nan
        val_hit5 = float(hist_best.iloc[0].get("val_Hit@5", np.nan)) if not hist_best.empty else np.nan
        val_ndcg10 = float(hist_best.iloc[0].get("val_NDCG@10", np.nan)) if not hist_best.empty else np.nan
        rows.append(
            {
                "flow_id": flow_id,
                "base_flow_id": str(rec.get("variant", flow_id)).split("_process_only_kg", 1)[0],
                "base_checkpoint_flow_id": rec.get("base_checkpoint_flow_id", ""),
                "readout_id": rec.get("readout_id", ""),
                "readout_selection": rec.get("readout_selection", ""),
                "readout_hidden": rec.get("readout_hidden", np.nan),
                "readout_extra_layers": rec.get("readout_extra_layers", np.nan),
                "readout_dropout": rec.get("readout_dropout", np.nan),
                "readout_lr": rec.get("readout_lr", np.nan),
                "target_delta_l2": rec.get("target_delta_l2", np.nan),
                "graph_id": rec.get("graph_id", ""),
                "kg_checkpoint_epoch": rec.get("kg_checkpoint_epoch", np.nan),
                "kg_dim": rec.get("kg_dim", np.nan),
                "Hit@10": rec.get("Hit@10", np.nan),
                "MeanAP": rec.get("MeanAP", np.nan),
                "AUROC": rec.get("AUROC", np.nan),
                "NDCG@10": rec.get("NDCG@10", np.nan),
                "best_epoch": rec.get("best_epoch", np.nan),
                "train_Hit@10_at_best_epoch": train_hit,
                "val_Hit@10_at_best_epoch": val_hit,
                "val_MeanAP_at_best_epoch": val_meanap,
                "val_Hit@5_at_best_epoch": val_hit5,
                "val_NDCG@10_at_best_epoch": val_ndcg10,
                "overfit_gap_at_best_epoch": train_hit - val_hit if not np.isnan(train_hit) and not np.isnan(val_hit) else np.nan,
            }
        )
    analysis_df = pd.DataFrame(rows)
    if not analysis_df.empty:
        analysis_df.to_csv(SOURCE / "round211_dim256_readout_capacity_analysis.tsv", sep="\t", index=False)
    return analysis_df


def train_kg_and_launch_readouts(args: argparse.Namespace, r176, labels, rows, data, device: torch.device) -> Tuple[pd.DataFrame, dict]:
    analyses = []
    graph_audits = {}
    selected_graph_ids = {item.strip() for item in args.graph_ids.split(",") if item.strip()} if args.graph_ids else None
    selected_graphs = [g for g in GRAPH_VARIANTS if selected_graph_ids is None or g["graph_id"] in selected_graph_ids]
    selected_flows = select_flows(args.flow_ids)
    readout_cfgs = select_readout_configs(args.readout_ids)
    for graph_variant in selected_graphs:
        graph_id = str(graph_variant["graph_id"])
        log(f"==== graph {graph_id} ====")
        triples, file_audit = read_triple_files(graph_variant["edge_files"])
        candidate_entities, target_entity_info = build_canonical_target_entities(labels, triples)
        encoded, ent2id, rel2id, vocab_meta = build_vocab(triples, candidate_entities)
        vocab_meta["audit"].update(target_entity_info["audit"])
        vocab_meta["target_entities"] = candidate_entities
        pd.DataFrame(target_entity_info["rows"]).to_csv(SOURCE / f"benchmark3_target_entity_audit_{graph_id}.tsv", sep="\t", index=False)
        eval_drugs = set(rows.loc[rows["split"].isin(["val", "test"]), "drug"].astype(str))
        eval_cids = sorted(f"CID:{d}" for d in eval_drugs if f"CID:{d}" in ent2id)
        if eval_cids:
            raise RuntimeError(f"Validation/test CID entities found in graph {graph_id}: {len(eval_cids)}")
        graph_audit = {
            "description": graph_variant["description"],
            **file_audit,
            **vocab_meta["audit"],
            "eval_cids_in_graph_count": 0,
        }
        graph_audits[graph_id] = graph_audit
        (QA / f"graph_input_audit_{graph_id}.json").write_text(json.dumps(graph_audit, indent=2), encoding="utf-8")
        if args.dry_run:
            log(
                f"dry-run graph={graph_id} triples={len(encoded)} entities={len(ent2id)} "
                f"relations={len(rel2id)} flows={[flow['flow_id'] for flow in selected_flows]} "
                f"readouts={[cfg['readout_id'] for cfg in readout_cfgs]}"
            )
            continue
        workers: List[subprocess.Popen] = []

        def run_checkpoint_probe(ckpt_path: Path) -> None:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            ckpt_flow = ckpt["flow"]
            for readout_cfg in readout_cfgs:
                if int(args.max_readout_workers) > 1:
                    launch_readout_worker(ckpt_path, readout_cfg, args, workers)
                else:
                    log(f"==== inline readout {ckpt_flow['flow_id']} readout={readout_cfg['readout_id']} ====")
                    test_summary, history_df = train_readout(ckpt_flow, ckpt, data, labels, r176, device, readout_cfg)
                    analysis = analyze_flow(ckpt_flow, test_summary, history_df, None)
                    analysis["graph_id"] = graph_id
                    analysis["kg_checkpoint_epoch"] = int(ckpt_flow["kg_checkpoint_epoch"])
                    analysis["readout_id"] = readout_cfg["readout_id"]
                    analyses.append(analysis)
                    pd.DataFrame(analyses).to_csv(SOURCE / "round210_dim256_checkpoint_sweep_analysis.tsv", sep="\t", index=False)
                    summarize_all()

        for flow_config in selected_flows:
            train_kg_decoder(
                flow_config,
                encoded,
                ent2id,
                rel2id,
                vocab_meta,
                device,
                graph_id,
                checkpoint_callback=run_checkpoint_probe,
            )
        wait_for_all_readouts(workers)
    summarize_all()
    analysis_df = collect_checkpoint_analyses()
    if analysis_df.empty and analyses:
        analysis_df = pd.DataFrame(analyses)
        analysis_df.to_csv(SOURCE / "round210_dim256_checkpoint_sweep_analysis.tsv", sep="\t", index=False)
    return analysis_df, graph_audits


def run_all(args: argparse.Namespace) -> None:
    ensure_dirs()
    set_seed()
    if args.summarize:
        summarize_all()
        return
    if args.readout_only:
        run_readout_only(args)
        return
    if args.readout_sweep:
        run_readout_sweep(args)
        return
    raise RuntimeError("Round211 is readout-only. Use --readout-sweep or --readout-only; KG pretraining is intentionally disabled.")
    (ROOT / "task_plan.md").write_text(
        "\n".join(
            [
                "# Round210 Dim-256 KG Checkpoint Sweep Plan",
                "",
                "## Goal",
                "Retune dim=256 KG checkpoint coverage using Round205 successful checkpoint-sweep discipline, then run selected readout configurations.",
                "",
                "## Fixed Evaluation",
                "- PC3 drug-mean",
                "- 668 targets",
                "- target anchors use graph-owned Protein/TF/RBP/mRNA entities before fallback",
                "- train new process-only dim=256 KG checkpoints at 5/10/15/20/25/30/35/40",
                "- KG is retrained in this round; downstream split/evaluation stay fixed",
                "- downstream readout architecture remains Round196-style diagonal target-delta readout",
                "- use the readout settings that Round208 showed were informative: rstrong_map, rstrong_hit10, rbase_hit10",
                "",
                "## Status",
                "Running automated Round210 256-dim KG checkpoint sweep on server.",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    r176 = load_round176_module()
    _landmarks, labels, rows, x, y = r176.load_response()
    data = split_data(rows, x, y, r176)
    device = torch.device(resolve_device(args.device))
    analysis_df, graph_audits = train_kg_and_launch_readouts(args, r176, labels, rows, data, device)
    if analysis_df.empty:
        return
    selected = (
        analysis_df.sort_values(
            ["base_checkpoint_flow_id", "readout_id", "val_Hit@10_at_best_epoch", "val_MeanAP_at_best_epoch", "val_Hit@5_at_best_epoch", "val_NDCG@10_at_best_epoch"],
            ascending=[True, True, False, False, False, False],
        )
        .groupby(["base_checkpoint_flow_id", "readout_id"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    selected.to_csv(SOURCE / "round210_validation_selected_by_checkpoint_readout.tsv", sep="\t", index=False)
    best = selected.sort_values(["val_Hit@10_at_best_epoch", "val_MeanAP_at_best_epoch", "Hit@10", "MeanAP"], ascending=False).head(1)
    lines = [
        "# Round210 dim-256 KG checkpoint sweep report",
        "",
        "New process-only dim=256 KG checkpoints were trained with full Round205-style checkpoint coverage. Each checkpoint was evaluated with selected Round208 readout settings.",
        "",
        "## Best validation-selected checkpoint in this run",
        "",
        best.to_csv(sep="\t", index=False),
        "",
        "## Validation-selected checkpoint per checkpoint/readout setting",
        "",
        selected.to_csv(sep="\t", index=False),
        "",
        "## All checkpoint analyses",
        "",
        analysis_df.to_csv(sep="\t", index=False),
        "",
        "## Combined comparison",
        "",
        pd.read_csv(SOURCE / "clean_test_comparison.tsv", sep="\t").to_csv(sep="\t", index=False) if (SOURCE / "clean_test_comparison.tsv").exists() else "Run with --summarize after flows finish.",
    ]
    (ROOT / "round210_result_report.md").write_text("\n".join(lines), encoding="utf-8")
    lock = {
        "round": "round210_dim256_checkpoint_sweep_pc3",
        "seed": SEED,
        "cell_line": CELL_LINE,
        "new_checkpoint_root": str(CHECKPOINTS / "process_only"),
        "graph_audits": graph_audits,
        "flows": select_flows(args.flow_ids),
        "checkpoint_epochs": CHECKPOINT_EPOCHS,
        "readout_configs": select_readout_configs(args.readout_ids),
        "max_readout_workers": int(args.max_readout_workers),
        "fixed_readout": "Round196-style diagonal target-delta readout",
        "evaluation_unit": "drug-mean",
        "candidate_space": 668,
        "selection_rule": "per readout config; see READOUT_CONFIGS",
    }
    (ROOT / "method_lock.json").write_text(json.dumps(lock, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--graph-ids", default="")
    parser.add_argument("--flow-ids", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--readout-only", action="store_true")
    parser.add_argument("--readout-sweep", action="store_true")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--checkpoints", default="")
    parser.add_argument("--checkpoint-patterns", default="r210_a01_reladd2_d256_f32_t020_edge050/*epoch020.pth,r210_a01_reladd2_d256_f32_t020_edge050/*epoch025.pth,r210_a01_reladd2_d256_f32_t020_edge050/*epoch040.pth,r210_a02_reladd2_d256_f16_t020_edge050/*epoch030.pth,r210_a02_reladd2_d256_f16_t020_edge050/*epoch035.pth,r210_a04_reladd2_d256_f24_t025_edge050/*epoch040.pth")
    parser.add_argument("--checkpoint-epochs", default="5,10,15,20,25,30,35,40")
    parser.add_argument("--base-flow-ids", default="")
    parser.add_argument("--readout-id", default="")
    parser.add_argument("--readout-ids", default="rbase_hit10_control,rwide256_hit10,rwide384_hit10,rdeep256_hit10,rwide256_lowreg_hit10,rwide384_lowreg_hit10")
    parser.add_argument("--max-readout-workers", type=int, default=2)
    args = parser.parse_args()
    run_all(args)


if __name__ == "__main__":
    main()

