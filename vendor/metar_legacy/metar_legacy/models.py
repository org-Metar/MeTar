"""Compact PyTorch modules for MeTar training workflows."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResidualMLPEncoder(nn.Module):
    """Sample-side encoder used for bulk, single-cell and proteomic queries."""

    def __init__(self, input_dim: int, embed_dim: int = 256, hidden_dim: int = 512, n_blocks: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout=dropout) for _ in range(n_blocks)])
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.output(self.blocks(self.input(x)))
        return F.normalize(q, dim=-1, eps=1e-8)


class TargetTableScorer(nn.Module):
    """Scaled-cosine scorer against a target representation table."""

    def __init__(self, target_table: torch.Tensor, trainable_target_table: bool = False, init_logit_scale: float = 10.0) -> None:
        super().__init__()
        target = F.normalize(target_table.detach().float(), dim=-1, eps=1e-8)
        if trainable_target_table:
            self.target_table = nn.Parameter(target)
        else:
            self.register_buffer("target_table", target)
        self.logit_scale = nn.Parameter(torch.tensor(float(init_logit_scale)).log())

    def normalized_targets(self) -> torch.Tensor:
        return F.normalize(self.target_table, dim=-1, eps=1e-8)

    def forward(self, query_embedding: torch.Tensor, target_indices: torch.Tensor | None = None) -> torch.Tensor:
        target = self.normalized_targets()
        if target_indices is not None:
            target = target.index_select(0, target_indices)
        return query_embedding @ target.T * self.logit_scale.exp().clamp(max=100.0)


class PPITargetEncoder(nn.Module):
    """Lightweight PPI message-passing encoder for full-space target tables.

    The release version implements the target-side operation used by the
    full-space branch in a compact form: node features are projected into the
    MeTar embedding dimension, neighbor messages are averaged over an undirected
    edge index and residual blocks refine the target representation.
    """

    def __init__(self, input_dim: int, embed_dim: int = 256, hidden_dim: int = 256, n_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(n_layers)
            ]
        )
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, embed_dim))

    @staticmethod
    def aggregate(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return x
        src, dst = edge_index.long()
        out = torch.zeros_like(x)
        out.index_add_(0, dst, x[src])
        deg = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        return out / deg.clamp_min(1.0).unsqueeze(1)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.input(node_features.float()))
        for layer in self.layers:
            msg = self.aggregate(h, edge_index)
            h = h + layer(0.5 * (h + msg))
        return F.normalize(self.output(h), dim=-1, eps=1e-8)


class QueryGatedSupportAdapter(nn.Module):
    """Episodic query-gated support adapter for held-out cell contexts.

    Support labels are used to construct a support target prototype from the
    fixed target table. The adapter returns a query-specific residual shift and
    gate; the shifted query remains in the same target space.
    """

    def __init__(self, target_table: torch.Tensor, embed_dim: int = 256, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.register_buffer("target_table", F.normalize(target_table.float(), dim=-1, eps=1e-8))
        self.delta = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def support_prototype(self, support_q: torch.Tensor, support_y: torch.Tensor, support_weight: torch.Tensor | None = None) -> torch.Tensor:
        target_proto = support_y.float() @ self.target_table
        denom = support_y.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        target_proto = target_proto / denom
        if support_weight is not None:
            weight = support_weight.float().reshape(-1, 1)
            target_proto = target_proto * weight
        proto = torch.cat([support_q, target_proto], dim=-1).mean(dim=0, keepdim=True)
        return F.normalize(proto[:, : support_q.shape[1]] + proto[:, support_q.shape[1] :], dim=-1, eps=1e-8)

    def forward(
        self,
        query_q: torch.Tensor,
        support_q: torch.Tensor,
        support_y: torch.Tensor,
        support_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        proto = self.support_prototype(support_q, support_y, support_weight).expand_as(query_q)
        features = torch.cat([query_q, proto, query_q - proto], dim=-1)
        delta = F.normalize(self.delta(features), dim=-1, eps=1e-8)
        gate = self.gate(features).squeeze(-1)
        return delta, gate


def apply_query_shift(query_q: torch.Tensor, delta: torch.Tensor, gate: torch.Tensor, shift_scale: float = 0.18) -> torch.Tensor:
    return F.normalize(query_q + float(shift_scale) * gate.reshape(-1, 1) * delta, dim=-1, eps=1e-8)


class SingleCellGatedAggregate(nn.Module):
    """Gated residual aggregation of cell-level embeddings into a drug query."""

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, cell_embeddings: torch.Tensor, group_index: torch.Tensor) -> torch.Tensor:
        groups = torch.unique(group_index, sorted=True)
        outputs = []
        for group in groups:
            cells = cell_embeddings[group_index == group]
            mean = cells.mean(dim=0, keepdim=True)
            features = torch.cat([cells, mean.expand_as(cells), cells - mean], dim=-1)
            weights = self.gate(features)
            pooled = (weights * cells).sum(dim=0) / weights.sum().clamp_min(1e-6)
            beta = torch.sigmoid(self.beta)
            outputs.append(F.normalize((1 - beta) * mean.squeeze(0) + beta * pooled, dim=-1, eps=1e-8))
        return torch.stack(outputs, dim=0)


class LabelAdapter(nn.Module):
    """Linear or bottleneck adapter applied to a fixed target table."""

    def __init__(self, embed_dim: int = 256, mode: str = "linear", bottleneck: int = 64) -> None:
        super().__init__()
        if mode == "linear":
            self.net = nn.Linear(embed_dim, embed_dim, bias=False)
            nn.init.eye_(self.net.weight)
        elif mode == "bottleneck":
            self.net = nn.Sequential(
                nn.Linear(embed_dim, bottleneck),
                nn.GELU(),
                nn.Linear(bottleneck, embed_dim),
            )
        else:
            raise ValueError(f"Unknown label adapter mode: {mode}")
        self.mode = mode

    def forward(self, target_table: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(target_table.float()), dim=-1, eps=1e-8)

    def regularization_loss(self, weight: float = 0.0) -> torch.Tensor:
        if weight <= 0 or not isinstance(self.net, nn.Linear):
            return torch.tensor(0.0, device=next(self.parameters()).device)
        eye = torch.eye(self.net.weight.shape[0], device=self.net.weight.device)
        return float(weight) * F.mse_loss(self.net.weight, eye)


def drug_normalized_bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy with positive and negative branches balanced per row."""

    loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
    pos = labels > 0
    neg = ~pos
    pos_loss = (loss * pos.float()).sum(dim=1) / pos.float().sum(dim=1).clamp_min(1.0)
    neg_loss = (loss * neg.float()).sum(dim=1) / neg.float().sum(dim=1).clamp_min(1.0)
    return 0.5 * (pos_loss + neg_loss).mean()


def label_bag_retrieval_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Multi-positive bag retrieval loss over target logits."""

    pos_mask = labels > 0
    if not bool(pos_mask.any()):
        return logits.sum() * 0.0
    log_prob = F.log_softmax(logits, dim=1)
    per_row = -(log_prob * pos_mask.float()).sum(dim=1) / pos_mask.float().sum(dim=1).clamp_min(1.0)
    return per_row.mean()
