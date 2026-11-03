"""Exact inference modules for the checkpoint schemas selected by the manuscript."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must contain a mapping: {path}")
    return payload


class ClosedDiagReadout(nn.Module):
    """Round246 978-feature scorer over the article's 668-target space."""

    def __init__(
        self,
        target_anchor: torch.Tensor,
        readout_cfg: dict[str, Any],
        input_dim: int = 978,
    ) -> None:
        super().__init__()
        dim = int(target_anchor.shape[1])
        self.register_buffer(
            "target_anchor", F.normalize(target_anchor.detach().float(), dim=1)
        )
        hidden = int(readout_cfg.get("hidden", 256))
        dropout = float(readout_cfg.get("dropout", 0.35))
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(max(0, int(readout_cfg.get("extra_layers", 0)))):
            layers.extend(
                [
                    nn.Linear(hidden, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.extend([nn.Linear(hidden, dim), nn.LayerNorm(dim)])
        self.query_encoder = nn.Sequential(*layers)
        self.query_norm = nn.LayerNorm(dim)
        self.diag = nn.Parameter(torch.ones(dim))
        self.target_delta = nn.Parameter(torch.zeros_like(self.target_anchor))
        self.score_scale = nn.Parameter(torch.tensor(0.7))
        self.target_bias = nn.Parameter(torch.zeros(int(target_anchor.shape[0])))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_norm(self.query_encoder(x)), dim=1)

    def normalized_targets(self) -> torch.Tensor:
        return F.normalize(self.target_anchor + self.target_delta, dim=1)

    def score_embeddings(self, query: torch.Tensor) -> torch.Tensor:
        score = (query * self.diag[None, :]) @ self.normalized_targets().T
        return (
            score * torch.exp(torch.clamp(self.score_scale, -2.0, 2.0))
            + self.target_bias[None, :]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.score_embeddings(self.encode(x))

    @classmethod
    def from_checkpoint(
        cls, path: str | Path
    ) -> tuple["ClosedDiagReadout", tuple[str, ...], dict[str, Any]]:
        payload = load_checkpoint(path)
        if (
            "readout_state_dict" not in payload
            or "readout_cfg" not in payload
            or "labels" not in payload
        ):
            raise ValueError(
                "expected a bulk checkpoint with readout_state_dict, readout_cfg, and labels"
            )
        state = payload["readout_state_dict"]
        target_anchor = state["target_anchor"].detach().float()
        input_dim = int(state["query_encoder.0.weight"].shape[1])
        model = cls(target_anchor, dict(payload["readout_cfg"]), input_dim=input_dim)
        model.load_state_dict(state, strict=True)
        model.eval()
        labels = tuple(str(value).split(":", 1)[-1] for value in payload["labels"])
        if len(labels) != target_anchor.shape[0] or len(set(labels)) != len(labels):
            raise ValueError("bulk checkpoint target labels are invalid")
        return model, labels, payload


class QueryGatedSupportAdapter(nn.Module):
    """Round252 EpiQGate adapter for support-conditioned cross-cell prediction."""

    def __init__(
        self,
        target_anchor: torch.Tensor,
        hidden_dim: int,
        dropout: float,
        max_delta_norm: float,
    ) -> None:
        super().__init__()
        dim = int(target_anchor.shape[1])
        self.dim = dim
        self.max_delta_norm = float(max_delta_norm)
        self.register_buffer("target_anchor", target_anchor.detach().clone())
        input_dim = dim * 4
        self.support_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.key_proj = nn.Linear(input_dim, dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(
        self,
        query_q: torch.Tensor,
        support_q: torch.Tensor,
        support_y: torch.Tensor,
        support_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positive_count = torch.clamp(support_y.sum(dim=1, keepdim=True), min=1.0)
        positive_prototype = F.normalize(
            (support_y @ self.target_anchor) / positive_count, dim=1
        )
        features = torch.cat(
            [
                support_q,
                positive_prototype,
                support_q * positive_prototype,
                positive_prototype - support_q,
            ],
            dim=1,
        )
        row_delta = self.support_mlp(features)
        support_key = F.normalize(self.key_proj(features), dim=1)
        attention_logits = query_q @ support_key.T / math.sqrt(float(self.dim))
        attention_logits = (
            attention_logits + torch.log(torch.clamp(support_w, min=1.0e-6))[None, :]
        )
        attention = torch.softmax(attention_logits, dim=1)
        delta = self.out_norm(attention @ row_delta)
        if self.max_delta_norm > 0:
            norm = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
            shrink = torch.clamp(
                torch.tensor(
                    self.max_delta_norm, dtype=delta.dtype, device=delta.device
                )
                / torch.clamp(norm, min=1.0e-6),
                max=1.0,
            )
            delta = delta * shrink
        context = attention @ support_key
        gate_features = torch.cat(
            [query_q, context, query_q * context, context - query_q], dim=1
        )
        gate = torch.sigmoid(self.gate_mlp(gate_features)).squeeze(-1)
        return delta, gate

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        base_target_anchor: torch.Tensor,
    ) -> tuple["QueryGatedSupportAdapter", dict[str, Any]]:
        payload = load_checkpoint(path)
        if "adapter_state_dict" not in payload or "cfg" not in payload:
            raise ValueError("expected a Round252 adapter checkpoint")
        cfg = dict(payload["cfg"])
        model = cls(
            base_target_anchor,
            hidden_dim=int(cfg["hidden_dim"]),
            dropout=float(cfg["dropout"]),
            max_delta_norm=float(cfg["max_delta_norm"]),
        )
        expected_anchor = base_target_anchor.detach().cpu().float()
        model.load_state_dict(payload["adapter_state_dict"], strict=True)
        if not torch.allclose(
            model.target_anchor.detach().cpu(), expected_anchor, atol=1.0e-6, rtol=0.0
        ):
            raise ValueError(
                "Round252 adapter target anchor does not match the Round246 base checkpoint"
            )
        model.eval()
        return model, payload


class Round259QueryEncoder(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, embed_dim: int, adapter_rank: int = 0
    ) -> None:
        super().__init__()
        self.query_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.query_norm = nn.LayerNorm(embed_dim)
        if adapter_rank > 0:
            self.query_adapter_down = nn.Linear(embed_dim, adapter_rank, bias=False)
            self.query_adapter_up = nn.Linear(adapter_rank, embed_dim, bias=False)
        else:
            self.query_adapter_down = None
            self.query_adapter_up = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.query_norm(self.query_encoder(x))
        if self.query_adapter_down is not None and self.query_adapter_up is not None:
            query = self.query_norm(
                query + self.query_adapter_up(F.gelu(self.query_adapter_down(query)))
            )
        return F.normalize(query, dim=1, eps=1.0e-8)


class FullSpaceScorer(nn.Module):
    """Round259 query encoder scored against its exported 16,583-gene target table."""

    def __init__(
        self,
        encoder: Round259QueryEncoder,
        targets: torch.Tensor,
        logit_scale_raw: torch.Tensor,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.register_buffer("targets", F.normalize(targets.detach().float(), dim=1))
        self.register_buffer(
            "logit_scale_raw", logit_scale_raw.detach().float().reshape(())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.exp(torch.clamp(self.logit_scale_raw, -2.0, 4.0))
        return (self.encoder(x) @ self.targets.T) * scale

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        target_values: np.ndarray,
    ) -> tuple["FullSpaceScorer", dict[str, Any]]:
        payload = load_checkpoint(path)
        state = payload.get("model_state_dict")
        if not isinstance(state, dict):
            raise ValueError("expected a full-space checkpoint with model_state_dict")
        prefix = "a01_encoder."
        query_state = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        required = [
            "query_encoder.0.weight",
            "query_encoder.4.weight",
            "query_norm.weight",
        ]
        if any(key not in query_state for key in required):
            raise ValueError("full-space checkpoint is missing query encoder parameters")
        input_dim = int(query_state["query_encoder.0.weight"].shape[1])
        hidden_dim = int(query_state["query_encoder.0.weight"].shape[0])
        embed_dim = int(query_state["query_encoder.4.weight"].shape[0])
        adapter_rank = int(
            query_state.get(
                "query_adapter_down.weight", torch.empty(0, embed_dim)
            ).shape[0]
        )
        encoder = Round259QueryEncoder(
            input_dim, hidden_dim, embed_dim, adapter_rank=adapter_rank
        )
        encoder.load_state_dict(query_state, strict=True)
        target_tensor = torch.as_tensor(target_values, dtype=torch.float32)
        if target_tensor.ndim != 2 or target_tensor.shape[1] != embed_dim:
            raise ValueError(
                f"target embedding table must have embedding dimension {embed_dim}, observed {tuple(target_tensor.shape)}"
            )
        scale_key = "h07_model.logit_scale_unseen"
        if scale_key not in state:
            raise ValueError(f"full-space checkpoint is missing {scale_key}")
        model = cls(encoder, target_tensor, state[scale_key])
        model.eval()
        return model, payload


class SingleCellModel(nn.Module):
    """Selected Round260 mean-residual-gate model for processed X128 cells."""

    def __init__(
        self, input_dim: int, hidden_dim: int, embed_dim: int, gate_hidden: int
    ) -> None:
        super().__init__()
        self.cell_encoder = nn.Module()
        self.cell_encoder.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(input_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )
        self.residual = nn.Linear(input_dim, embed_dim)
        self.beta_logit = nn.Parameter(torch.tensor(0.0))
        self.logit_scale_raw = nn.Parameter(
            torch.tensor(np.log(np.exp(10.0) - 1.0), dtype=torch.float32)
        )

    def forward_query(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.cell_encoder.net(x)
        weights = torch.softmax(self.gate(x).squeeze(-1), dim=1).unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1)
        residual = self.residual(x.mean(dim=1))
        beta = torch.sigmoid(self.beta_logit)
        return F.normalize((1.0 - beta) * pooled + beta * residual, dim=-1)

    def score(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        scale = F.softplus(self.logit_scale_raw) + 1.0e-6
        return scale * self.forward_query(x) @ F.normalize(targets, dim=1).T

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "SingleCellModel":
        state = load_checkpoint(path)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        input_dim = int(state["cell_encoder.net.0.weight"].shape[1])
        hidden_dim = int(state["cell_encoder.net.0.weight"].shape[0])
        embed_dim = int(state["cell_encoder.net.4.weight"].shape[0])
        gate_hidden = int(state["gate.0.weight"].shape[0])
        model = cls(input_dim, hidden_dim, embed_dim, gate_hidden)
        model.load_state_dict(state, strict=True)
        model.eval()
        return model


class ProteomicsResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
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


class ProteomicsEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        projection_dim: int,
        embed_dim: int,
        n_res: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res_blocks = nn.ModuleList(
            [ProteomicsResidualBlock(projection_dim, dropout) for _ in range(n_res)]
        )
        self.head = nn.Linear(projection_dim, embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))

    def forward_embed(self, x: torch.Tensor) -> torch.Tensor:
        value = self.proj(x)
        for block in self.res_blocks:
            value = block(value)
        return F.normalize(self.head(value), dim=1)


class ProteomicsLabelAdapter(nn.Module):
    def __init__(self, dim: int, mode: str, bottleneck: int = 64) -> None:
        super().__init__()
        self.mode = mode
        if mode == "identity":
            self.core = None
        elif mode == "linear":
            self.core = nn.Linear(dim, dim, bias=False)
        elif mode == "residual_mlp":
            self.down = nn.Linear(dim, bottleneck, bias=False)
            self.up = nn.Linear(bottleneck, dim, bias=False)
            self.gate = nn.Parameter(torch.tensor(-2.0))
        else:
            raise ValueError(f"unknown proteomics label adapter: {mode}")

    def forward(self, targets: torch.Tensor) -> torch.Tensor:
        if self.mode == "identity":
            return F.normalize(targets, dim=1)
        if self.mode == "linear":
            return F.normalize(self.core(targets), dim=1)
        return F.normalize(
            targets + torch.sigmoid(self.gate) * self.up(F.gelu(self.down(targets))),
            dim=1,
        )


class ProteomicsScorer(nn.Module):
    def __init__(
        self,
        encoder: ProteomicsEncoder,
        adapter: ProteomicsLabelAdapter,
        targets: torch.Tensor,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter
        self.register_buffer("targets", F.normalize(targets.detach().float(), dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.encoder.logit_scale.exp().clamp(max=100.0)
        return (self.encoder.forward_embed(x) @ self.adapter(self.targets).T) * scale

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        target_values: np.ndarray,
        *,
        dropout: float = 0.45,
    ) -> tuple["ProteomicsScorer", dict[str, Any]]:
        payload = load_checkpoint(path)
        model_state = payload.get("model_state_dict")
        adapter_state = payload.get("label_adapter_state_dict")
        if not isinstance(model_state, dict) or not isinstance(adapter_state, dict):
            raise ValueError("expected a Round261 checkpoint")
        input_dim = int(model_state["proj.0.weight"].shape[1])
        projection_dim = int(model_state["proj.0.weight"].shape[0])
        embed_dim = int(model_state["head.weight"].shape[0])
        residual_indices = {
            int(key.split(".")[1])
            for key in model_state
            if key.startswith("res_blocks.")
        }
        encoder = ProteomicsEncoder(
            input_dim, projection_dim, embed_dim, len(residual_indices), dropout
        )
        encoder.load_state_dict(model_state, strict=True)
        if "core.weight" in adapter_state:
            mode = "linear"
            bottleneck = 64
        elif "down.weight" in adapter_state:
            mode = "residual_mlp"
            bottleneck = int(adapter_state["down.weight"].shape[0])
        else:
            mode = "identity"
            bottleneck = 64
        adapter = ProteomicsLabelAdapter(embed_dim, mode, bottleneck=bottleneck)
        adapter.load_state_dict(adapter_state, strict=True)
        targets = torch.as_tensor(target_values, dtype=torch.float32)
        if targets.ndim != 2 or targets.shape[1] != embed_dim:
            raise ValueError(f"proteomics target table must have dimension {embed_dim}")
        scorer = cls(encoder, adapter, targets)
        scorer.eval()
        return scorer, payload
