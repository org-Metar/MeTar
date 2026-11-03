"""Python prediction APIs for every model-use scenario reported in the manuscript."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .models import (
    ClosedDiagReadout,
    FullSpaceScorer,
    ProteomicsScorer,
    QueryGatedSupportAdapter,
    SingleCellModel,
)
from .preprocessing import ProteomicsPreprocessor
from .result import PredictionResult


def resolve_device(value: str | torch.device) -> torch.device:
    requested = torch.device(value)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def _batched_forward(
    model: torch.nn.Module,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model = model.to(device)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.as_tensor(
                features[start : start + batch_size], dtype=torch.float32, device=device
            )
            outputs.append(model(batch).detach().cpu().numpy())
    if not outputs:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(outputs, axis=0)


def predict_bulk_668(
    features: np.ndarray,
    query_ids: tuple[str, ...],
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 512,
) -> PredictionResult:
    """Rank the 668 article targets directly from ordered 978-feature responses."""

    model, targets, _payload = ClosedDiagReadout.from_checkpoint(checkpoint)
    values = np.asarray(features, dtype=np.float32)
    expected_dim = int(model.query_encoder[0].in_features)
    if values.ndim != 2 or values.shape != (len(query_ids), expected_dim):
        raise ValueError(
            f"bulk input must have shape ({len(query_ids)}, {expected_dim}), observed {values.shape}"
        )
    scores = _batched_forward(model, values, resolve_device(device), batch_size)
    return PredictionResult(scores, targets, query_ids, "bulk_668_direct")


def predict_cross_cell(
    query_features: np.ndarray,
    query_ids: tuple[str, ...],
    support_features: np.ndarray,
    support_labels: np.ndarray,
    base_checkpoint: str | Path,
    adapter_checkpoint: str | Path,
    *,
    support_ids: tuple[str, ...] | None = None,
    support_weights: np.ndarray | None = None,
    device: str | torch.device = "cpu",
) -> PredictionResult:
    """Apply the selected Round252 support adapter in a held-out cell context."""

    base, targets, _base_payload = ClosedDiagReadout.from_checkpoint(base_checkpoint)
    adapter, adapter_payload = QueryGatedSupportAdapter.from_checkpoint(
        adapter_checkpoint, base.target_anchor
    )
    query = np.asarray(query_features, dtype=np.float32)
    support = np.asarray(support_features, dtype=np.float32)
    labels = np.asarray(support_labels, dtype=np.float32)
    input_dim = int(base.query_encoder[0].in_features)
    if query.shape != (len(query_ids), input_dim):
        raise ValueError(
            f"query features must have shape ({len(query_ids)}, {input_dim})"
        )
    if support.ndim != 2 or support.shape[1] != input_dim:
        raise ValueError(f"support features must have shape (*, {input_dim})")
    if labels.shape != (len(support), len(targets)):
        raise ValueError(
            f"support label matrix must have shape ({len(support)}, {len(targets)})"
        )
    if support_ids is not None:
        if len(support_ids) != len(support) or len(set(support_ids)) != len(
            support_ids
        ):
            raise ValueError(
                "support identifiers must be unique and aligned to support rows"
            )
        overlap = sorted(set(query_ids).intersection(support_ids))
        if overlap:
            raise ValueError(
                f"query and support identifiers must be disjoint, overlap: {overlap[:5]}"
            )
    weights = (
        np.ones(len(support), dtype=np.float32)
        if support_weights is None
        else np.asarray(support_weights, dtype=np.float32)
    )
    if (
        weights.shape != (len(support),)
        or np.any(weights <= 0)
        or not np.isfinite(weights).all()
    ):
        raise ValueError(
            "support weights must be finite positive values aligned to support rows"
        )
    runtime = resolve_device(device)
    base = base.to(runtime)
    adapter = adapter.to(runtime)
    with torch.no_grad():
        query_q = base.encode(torch.as_tensor(query, device=runtime))
        support_q = base.encode(torch.as_tensor(support, device=runtime))
        delta, gate = adapter(
            query_q,
            support_q,
            torch.as_tensor(labels, device=runtime),
            torch.as_tensor(weights, device=runtime),
        )
        shift_scale = float(adapter_payload["cfg"]["shift_scale"])
        adapted = F.normalize(query_q + shift_scale * gate[:, None] * delta, dim=1)
        scores = base.score_embeddings(adapted).detach().cpu().numpy()
    metadata = pd.DataFrame(
        {
            "prediction_mode": "support_conditioned",
            "support_count": len(support),
            "shift_scale": shift_scale,
        },
        index=range(len(query_ids)),
    )
    return PredictionResult(
        scores, targets, query_ids, "cross_cell_support_adapted", metadata
    )


def predict_full_space(
    features: np.ndarray,
    query_ids: tuple[str, ...],
    checkpoint: str | Path,
    target_names: tuple[str, ...],
    target_values: np.ndarray,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 256,
) -> PredictionResult:
    """Rank the exported Round259 full candidate-gene space."""

    model, _payload = FullSpaceScorer.from_checkpoint(checkpoint, target_values)
    values = np.asarray(features, dtype=np.float32)
    expected_dim = int(model.encoder.query_encoder[0].in_features)
    if values.shape != (len(query_ids), expected_dim):
        raise ValueError(
            f"full-space input must have shape ({len(query_ids)}, {expected_dim})"
        )
    if len(target_names) != model.targets.shape[0]:
        raise ValueError("target labels do not match the target embedding table")
    scores = _batched_forward(model, values, resolve_device(device), batch_size)
    return PredictionResult(scores, target_names, query_ids, "bulk_full_space")


def _stable_group_subset(
    indices: np.ndarray, query_id: str, max_cells: int, seed: int
) -> np.ndarray:
    if max_cells <= 0 or len(indices) <= max_cells:
        return indices
    digest = hashlib.sha256(f"{seed}:{query_id}".encode("utf-8")).digest()
    local_seed = int.from_bytes(digest[:8], "little") % (2**32 - 1)
    rng = np.random.RandomState(local_seed)
    return np.sort(rng.choice(indices, size=max_cells, replace=False))


def predict_single_cell(
    cell_features: np.ndarray,
    group_ids: tuple[str, ...],
    checkpoint: str | Path,
    target_names: tuple[str, ...],
    target_values: np.ndarray,
    *,
    cell_line: str,
    max_cells: int = 300,
    sampling_seed: int = 20260327,
    device: str | torch.device = "cpu",
) -> PredictionResult:
    """Rank targets for grouped processed X128 cells using one selected Round260 model."""

    features = np.asarray(cell_features, dtype=np.float32)
    if features.ndim != 2 or len(features) != len(group_ids) or len(group_ids) == 0:
        raise ValueError("single-cell group identifiers must align to nonempty feature rows")
    if not np.isfinite(features).all():
        raise ValueError("single-cell features must be finite")
    model = SingleCellModel.from_checkpoint(checkpoint)
    if features.ndim != 2 or features.shape[1] != model.cell_encoder.net[0].in_features:
        raise ValueError(
            "single-cell features do not match the selected Round260 checkpoint"
        )
    targets = torch.as_tensor(target_values, dtype=torch.float32)
    if targets.ndim != 2 or targets.shape[1] != model.residual.out_features:
        raise ValueError(
            "single-cell target table does not match the selected Round260 checkpoint"
        )
    query_ids = tuple(dict.fromkeys(group_ids))
    runtime = resolve_device(device)
    model = model.to(runtime)
    targets = targets.to(runtime)
    rows: list[np.ndarray] = []
    counts: list[int] = []
    with torch.no_grad():
        group_array = np.asarray(group_ids, dtype=str)
        for query_id in query_ids:
            indices = np.flatnonzero(group_array == query_id)
            chosen = _stable_group_subset(indices, query_id, max_cells, sampling_seed)
            batch = torch.as_tensor(
                features[chosen][None, :, :], dtype=torch.float32, device=runtime
            )
            rows.append(model.score(batch, targets).detach().cpu().numpy()[0])
            counts.append(len(chosen))
    metadata = pd.DataFrame(
        {"cell_line": cell_line, "cells_used": counts, "sampling_seed": sampling_seed}
    )
    return PredictionResult(
        np.stack(rows), target_names, query_ids, "single_cell_full_space", metadata
    )


def predict_proteomics(
    frame: pd.DataFrame,
    query_ids: tuple[str, ...],
    checkpoint: str | Path,
    target_names: tuple[str, ...],
    target_values: np.ndarray,
    preprocessor: ProteomicsPreprocessor,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 64,
) -> PredictionResult:
    """Rank full-space targets from one processed protein-abundance row per query."""

    features = preprocessor.transform(frame)
    model, _payload = ProteomicsScorer.from_checkpoint(checkpoint, target_values)
    if features.shape[1] != model.encoder.proj[0].in_features:
        raise ValueError(
            "proteomics preprocessor and checkpoint input dimensions differ"
        )
    if len(query_ids) != len(features):
        raise ValueError("proteomics query identifiers are not aligned to rows")
    scores = _batched_forward(model, features, resolve_device(device), batch_size)
    return PredictionResult(scores, target_names, query_ids, "proteomics_full_space")
