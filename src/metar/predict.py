"""Python prediction APIs for every model-use scenario reported in the manuscript."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .io import validate_identifiers
from .models import (
    ClosedDiagReadout,
    FullSpaceScorer,
    ProteomicsScorer,
    QueryGatedSupportAdapter,
    SingleCellModel,
)
from .preprocessing import ProteomicsPreprocessor
from .result import PredictionResult
from .runtime import report_metrics, report_progress, report_step
from .runtime import resolve_device as _resolve_device


def resolve_device(value: str | torch.device) -> torch.device:
    """Retain the historical torch.device return type using shared validation."""
    return torch.device(_resolve_device(str(value)))


def _validate_batch_size(batch_size: int) -> None:
    if (
        not isinstance(batch_size, (int, np.integer))
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")


def _feature_values(features: np.ndarray, query_ids: tuple[str, ...]) -> np.ndarray:
    validate_identifiers(query_ids, label="query identifiers")
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or len(values) != len(query_ids) or not len(values):
        raise ValueError(
            "query features must be a nonempty matrix aligned to query identifiers"
        )
    if not np.isfinite(values).all():
        raise ValueError("query features must be finite")
    return values


def _target_values(
    target_names: tuple[str, ...], target_values: np.ndarray
) -> np.ndarray:
    validate_identifiers(target_names, label="target labels")
    values = np.asarray(target_values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != len(target_names) or not values.shape[1]:
        raise ValueError(
            "target labels must align to a nonempty target embedding table"
        )
    if not np.isfinite(values).all():
        raise ValueError("target embedding table must be finite")
    return values


def _batched_forward(
    model: torch.nn.Module,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    _validate_batch_size(batch_size)
    model = model.to(device)
    total_batches = (len(features) + batch_size - 1) // batch_size
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for batch_index, start in enumerate(
            range(0, len(features), batch_size), start=1
        ):
            batch = torch.as_tensor(
                features[start : start + batch_size], dtype=torch.float32, device=device
            )
            outputs.append(model(batch).detach().cpu().numpy())
            report_progress(
                batch_index,
                total_batches,
                label="Prediction batches",
                metrics={"queries": min(start + batch_size, len(features))},
            )
    return np.concatenate(outputs, axis=0)


def predict_bulk_668(
    features: np.ndarray,
    query_ids: tuple[str, ...],
    checkpoint: str | Path,
    *,
    device: str | torch.device = "auto",
    batch_size: int = 512,
) -> PredictionResult:
    """Rank the 668 article targets directly from ordered 978-feature responses."""

    _validate_batch_size(batch_size)
    values = _feature_values(features, query_ids)
    runtime = resolve_device(device)
    report_step("Load bulk checkpoint")
    model, targets, _payload = ClosedDiagReadout.from_checkpoint(checkpoint)
    expected_dim = int(model.query_encoder[0].in_features)
    if values.ndim != 2 or values.shape != (len(query_ids), expected_dim):
        raise ValueError(
            f"bulk input must have shape ({len(query_ids)}, {expected_dim}), observed {values.shape}"
        )
    report_metrics(
        {
            "queries": len(query_ids),
            "features": expected_dim,
            "targets": len(targets),
            "batch_size": batch_size,
            "device": str(runtime),
        },
        label="Prediction",
    )
    scores = _batched_forward(model, values, runtime, batch_size)
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
    device: str | torch.device = "auto",
    batch_size: int = 256,
) -> PredictionResult:
    """Apply the selected Round252 support adapter in a held-out cell context."""

    _validate_batch_size(batch_size)
    query = _feature_values(query_features, query_ids)
    runtime = resolve_device(device)
    report_step("Load bulk checkpoint and support adapter")
    base, targets, _base_payload = ClosedDiagReadout.from_checkpoint(base_checkpoint)
    adapter, adapter_payload = QueryGatedSupportAdapter.from_checkpoint(
        adapter_checkpoint, base.target_anchor
    )
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
    if not len(support):
        raise ValueError("adapted prediction requires at least one support row")
    if not np.isfinite(support).all():
        raise ValueError("support features must be finite")
    if not np.isfinite(labels).all() or np.any(labels < 0):
        raise ValueError("support labels must be finite and nonnegative")
    if np.any(labels.sum(axis=1) <= 0):
        raise ValueError(
            "every support row must have at least one positive target annotation"
        )
    if support_ids is not None:
        validate_identifiers(support_ids, label="support identifiers")
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
    base = base.to(runtime)
    adapter = adapter.to(runtime)
    shift_scale = float(adapter_payload["cfg"]["shift_scale"])
    if not np.isfinite(shift_scale):
        raise ValueError("support adapter shift_scale must be finite")
    report_metrics(
        {
            "queries": len(query_ids),
            "features": input_dim,
            "targets": len(targets),
            "support_rows": len(support),
            "batch_size": batch_size,
            "device": str(runtime),
        },
        label="Prediction",
    )
    report_step("Encode support responses")
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        support_batches: list[torch.Tensor] = []
        total_support_batches = (len(support) + batch_size - 1) // batch_size
        for batch_index, start in enumerate(
            range(0, len(support), batch_size), start=1
        ):
            support_batches.append(
                base.encode(
                    torch.as_tensor(support[start : start + batch_size], device=runtime)
                )
            )
            report_progress(
                batch_index,
                total_support_batches,
                label="Support encoding batches",
                metrics={"support_rows": min(start + batch_size, len(support))},
            )
        support_q = torch.cat(support_batches)
        support_y = torch.as_tensor(labels, device=runtime)
        support_w = torch.as_tensor(weights, device=runtime)
        total_batches = (len(query) + batch_size - 1) // batch_size
        for batch_index, start in enumerate(range(0, len(query), batch_size), start=1):
            query_q = base.encode(
                torch.as_tensor(query[start : start + batch_size], device=runtime)
            )
            delta, gate = adapter(query_q, support_q, support_y, support_w)
            adapted = F.normalize(query_q + shift_scale * gate[:, None] * delta, dim=1)
            outputs.append(base.score_embeddings(adapted).detach().cpu().numpy())
            report_progress(
                batch_index,
                total_batches,
                label="Prediction batches",
                metrics={"queries": min(start + batch_size, len(query))},
            )
    scores = np.concatenate(outputs, axis=0)
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
    device: str | torch.device = "auto",
    batch_size: int = 256,
) -> PredictionResult:
    """Rank the exported Round259 full candidate-gene space."""

    _validate_batch_size(batch_size)
    values = _feature_values(features, query_ids)
    targets = _target_values(target_names, target_values)
    runtime = resolve_device(device)
    report_step("Load full-space checkpoint and target embeddings")
    model, _payload = FullSpaceScorer.from_checkpoint(checkpoint, targets)
    expected_dim = int(model.encoder.query_encoder[0].in_features)
    if values.shape != (len(query_ids), expected_dim):
        raise ValueError(
            f"full-space input must have shape ({len(query_ids)}, {expected_dim})"
        )
    if len(target_names) != model.targets.shape[0]:
        raise ValueError("target labels do not match the target embedding table")
    report_metrics(
        {
            "queries": len(query_ids),
            "features": expected_dim,
            "targets": len(target_names),
            "batch_size": batch_size,
            "device": str(runtime),
        },
        label="Prediction",
    )
    scores = _batched_forward(model, values, runtime, batch_size)
    return PredictionResult(scores, target_names, query_ids, "bulk_full_space")


def _stable_group_subset(
    indices: np.ndarray, query_id: str, max_cells: int, seed: int
) -> np.ndarray:
    if max_cells <= 0 or len(indices) <= max_cells:
        return indices
    digest = hashlib.sha256(f"{seed}:{query_id}".encode()).digest()
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
    device: str | torch.device = "auto",
) -> PredictionResult:
    """Rank targets for grouped processed X128 cells using one selected Round260 model."""

    features = np.asarray(cell_features, dtype=np.float32)
    validate_identifiers(group_ids, label="single-cell group identifiers", unique=False)
    if features.ndim != 2 or len(features) != len(group_ids) or len(group_ids) == 0:
        raise ValueError(
            "single-cell group identifiers must align to nonempty feature rows"
        )
    if not np.isfinite(features).all():
        raise ValueError("single-cell features must be finite")
    if not isinstance(max_cells, (int, np.integer)) or isinstance(max_cells, bool):
        raise ValueError(  # noqa: TRY004 - Public input errors use the CLI's ValueError path.
            "max_cells must be an integer; nonpositive values use all cells"
        )
    target_array = _target_values(target_names, target_values)
    runtime = resolve_device(device)
    report_step("Load single-cell checkpoint and target embeddings")
    model = SingleCellModel.from_checkpoint(checkpoint)
    if features.ndim != 2 or features.shape[1] != model.cell_encoder.net[0].in_features:
        raise ValueError(
            "single-cell features do not match the selected Round260 checkpoint"
        )
    targets = torch.as_tensor(target_array, dtype=torch.float32)
    if targets.ndim != 2 or targets.shape[1] != model.residual.out_features:
        raise ValueError(
            "single-cell target table does not match the selected Round260 checkpoint"
        )
    query_ids = tuple(dict.fromkeys(group_ids))
    report_metrics(
        {
            "queries": len(query_ids),
            "cells": len(features),
            "features": features.shape[1],
            "targets": len(target_names),
            "max_cells": max_cells,
            "sampling_seed": sampling_seed,
            "device": str(runtime),
        },
        label="Prediction",
    )
    model = model.to(runtime)
    targets = targets.to(runtime)
    rows: list[np.ndarray] = []
    counts: list[int] = []
    with torch.no_grad():
        group_array = np.asarray(group_ids, dtype=str)
        for group_index, query_id in enumerate(query_ids, start=1):
            indices = np.flatnonzero(group_array == query_id)
            chosen = _stable_group_subset(indices, query_id, max_cells, sampling_seed)
            batch = torch.as_tensor(
                features[chosen][None, :, :], dtype=torch.float32, device=runtime
            )
            rows.append(model.score(batch, targets).detach().cpu().numpy()[0])
            counts.append(len(chosen))
            report_progress(
                group_index,
                len(query_ids),
                label="Single-cell groups",
                metrics={"cells_used": len(chosen)},
            )
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
    device: str | torch.device = "auto",
    batch_size: int = 64,
) -> PredictionResult:
    """Rank full-space targets from one processed protein-abundance row per query."""

    _validate_batch_size(batch_size)
    targets = _target_values(target_names, target_values)
    runtime = resolve_device(device)
    report_step("Apply locked proteomics preprocessing")
    features = _feature_values(preprocessor.transform(frame), query_ids)
    report_step("Load proteomics checkpoint and target embeddings")
    model, _payload = ProteomicsScorer.from_checkpoint(checkpoint, targets)
    if features.shape[1] != model.encoder.proj[0].in_features:
        raise ValueError(
            "proteomics preprocessor and checkpoint input dimensions differ"
        )
    if len(query_ids) != len(features):
        raise ValueError("proteomics query identifiers are not aligned to rows")
    report_metrics(
        {
            "queries": len(query_ids),
            "features": features.shape[1],
            "targets": len(target_names),
            "batch_size": batch_size,
            "device": str(runtime),
        },
        label="Prediction",
    )
    scores = _batched_forward(model, features, runtime, batch_size)
    return PredictionResult(scores, target_names, query_ids, "proteomics_full_space")
