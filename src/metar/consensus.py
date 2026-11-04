"""Prediction-time multi-cell-line consensus used by the article's Figure 7."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .io import _string_identifiers, validate_identifiers
from .result import PredictionResult
from .runtime import report_metrics, report_progress, report_step


def zscore_vector(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if not vector.size or not np.isfinite(vector).all():
        raise ValueError("z-score input must be nonempty and finite")
    standard_deviation = float(vector.std())
    if not math.isfinite(standard_deviation) or standard_deviation < 1.0e-12:
        return np.zeros_like(vector)
    return (vector - float(vector.mean())) / standard_deviation


def consensus_from_long(
    frame: pd.DataFrame,
    *,
    method: str = "cell_zscore_mean",
    drug_column: str = "query_id",
    cell_line_column: str = "cell_line",
    min_cell_lines: int = 2,
    target_order: tuple[str, ...] | None = None,
) -> PredictionResult:
    """Aggregate long-format MeTar scores into one target ranking per drug."""

    required = {drug_column, cell_line_column, "target", "score"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"consensus input is missing columns: {missing}")
    if (
        not isinstance(min_cell_lines, (int, np.integer))
        or isinstance(min_cell_lines, bool)
        or min_cell_lines < 1
    ):
        raise ValueError("min_cell_lines must be positive")
    if method not in {"cell_zscore_mean", "raw_score_mean"}:
        raise ValueError("method must be cell_zscore_mean or raw_score_mean")
    if frame.empty:
        raise ValueError("consensus input must contain score rows")
    if frame[list(required)].isna().any().any():
        raise ValueError("consensus identifiers, targets, and scores must be present")
    if not np.isfinite(pd.to_numeric(frame["score"], errors="raise")).all():
        raise ValueError("consensus scores must be finite")
    frame = frame.copy()
    for column in (drug_column, cell_line_column, "target"):
        frame[column] = _string_identifiers(frame[column], label=f"consensus {column}")
    frame["score"] = pd.to_numeric(frame["score"], errors="raise")
    duplicate = frame.duplicated([drug_column, cell_line_column, "target"])
    if duplicate.any():
        raise ValueError("consensus input contains duplicate drug-cell-target scores")
    if "candidate_count" in frame.columns:
        declared = pd.to_numeric(frame["candidate_count"], errors="raise")
        if (
            declared.isna().any()
            or not np.isfinite(declared).all()
            or not declared.gt(0).all()
            or not declared.eq(np.floor(declared)).all()
        ):
            raise ValueError(
                "candidate_count must contain positive integers for every score"
            )
        candidate_counts = declared.astype(int).unique()
        if len(candidate_counts) != 1:
            raise ValueError("consensus inputs declare different candidate counts")
        observed_per_unit = frame.groupby([drug_column, cell_line_column])[
            "target"
        ].nunique()
        if not observed_per_unit.eq(int(candidate_counts[0])).all():
            raise ValueError(
                "consensus requires complete score vectors; at least one input appears truncated by --top-k"
            )
    observed_targets = tuple(frame["target"].astype(str).drop_duplicates())
    targets = observed_targets if target_order is None else tuple(target_order)
    validate_identifiers(targets, label="canonical target labels")
    if len(set(targets)) != len(targets) or set(targets) != set(observed_targets):
        raise ValueError(
            "canonical target order does not match the consensus input target set"
        )
    rows: list[np.ndarray] = []
    query_ids: list[str] = []
    metadata: list[dict[str, object]] = []
    groups = frame.groupby(drug_column, sort=True, dropna=False)
    report_step("Aggregate complete scores across cell lines")
    report_metrics(
        {
            "input_rows": len(frame),
            "queries": len(groups),
            "targets": len(targets),
            "method": method,
            "min_cell_lines": min_cell_lines,
        },
        label="Consensus",
    )
    skipped = 0
    for group_index, (drug, group) in enumerate(groups, start=1):
        cell_lines = tuple(sorted(group[cell_line_column].astype(str).unique()))
        if len(cell_lines) < min_cell_lines:
            skipped += 1
            report_progress(
                group_index,
                len(groups),
                label="Consensus groups",
                metrics={"eligible": len(rows), "skipped": skipped},
            )
            continue
        vectors: list[np.ndarray] = []
        for cell_line in cell_lines:
            cell = group[group[cell_line_column].astype(str).eq(cell_line)].set_index(
                "target"
            )
            if set(cell.index.astype(str)) != set(targets):
                raise ValueError(f"target space differs across cell lines for {drug!r}")
            vector = cell.loc[list(targets), "score"].to_numpy(dtype=np.float64)
            vectors.append(vector)
        matrix = np.vstack(vectors)
        if method == "cell_zscore_mean":
            combined = np.vstack([zscore_vector(vector) for vector in matrix]).mean(
                axis=0
            )
        elif method == "raw_score_mean":
            combined = matrix.mean(axis=0)
        else:
            raise ValueError("method must be cell_zscore_mean or raw_score_mean")
        rows.append(combined)
        query_ids.append(str(drug))
        metadata.append(
            {
                "drug": str(drug),
                "n_cell_lines": len(cell_lines),
                "cell_lines": "|".join(cell_lines),
                "aggregation_method": method,
            }
        )
        report_progress(
            group_index,
            len(groups),
            label="Consensus groups",
            metrics={"eligible": len(rows), "skipped": skipped},
        )
    if not rows:
        raise ValueError(f"no drugs have at least {min_cell_lines} cell lines")
    return PredictionResult(
        np.stack(rows),
        targets,
        tuple(query_ids),
        "multi_cell_consensus",
        pd.DataFrame(metadata),
    )
