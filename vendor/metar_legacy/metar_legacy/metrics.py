"""Ranking metrics used by MeTar analyses.

The manuscript evaluates multi-label target retrieval. Each query unit can have
one or more annotated positives, and Hit@K is an any-hit endpoint: it is 1 when
at least one positive target is ranked in the first K candidates.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_KS = (1, 5, 10, 20, 30, 50)


def _as_1d_float(scores: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _as_1d_bool(labels: Sequence[int | bool | float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(labels).reshape(-1)
    return arr.astype(bool)


def descending_order(scores: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return target indices sorted by decreasing score with deterministic ties."""

    return np.argsort(-_as_1d_float(scores), kind="mergesort")


def positive_indices(labels: Sequence[int | bool | float] | np.ndarray) -> np.ndarray:
    return np.flatnonzero(_as_1d_bool(labels))


def hit_at_k(scores: Sequence[float] | np.ndarray, labels: Sequence[int | bool | float] | np.ndarray, k: int) -> float:
    """Any-positive Hit@K for a single query unit."""

    y = _as_1d_bool(labels)
    if y.sum() == 0:
        return float("nan")
    k = max(0, min(int(k), y.size))
    if k == 0:
        return 0.0
    top = descending_order(scores)[:k]
    return float(bool(y[top].any()))


def best_true_rank(scores: Sequence[float] | np.ndarray, labels: Sequence[int | bool | float] | np.ndarray) -> int | None:
    """Best 1-based rank among positive labels; None when no positive exists."""

    y = _as_1d_bool(labels)
    if y.sum() == 0:
        return None
    order = descending_order(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    return int(ranks[y].min())


def best_true_percentile(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[int | bool | float] | np.ndarray,
    *,
    higher_is_better: bool = True,
) -> float:
    """Best positive-target percentile.

    When ``higher_is_better`` is true, the top-ranked item has a percentile close
    to 100 and a top-10 item in a 668-label space has percentile >= 98.5. This is
    the convention used for the seen-label rank-distribution panels. When false,
    the value is the rank percentile from the top, used for native-universe
    full-space summaries where lower values indicate earlier recovery.
    """

    rank = best_true_rank(scores, labels)
    if rank is None:
        return float("nan")
    n = len(_as_1d_float(scores))
    if higher_is_better:
        return float(100.0 * (1.0 - rank / n))
    return float(100.0 * rank / n)


def top_percentile_recovery(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[int | bool | float] | np.ndarray,
    threshold: float,
) -> float:
    """Any-positive recovery within the top fraction of the ranked universe."""

    y = _as_1d_bool(labels)
    if y.sum() == 0:
        return float("nan")
    n = y.size
    frac = float(threshold)
    if frac > 1.0:
        frac = frac / 100.0
    cutoff = max(1, int(math.ceil(n * frac)))
    return hit_at_k(scores, y, cutoff)


def dcg_at_k(relevance: Sequence[int | bool | float] | np.ndarray, k: int) -> float:
    rel = np.asarray(relevance, dtype=np.float64).reshape(-1)[: int(k)]
    if rel.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, rel.size + 2, dtype=np.float64))
    return float(np.sum(rel / discounts))


def ndcg_at_k(scores: Sequence[float] | np.ndarray, labels: Sequence[int | bool | float] | np.ndarray, k: int) -> float:
    y = _as_1d_bool(labels).astype(np.float64)
    if y.sum() == 0:
        return float("nan")
    order = descending_order(scores)
    observed = dcg_at_k(y[order], k)
    ideal = dcg_at_k(np.sort(y)[::-1], k)
    return float(observed / ideal) if ideal > 0 else float("nan")


def average_precision(scores: Sequence[float] | np.ndarray, labels: Sequence[int | bool | float] | np.ndarray) -> float:
    """Full-ranking average precision for binary labels."""

    y = _as_1d_bool(labels)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = descending_order(scores)
    ranked = y[order]
    hit_cum = np.cumsum(ranked)
    ranks = np.arange(1, ranked.size + 1)
    precision_at_hits = hit_cum[ranked] / ranks[ranked]
    return float(np.mean(precision_at_hits))


def auroc(scores: Sequence[float] | np.ndarray, labels: Sequence[int | bool | float] | np.ndarray) -> float:
    """AUROC with 0.5 returned for single-class rows."""

    s = _as_1d_float(scores)
    y = _as_1d_bool(labels)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)
        ranks[start:end] = avg_rank
        start = end
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    pos_rank_sum = float(original_ranks[y].sum())
    u_stat = pos_rank_sum - n_pos * (n_pos + 1) / 2.0
    return float(u_stat / (n_pos * n_neg))


def any_hit_random_probability(universe_size: int, n_positive: int, cutoff_k: int) -> float:
    """Random any-hit probability for multi-label target retrieval."""

    n = int(universe_size)
    m = int(n_positive)
    k = max(0, min(int(cutoff_k), n))
    if n <= 0 or m <= 0:
        return float("nan")
    if k == 0:
        return 0.0
    if k >= n:
        return 1.0
    prob_miss = 1.0
    for i in range(k):
        prob_miss *= (n - m - i) / (n - i)
    return float(1.0 - prob_miss)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""

    n = int(total)
    x = int(successes)
    if n <= 0:
        return (float("nan"), float("nan"))
    p = x / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))


@dataclass(frozen=True)
class UnitMetricConfig:
    ks: tuple[int, ...] = DEFAULT_KS
    best_percentile_higher_is_better: bool = True


def compute_unit_metrics(
    score_matrix: np.ndarray,
    label_matrix: np.ndarray,
    row_metadata: pd.DataFrame | None = None,
    *,
    config: UnitMetricConfig | None = None,
) -> pd.DataFrame:
    """Compute retrieval metrics for each query unit."""

    cfg = config or UnitMetricConfig()
    scores = np.asarray(score_matrix, dtype=np.float64)
    labels = np.asarray(label_matrix).astype(bool)
    if scores.shape != labels.shape:
        raise ValueError(f"score/label shape mismatch: {scores.shape} != {labels.shape}")

    rows: list[dict[str, object]] = []
    for i in range(scores.shape[0]):
        rec: dict[str, object] = {}
        if row_metadata is not None:
            rec.update(row_metadata.iloc[i].to_dict())
        y = labels[i]
        rec["n_true_targets"] = int(y.sum())
        for k in cfg.ks:
            rec[f"Hit@{k}"] = hit_at_k(scores[i], y, k)
        rec["AUROC"] = auroc(scores[i], y)
        rec["NDCG@10"] = ndcg_at_k(scores[i], y, 10)
        rec["MeanAP"] = average_precision(scores[i], y)
        rec["best_true_target_rank"] = best_true_rank(scores[i], y)
        rec["best_true_target_percentile"] = best_true_percentile(
            scores[i],
            y,
            higher_is_better=cfg.best_percentile_higher_is_better,
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_metric_table(
    per_unit: pd.DataFrame,
    *,
    metric_columns: Iterable[str] | None = None,
    group_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Average metric columns, optionally by groups."""

    metrics = list(metric_columns or [f"Hit@{k}" for k in DEFAULT_KS] + ["AUROC", "NDCG@10", "MeanAP"])
    groups = list(group_cols or [])
    if not groups:
        row = {"n_units": int(len(per_unit))}
        row.update({m: float(per_unit[m].mean()) for m in metrics if m in per_unit})
        return pd.DataFrame([row])
    out = []
    for keys, grp in per_unit.groupby(groups, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groups, keys))
        row["n_units"] = int(len(grp))
        row.update({m: float(grp[m].mean()) for m in metrics if m in grp})
        out.append(row)
    return pd.DataFrame(out)
