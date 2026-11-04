"""Validated tabular and array input handling for public interfaces."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


def _input_path(path: str | Path) -> Path:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Input file does not exist: {source}")
    if source.stat().st_size == 0:
        raise ValueError(f"Input file is empty: {source}; supply a complete data file.")
    return source


def validate_identifiers(
    values: Iterable[str], *, label: str, unique: bool = True
) -> None:
    """Reject absent identifiers before string conversion can conceal missing data."""
    identifiers = tuple(values)
    if not identifiers or any(
        not isinstance(value, str) or not value.strip() for value in identifiers
    ):
        raise ValueError(f"{label} must be present and nonempty")
    if unique and len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{label} must be unique")


def _string_identifiers(values: Iterable[object], *, label: str) -> tuple[str, ...]:
    raw = pd.Series(values, dtype=object)
    if raw.isna().any():
        raise ValueError(f"{label} must be present and nonempty")
    identifiers = tuple(raw.astype(str))
    validate_identifiers(identifiers, label=label, unique=False)
    return identifiers


def read_table(path: str | Path) -> pd.DataFrame:
    source = _input_path(path)
    suffixes = "".join(source.suffixes).lower()
    if suffixes.endswith((".tsv", ".tsv.gz")):
        return pd.read_csv(source, sep="\t")
    if suffixes.endswith((".csv", ".csv.gz")):
        return pd.read_csv(source)
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(source)
    raise ValueError(f"unsupported table suffix: {source}")


def read_feature_order(path: str | Path, column: str | None = None) -> list[str]:
    source = _input_path(path)
    if source.suffix.lower() in {".txt", ".list"}:
        lines = [
            line.strip() for line in source.read_text(encoding="utf-8-sig").splitlines()
        ]
        if lines and "\t" not in lines[0] and "," not in lines[0]:
            values = [line for line in lines if line]
            validate_identifiers(values, label="feature order")
            return values
    frame = (
        read_table(source)
        if source.suffix.lower() != ".txt"
        else pd.read_csv(source, sep="\t")
    )
    if "feature_space" in frame.columns:
        frame = frame[frame["feature_space"].astype(str).str.lower().eq("landmark")]
    candidates = (
        [column]
        if column
        else ["gene_symbol", "feature", "gene", "symbol", "ensembl_id"]
    )
    selected = next(
        (name for name in candidates if name and name in frame.columns), None
    )
    if selected is None:
        if frame.shape[1] != 1:
            raise ValueError(f"cannot infer feature column from {source}")
        selected = str(frame.columns[0])
    values = list(_string_identifiers(frame[selected], label="feature order"))
    validate_identifiers(values, label="feature order")
    return values


def load_target_table(path: str | Path) -> tuple[tuple[str, ...], np.ndarray]:
    frame = read_table(path)
    if frame.shape[1] < 2:
        raise ValueError(
            "target table must contain a target column and numeric dimensions"
        )
    first = str(frame.columns[0])
    targets = _string_identifiers(frame[first], label="target labels")
    validate_identifiers(targets, label="target labels")
    values = (
        frame.iloc[:, 1:]
        .apply(pd.to_numeric, errors="raise")
        .to_numpy(dtype=np.float32)
    )
    if not np.isfinite(values).all():
        raise ValueError("target table contains nonfinite values")
    if len(set(targets)) != len(targets):
        raise ValueError("target table contains duplicate target labels")
    return targets, values


def load_feature_matrix(
    path: str | Path,
    *,
    expected_dim: int,
    id_column: str = "query_id",
    feature_order: Iterable[str] | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    source = _input_path(path)
    suffix = source.suffix.lower()
    if suffix == ".npy":
        values = np.load(source, allow_pickle=False)
        row_count = 1 if values.ndim < 2 else len(values)
        ids = tuple(f"query_{index}" for index in range(row_count))
    elif suffix == ".npz":
        with np.load(source, allow_pickle=False) as payload:
            key = next((key for key in ("x", "features") if key in payload.files), None)
            if key is None:
                raise ValueError("feature npz must contain an 'x' or 'features' array")
            values = payload[key]
            if "query_ids" in payload.files:
                if payload["query_ids"].ndim != 1:
                    raise ValueError("query_ids must be a one-dimensional array")
                ids = _string_identifiers(
                    payload["query_ids"], label="query identifiers"
                )
            else:
                ids = tuple(
                    f"query_{index}"
                    for index in range(1 if values.ndim < 2 else len(values))
                )
    else:
        frame = read_table(source)
        ids = (
            _string_identifiers(frame[id_column], label="query identifiers")
            if id_column in frame.columns
            else tuple(f"query_{index}" for index in range(len(frame)))
        )
        order = list(feature_order) if feature_order is not None else []
        if order:
            validate_identifiers(order, label="feature order")
            missing = [name for name in order if name not in frame.columns]
            if missing:
                raise ValueError(
                    f"input is missing {len(missing)} ordered features, e.g. {missing[:5]}"
                )
            values = (
                frame[order]
                .apply(pd.to_numeric, errors="raise")
                .to_numpy(dtype=np.float32)
            )
        else:
            candidates = frame.drop(columns=[id_column], errors="ignore")
            numeric = candidates.select_dtypes(include=[np.number])
            if numeric.shape[1] != expected_dim:
                raise ValueError(
                    f"expected {expected_dim} numeric features, observed {numeric.shape[1]}; "
                    "provide --feature-order when the table contains metadata"
                )
            values = numeric.to_numpy(dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != expected_dim:
        raise ValueError(
            f"expected feature matrix (*, {expected_dim}), observed {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("feature matrix contains nonfinite values")
    validate_identifiers(ids, label="query identifiers")
    if len(ids) != len(values) or len(set(ids)) != len(ids):
        raise ValueError(
            "query identifiers must be present, unique, and aligned to input rows"
        )
    return values, ids


def split_targets(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith(("[", "(")):
        parsed = ast.literal_eval(text)
        return [str(item).strip() for item in parsed if str(item).strip()]
    for separator in ("|", ";", ","):
        if separator in text:
            return [item.strip() for item in text.split(separator) if item.strip()]
    return [text]


def load_support_labels(
    path: str | Path,
    support_ids: tuple[str, ...],
    targets: tuple[str, ...],
    *,
    id_column: str = "query_id",
    target_column: str = "targets",
) -> np.ndarray:
    frame = read_table(path)
    if id_column not in frame.columns or target_column not in frame.columns:
        raise ValueError(
            f"support labels require columns {id_column!r} and {target_column!r}"
        )
    label_ids = _string_identifiers(frame[id_column], label="support label identifiers")
    validate_identifiers(label_ids, label="support label identifiers")
    mapping = dict(zip(label_ids, frame[target_column]))
    missing = [query_id for query_id in support_ids if query_id not in mapping]
    if missing:
        raise ValueError(f"support annotations missing for {missing[:5]}")
    target_to_index = {target: index for index, target in enumerate(targets)}
    labels = np.zeros((len(support_ids), len(targets)), dtype=np.float32)
    for row, query_id in enumerate(support_ids):
        unknown: list[str] = []
        for target in split_targets(mapping[query_id]):
            index = target_to_index.get(target)
            if index is None:
                unknown.append(target)
            else:
                labels[row, index] = 1.0
        if unknown:
            raise ValueError(
                f"support {query_id!r} contains targets outside the 668-target space: {unknown[:5]}"
            )
        if labels[row].sum() == 0:
            raise ValueError(f"support {query_id!r} has no valid target annotation")
    return labels


def load_grouped_cells(
    path: str | Path,
    *,
    group_column: str = "query_id",
    expected_dim: int = 128,
) -> tuple[np.ndarray, tuple[str, ...]]:
    source = _input_path(path)
    if source.suffix.lower() == ".h5ad":
        try:
            import anndata as ad
        except ImportError as exc:
            raise RuntimeError(
                "install metar-reproducibility[single-cell] to read h5ad files"
            ) from exc
        adata = ad.read_h5ad(source)
        if group_column not in adata.obs:
            raise ValueError(f"h5ad obs is missing {group_column!r}")
        raw = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        values = np.asarray(raw, dtype=np.float32)
        groups = _string_identifiers(
            adata.obs[group_column], label="single-cell group identifiers"
        )
    elif source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as payload:
            feature_key = next(
                (key for key in ("x", "features") if key in payload.files), None
            )
            group_key = next(
                (key for key in ("query_ids", "groups") if key in payload.files), None
            )
            if feature_key is None or group_key is None:
                raise ValueError(
                    "single-cell npz requires x/features and query_ids/groups arrays"
                )
            values = np.asarray(payload[feature_key], dtype=np.float32)
            if payload[group_key].ndim != 1:
                raise ValueError(
                    "single-cell group identifiers must be a one-dimensional array"
                )
            groups = _string_identifiers(
                payload[group_key], label="single-cell group identifiers"
            )
    else:
        frame = read_table(source)
        if group_column not in frame.columns:
            raise ValueError(f"single-cell table is missing {group_column!r}")
        groups = _string_identifiers(
            frame[group_column], label="single-cell group identifiers"
        )
        numeric = frame.drop(columns=[group_column]).select_dtypes(include=[np.number])
        values = numeric.to_numpy(dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != expected_dim:
        raise ValueError(
            f"expected single-cell matrix (*, {expected_dim}), observed {values.shape}"
        )
    if len(values) != len(groups) or not np.isfinite(values).all():
        raise ValueError(
            "single-cell features and group identifiers are invalid or misaligned"
        )
    validate_identifiers(groups, label="single-cell group identifiers", unique=False)
    return values, groups
