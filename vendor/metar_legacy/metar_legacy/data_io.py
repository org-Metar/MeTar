"""Data loading helpers for MeTar release scripts."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    """Read csv/tsv/parquet tables using the file suffix."""

    p = Path(path)
    suffixes = "".join(p.suffixes).lower()
    if suffixes.endswith(".tsv") or suffixes.endswith(".tsv.gz"):
        return pd.read_csv(p, sep="\t")
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(p)
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported table suffix for {p}")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(p.suffixes).lower()
    if suffixes.endswith(".tsv") or suffixes.endswith(".tsv.gz"):
        df.to_csv(p, sep="\t", index=False)
    elif suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        df.to_csv(p, index=False)
    elif suffixes.endswith(".parquet"):
        df.to_parquet(p, index=False)
    else:
        raise ValueError(f"Unsupported table suffix for {p}")


def read_label_order(path: str | Path) -> list[str]:
    p = Path(path)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("labels") or data.get("genes") or data.get("targets")
        return [str(x) for x in data]
    df = read_table(p)
    if df.shape[1] == 1:
        return df.iloc[:, 0].astype(str).tolist()
    for col in ["gene", "target", "label", "symbol"]:
        if col in df.columns:
            return df[col].astype(str).tolist()
    raise ValueError(f"Cannot infer label column from {p}")


def split_targets(value: object) -> list[str]:
    """Parse a target-list cell from csv/tsv source data."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value if str(x)]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") or text.startswith("("):
        parsed = ast.literal_eval(text)
        return [str(x) for x in parsed if str(x)]
    for sep in ["|", ";", ","]:
        if sep in text:
            return [x.strip() for x in text.split(sep) if x.strip()]
    return [text]


def labels_to_matrix(rows: pd.DataFrame, label_order: Iterable[str], target_col: str = "true_targets") -> np.ndarray:
    labels = list(map(str, label_order))
    pos = {label: i for i, label in enumerate(labels)}
    mat = np.zeros((len(rows), len(labels)), dtype=np.float32)
    for i, value in enumerate(rows[target_col].tolist()):
        for target in split_targets(value):
            j = pos.get(str(target))
            if j is not None:
                mat[i, j] = 1.0
    return mat


def load_target_table(path: str | Path, gene_col: str | None = None) -> tuple[list[str], np.ndarray]:
    """Load a target embedding table.

    The file can either have a gene column or use the first column as the index.
    Remaining columns must be numeric embedding dimensions.
    """

    df = read_table(path)
    if gene_col and gene_col in df.columns:
        genes = df[gene_col].astype(str).tolist()
        values = df.drop(columns=[gene_col]).to_numpy(dtype=np.float32)
    elif df.columns[0].lower() in {"gene", "target", "symbol"}:
        genes = df.iloc[:, 0].astype(str).tolist()
        values = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    else:
        genes = df.iloc[:, 0].astype(str).tolist()
        values = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Target table must be two-dimensional")
    return genes, values


def collapse_score_mean(
    df: pd.DataFrame,
    label_cols: list[str],
    group_cols: list[str],
    metadata_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Average score vectors over replicate rows within a query unit."""

    metadata_cols = metadata_cols or []
    score_mean = df.groupby(group_cols, dropna=False)[label_cols].mean().reset_index()
    if metadata_cols:
        meta = df[group_cols + metadata_cols].drop_duplicates(group_cols)
        score_mean = score_mean.merge(meta, on=group_cols, how="left", validate="one_to_one")
    return score_mean
