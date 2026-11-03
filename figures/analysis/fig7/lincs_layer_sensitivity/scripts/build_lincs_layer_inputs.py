#!/usr/bin/env python
"""Build matched L3/L4/L5 training tables for the Round429 sensitivity run.

This script expects each layer directory to contain train/val/test profile
tables with identical metadata and target-label columns. It intersects rows
across all layers, applies train-fold gene-wise median imputation and standard
scaling per layer, and writes aligned tables for MeTar training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import companion_root_from_config, first_existing_column, parse_args_with_config, read_table, read_yaml, write_table


def metadata_columns(df: pd.DataFrame, cfg: dict) -> dict[str, str]:
    columns = list(df.columns)
    c = cfg["columns"]
    return {
        "profile_id": first_existing_column(columns, c["profile_id_candidates"]),
        "pert_id": first_existing_column(columns, c["pert_id_candidates"], required=False) or "",
        "drug": first_existing_column(columns, c["drug_candidates"]),
        "cell_line": first_existing_column(columns, c["cell_line_candidates"]),
        "dose": first_existing_column(columns, c["dose_candidates"], required=False) or "",
        "time": first_existing_column(columns, c["time_candidates"], required=False) or "",
        "split": first_existing_column(columns, c["split_candidates"], required=False) or "",
        "target": c["target_col"],
    }


def infer_feature_cols(df: pd.DataFrame, meta: dict[str, str], expected_count: int) -> list[str]:
    protected = {v for v in meta.values() if v}
    feature_cols = [c for c in df.columns if c not in protected and pd.api.types.is_numeric_dtype(df[c])]
    if len(feature_cols) != expected_count:
        raise ValueError(f"Expected {expected_count} numeric feature columns, found {len(feature_cols)}")
    return feature_cols


def normalized_manifest(df: pd.DataFrame, meta: dict[str, str], split_name: str, layer: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "row_key": df[meta["profile_id"]].astype(str),
            "layer": layer,
            "split": split_name,
            "sig_id": df[meta["profile_id"]].astype(str),
            "drug": df[meta["drug"]].astype(str),
            "cell_line": df[meta["cell_line"]].astype(str),
            "true_targets": df[meta["target"]].astype(str),
        }
    )
    if meta.get("pert_id"):
        out["pert_id"] = df[meta["pert_id"]].astype(str)
    if meta.get("dose"):
        out["pert_dose"] = df[meta["dose"]].astype(str)
    if meta.get("time"):
        out["pert_time"] = df[meta["time"]].astype(str)
    return out


def row_missing_fraction(df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    return df[feature_cols].isna().mean(axis=1)


def fit_scaler(train: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.Series, pd.Series]:
    median = train[feature_cols].median(axis=0)
    filled = train[feature_cols].fillna(median)
    mean = filled.mean(axis=0)
    std = filled.std(axis=0, ddof=0).replace(0, 1.0).fillna(1.0)
    return median, mean, std


def transform(df: pd.DataFrame, feature_cols: list[str], median: pd.Series, mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    out = df.copy()
    values = out[feature_cols].fillna(median)
    out[feature_cols] = (values - mean) / std
    return out


def main() -> None:
    args = parse_args_with_config("Build matched LINCS layer inputs")
    cfg_path = Path(args.config).resolve()
    cfg = read_yaml(cfg_path)
    overlay = cfg_path.parents[1]
    companion_root = companion_root_from_config(cfg)
    out_root = overlay / cfg["outputs"]["root"]
    out_root.mkdir(parents=True, exist_ok=True)

    layer_frames: dict[str, dict[str, pd.DataFrame]] = {}
    layer_features: dict[str, list[str]] = {}
    manifest_parts = []
    audit_rows = []
    splits = cfg["splits"]

    for layer, layer_cfg in cfg["layers"].items():
        layer_root = companion_root / layer_cfg["input_root"]
        if not layer_root.exists():
            raise FileNotFoundError(f"Layer input root not found: {layer_root}")
        layer_frames[layer] = {}
        feature_cols_for_layer: list[str] | None = None
        for split_name, filename in splits.items():
            table_path = layer_root / filename
            df = read_table(table_path)
            meta = metadata_columns(df, cfg)
            if meta["target"] not in df.columns:
                raise ValueError(f"{table_path}: missing target column {meta['target']}")
            feature_cols = infer_feature_cols(df, meta, int(cfg["feature_count"]))
            if feature_cols_for_layer is None:
                feature_cols_for_layer = feature_cols
            elif feature_cols != feature_cols_for_layer:
                raise ValueError(f"{layer}: feature columns differ across splits")
            missing_fraction = row_missing_fraction(df, feature_cols)
            keep = missing_fraction <= 0.05
            df = df.loc[keep].copy()
            df["_row_key"] = df[meta["profile_id"]].astype(str)
            df["_split"] = split_name
            df["_drug"] = df[meta["drug"]].astype(str)
            df["_cell_line"] = df[meta["cell_line"]].astype(str)
            df["_true_targets"] = df[meta["target"]].astype(str)
            layer_frames[layer][split_name] = df
            manifest_parts.append(normalized_manifest(df, meta, split_name, layer))
            audit_rows.append(
                {
                    "layer": layer,
                    "split": split_name,
                    "input_path": str(table_path),
                    "rows_after_missing_filter": int(len(df)),
                    "dropped_missing_gt_5pct": int((~keep).sum()),
                    "feature_count": len(feature_cols),
                    "mean_row_missing_fraction_before_filter": float(missing_fraction.mean()),
                    "n_drugs": int(df["_drug"].nunique()),
                    "n_cell_lines": int(df["_cell_line"].nunique()),
                }
            )
        layer_features[layer] = feature_cols_for_layer or []

    key_sets = []
    for layer, split_map in layer_frames.items():
        keys = set()
        for df in split_map.values():
            keys |= set(df["_row_key"].astype(str))
        key_sets.append(keys)
    common_keys = set.intersection(*key_sets)
    if not common_keys:
        raise ValueError("No common profile rows across all LINCS layers")

    manifest = pd.concat(manifest_parts, ignore_index=True)
    manifest["matched_all_layers"] = manifest["row_key"].isin(common_keys)
    write_table(manifest, out_root / cfg["outputs"]["profile_layer_manifest"])

    for layer, split_map in layer_frames.items():
        layer_out = out_root / "inputs" / layer
        layer_out.mkdir(parents=True, exist_ok=True)
        feature_cols = layer_features[layer]
        aligned = {split: df[df["_row_key"].isin(common_keys)].copy() for split, df in split_map.items()}
        median, mean, std = fit_scaler(aligned["train"], feature_cols)
        for split_name, df in aligned.items():
            scaled = transform(df, feature_cols, median, mean, std)
            keep_cols = [c for c in scaled.columns if not c.startswith("_")]
            write_table(scaled[keep_cols], layer_out / splits[split_name])
            audit_rows.append(
                {
                    "layer": layer,
                    "split": f"{split_name}_aligned_scaled",
                    "input_path": str(layer_out / splits[split_name]),
                    "rows_after_missing_filter": int(len(scaled)),
                    "dropped_missing_gt_5pct": 0,
                    "feature_count": len(feature_cols),
                    "mean_row_missing_fraction_before_filter": 0.0,
                    "n_drugs": int(scaled["_drug"].nunique()),
                    "n_cell_lines": int(scaled["_cell_line"].nunique()),
                    "common_row_count": len(common_keys),
                }
            )

    audit = pd.DataFrame(audit_rows)
    audit["all_layer_common_rows"] = len(common_keys)
    write_table(audit, out_root / cfg["outputs"]["layer_input_audit"])
    feature_file = out_root / "l1000_feature_cols.txt"
    feature_file.write_text("\n".join(next(iter(layer_features.values()))) + "\n", encoding="utf-8")
    print(out_root)


if __name__ == "__main__":
    main()
