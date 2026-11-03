#!/usr/bin/env python
"""Build Round429 LINCS layer inputs from beta GCTX files with recoverable HDF5 reads."""

from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import parse_args_with_config, read_table, read_yaml, resolve_path, write_json, write_table


META_COLS = ["sig_id", "drug", "cell_line", "pert_dose", "pert_time", "split", "n_profiles", "true_targets"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def read_hdf5_strings(dataset) -> list[str]:
    arr = dataset[:]
    return [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in arr]


class GctxMatrix:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = h5py.File(path, "r")
        self.matrix = self.handle["0/DATA/0/matrix"]
        self.cids = read_hdf5_strings(self.handle["0/META/COL/id"])
        self.rids = read_hdf5_strings(self.handle["0/META/ROW/id"])
        self.cid_to_i = {cid: i for i, cid in enumerate(self.cids)}
        self.rid_to_i = {rid: i for i, rid in enumerate(self.rids)}

    def close(self) -> None:
        self.handle.close()

    def feature_indices(self, features: list[str]) -> list[int]:
        missing = [rid for rid in features if rid not in self.rid_to_i]
        if missing:
            raise ValueError(f"{self.path}: missing feature ids: {missing[:5]}")
        return [self.rid_to_i[rid] for rid in features]

    def row_mean_for_columns(self, column_ids: list[str], feature_indices: list[int]) -> np.ndarray | None:
        col_idx = sorted(self.cid_to_i[cid] for cid in column_ids if cid in self.cid_to_i)
        if not col_idx:
            return None
        values = self.matrix[col_idx, :][:, feature_indices]
        return np.asarray(values, dtype=np.float32).mean(axis=0)

    def replicate_means_for_signatures(
        self,
        selected: pd.DataFrame,
        features: list[str],
        *,
        chunk_size: int = 8000,
    ) -> pd.DataFrame:
        feature_indices = self.feature_indices(features)
        sig_ids = selected["sig_id"].astype(str).tolist()
        sig_to_pos = {sig_id: i for i, sig_id in enumerate(sig_ids)}
        inst_to_sig_positions: dict[int, list[int]] = {}
        for sig_id, inst_ids in zip(selected["sig_id"].astype(str), selected["distil_ids_list"]):
            pos = sig_to_pos[sig_id]
            for inst_id in inst_ids:
                inst_pos = self.cid_to_i.get(inst_id)
                if inst_pos is None:
                    continue
                inst_to_sig_positions.setdefault(inst_pos, []).append(pos)

        items = sorted(inst_to_sig_positions.items())
        sums = np.zeros((len(sig_ids), len(features)), dtype=np.float32)
        counts = np.zeros(len(sig_ids), dtype=np.int32)
        for start in range(0, len(items), chunk_size):
            chunk = items[start : start + chunk_size]
            row_idx = [row for row, _ in chunk]
            data = np.asarray(self.matrix[row_idx, :], dtype=np.float32)[:, feature_indices]
            expanded_sig_pos: list[int] = []
            expanded_data_pos: list[int] = []
            for local_i, (_, sig_positions) in enumerate(chunk):
                expanded_sig_pos.extend(sig_positions)
                expanded_data_pos.extend([local_i] * len(sig_positions))
            np.add.at(sums, np.asarray(expanded_sig_pos, dtype=np.int64), data[np.asarray(expanded_data_pos, dtype=np.int64)])
            np.add.at(counts, np.asarray(expanded_sig_pos, dtype=np.int64), 1)
            log(
                f"{self.path.name}: read replicate rows "
                f"{min(start + chunk_size, len(items))}/{len(items)}; "
                f"signatures with data={int((counts > 0).sum())}/{len(sig_ids)}"
            )

        keep = counts > 0
        sums[keep] = sums[keep] / counts[keep, None]
        return pd.DataFrame(sums[keep], index=np.asarray(sig_ids, dtype=object)[keep], columns=features)

    def rows_for_signature_columns(self, sig_ids: list[str], feature_indices: list[int], block_size: int = 4096) -> pd.DataFrame:
        rows: list[np.ndarray] = []
        index: list[str] = []
        for start in range(0, len(sig_ids), block_size):
            block = sig_ids[start : start + block_size]
            present = [(sig_id, self.cid_to_i[sig_id]) for sig_id in block if sig_id in self.cid_to_i]
            if not present:
                continue
            order = np.argsort([x[1] for x in present])
            sorted_pairs = [present[i] for i in order]
            idx = [i for _, i in sorted_pairs]
            data = np.asarray(self.matrix[idx, :][:, feature_indices], dtype=np.float32)
            pair_to_row = {sig: data[j] for j, (sig, _) in enumerate(sorted_pairs)}
            for sig_id in block:
                if sig_id in pair_to_row:
                    rows.append(pair_to_row[sig_id])
                    index.append(sig_id)
            log(f"{self.path.name}: read {min(start + block_size, len(sig_ids))}/{len(sig_ids)} signature rows")
        return pd.DataFrame(rows, index=index, columns=[str(i) for i in range(len(feature_indices))])


def label_order(path: Path) -> list[str]:
    df = read_table(path)
    values = df.iloc[:, 0].astype(str).tolist() if df.shape[1] == 1 else df["gene"].astype(str).tolist()
    if values and values[0].lower() in {"gene", "target", "label"}:
        values = values[1:]
    return values


def load_drug_targets(path: Path, labels: list[str]) -> dict[str, str]:
    label_set = set(labels)
    df = pd.read_csv(path, sep="\t", usecols=["pert_id", "target"]).dropna()
    df["pert_id"] = df["pert_id"].astype(str)
    df["target"] = df["target"].astype(str).str.strip()
    df = df[(df["target"] != "") & (df["target"] != "0")].copy()
    grouped = df.groupby("pert_id")["target"].agg(lambda s: tuple(pd.unique(s))).to_dict()
    return {str(k): "|".join(v) for k, v in grouped.items() if set(map(str, v)).issubset(label_set)}


def read_bulk_profiles(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["drug", "cell_line", "method"])
    df = df[df["method"].eq(0)].reset_index().rename(columns={"cid": "sig_id"})
    if "sig_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "sig_id"})
    df["sig_id"] = df["sig_id"].astype(str)
    df["drug"] = df["drug"].astype(str)
    df["cell_line"] = df["cell_line"].astype(str)
    return df[["sig_id", "drug", "cell_line"]]


def read_siginfo_for_ids(path: Path, sig_ids: set[str], chunksize: int = 250_000) -> pd.DataFrame:
    usecols = ["sig_id", "pert_id", "pert_dose", "pert_time", "cell_iname", "distil_ids"]
    hits = []
    for chunk in pd.read_csv(path, sep="\t", usecols=usecols, chunksize=chunksize, low_memory=False):
        chunk["sig_id"] = chunk["sig_id"].astype(str)
        keep = chunk[chunk["sig_id"].isin(sig_ids)].copy()
        if len(keep):
            hits.append(keep)
        found = sum(len(x) for x in hits)
        log(f"siginfo scan: found {found}/{len(sig_ids)} selected signatures")
        if found >= len(sig_ids):
            break
    if not hits:
        raise ValueError("No selected signatures were found in siginfo_beta")
    return pd.concat(hits, ignore_index=True).drop_duplicates("sig_id")


def split_rows(bulk: pd.DataFrame, cfg: dict, labels: list[str]) -> pd.DataFrame:
    src = cfg["server_sources"]
    proto = json.loads(resolve_path(src["protocol_lock"]).read_text(encoding="utf-8"))
    official = read_table(src["official_heldout_cells"])["cell_line"].astype(str).tolist()
    drug_targets = load_drug_targets(resolve_path(src["compoundinfo_beta"]), labels)
    df = bulk.copy()
    df["true_targets"] = df["drug"].map(drug_targets)
    df = df[df["true_targets"].notna()].copy()
    conditions = [
        df["cell_line"].isin(set(proto["cell_split"]["train_cells"])) & df["drug"].isin(set(proto["drug_split"]["inner_train_drugs"])),
        df["cell_line"].isin(set(proto["cell_split"]["train_cells"])) & df["drug"].isin(set(proto["drug_split"]["val_drugs"])),
        df["cell_line"].isin(set(official)) & df["drug"].isin(set(proto["drug_split"]["test_drugs"])),
    ]
    df["split"] = np.select(conditions, ["train", "val", "test"], default="")
    df = df[df["split"] != ""].copy()
    df["n_profiles"] = 1
    return df


def ordered_feature_ids(path: Path, expected: int) -> list[str]:
    names = pq.ParquetFile(path).schema_arrow.names
    features = [name for name in names if name not in {"drug", "cell_line", "method", "cid"}]
    if len(features) != expected:
        raise ValueError(f"Expected {expected} feature columns from bulk parquet, found {len(features)}")
    return [str(x) for x in features]


def parse_distil_ids(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def fit_transform_layer(base: pd.DataFrame, features: list[str]) -> tuple[dict[str, pd.DataFrame], list[dict[str, object]]]:
    split_tables = {name: base[base["split"].eq(name)].copy() for name in ["train", "val", "test"]}
    filtered = {}
    audits = []
    for split, df in split_tables.items():
        frac = df[features].isna().mean(axis=1)
        keep = frac <= 0.05
        filtered[split] = df.loc[keep].copy()
        audits.append(
            {
                "split": split,
                "rows_before_missing_filter": int(len(df)),
                "rows_after_missing_filter": int(keep.sum()),
                "dropped_missing_gt_5pct": int((~keep).sum()),
                "mean_missing_fraction_before_filter": float(frac.mean()) if len(frac) else 0.0,
            }
        )
    med = filtered["train"][features].median(axis=0)
    x_train = filtered["train"][features].fillna(med)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0, ddof=0).replace(0, 1.0).fillna(1.0)
    out = {}
    for split, df in filtered.items():
        scaled = df.copy()
        scaled[features] = (scaled[features].fillna(med) - mean) / std
        out[split] = scaled
    return out, audits


def build_replicate_layer(layer_name: str, gctx_path: Path, selected: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    log(f"open {layer_name}: {gctx_path}")
    gctx = GctxMatrix(gctx_path)
    try:
        mat = gctx.replicate_means_for_signatures(selected, features)
        log(f"{layer_name}: retained {len(mat)}/{len(selected)} signatures")
        return mat
    finally:
        gctx.close()


def build_l5_layer(gctx_path: Path, selected: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    log(f"open L5_current_signature: {gctx_path}")
    gctx = GctxMatrix(gctx_path)
    try:
        fidx = gctx.feature_indices(features)
        mat = gctx.rows_for_signature_columns(selected["sig_id"].tolist(), fidx)
        mat.columns = features
        return mat
    finally:
        gctx.close()


def write_layer(layer: str, mat: pd.DataFrame, selected: pd.DataFrame, out_root: Path, cfg: dict, features: list[str]) -> list[dict[str, object]]:
    layer_selected = selected[selected["sig_id"].isin(mat.index)].copy()
    mat = mat.reindex(layer_selected["sig_id"].tolist())
    base = pd.concat([layer_selected[META_COLS].reset_index(drop=True), mat.reset_index(drop=True)], axis=1)
    scaled, audits = fit_transform_layer(base, features)
    layer_dir = out_root / "inputs" / layer
    for split, df in scaled.items():
        write_table(df[META_COLS + features], layer_dir / f"{split}_profiles.parquet")
        if cfg.get("export_tsv_gz_inputs", False):
            write_table(df[META_COLS + features], layer_dir / cfg["splits"][split])
    rows = []
    for audit in audits:
        df = scaled[audit["split"]]
        rows.append(
            {
                "layer": layer,
                **audit,
                "n_drugs": int(df["drug"].nunique()),
                "n_cell_lines": int(df["cell_line"].nunique()),
            }
        )
    return rows


def main() -> None:
    args = parse_args_with_config("Build Round429 LINCS beta layer inputs")
    cfg_path = Path(args.config).resolve()
    cfg = read_yaml(cfg_path)
    src = cfg["server_sources"]
    overlay = cfg_path.parents[1]
    out_root = overlay / cfg["outputs"]["root"]
    out_root.mkdir(parents=True, exist_ok=True)

    labels = label_order(resolve_path(src["label_order"]))
    if len(labels) != int(cfg["candidate_space"]):
        raise ValueError(f"Expected {cfg['candidate_space']} labels, found {len(labels)}")
    features = ordered_feature_ids(resolve_path(src["bulk_parquet"]), int(cfg["feature_count"]))
    (out_root / "l1000_feature_cols.txt").write_text("\n".join(features) + "\n", encoding="utf-8")

    log("load current bulk rows and current split")
    selected = split_rows(read_bulk_profiles(resolve_path(src["bulk_parquet"])), cfg, labels)
    log(f"selected rows before siginfo: {len(selected)}")
    siginfo = read_siginfo_for_ids(resolve_path(src["siginfo_beta"]), set(selected["sig_id"]))
    selected = selected.merge(siginfo, on="sig_id", how="left", validate="many_to_one")
    selected["pert_dose"] = selected["pert_dose"].astype(str)
    selected["pert_time"] = selected["pert_time"].astype(str)
    selected["distil_ids_list"] = selected["distil_ids"].map(parse_distil_ids)
    selected = selected[selected["distil_ids_list"].map(bool)].copy().drop_duplicates("sig_id")
    log(f"selected rows with distil ids: {len(selected)}")

    layer_paths = {
        "L5_current_signature": resolve_path(src["l5_gctx"]),
        "L4_zscore_profile": resolve_path(src["l4_gctx"]),
        "L3_normalized_expression": resolve_path(src["l3_gctx"]),
    }
    layer_mats: dict[str, pd.DataFrame] = {}
    for layer, path in layer_paths.items():
        cache = out_root / "cache" / f"{layer}_unscaled.parquet"
        if cache.exists():
            log(f"load cached {layer}: {cache}")
            layer_mats[layer] = pd.read_parquet(cache).set_index("sig_id")
            continue
        if layer == "L5_current_signature":
            mat = build_l5_layer(path, selected, features)
        else:
            mat = build_replicate_layer(layer, path, selected, features)
        cache.parent.mkdir(parents=True, exist_ok=True)
        mat.reset_index(names="sig_id").to_parquet(cache, index=False)
        layer_mats[layer] = mat
        log(f"cached {layer}: {cache} rows={len(mat)}")

    common = set(selected["sig_id"])
    for mat in layer_mats.values():
        common &= set(mat.index)
    selected = selected[selected["sig_id"].isin(common)].copy()
    selected = selected.sort_values(["split", "cell_line", "drug", "sig_id"]).reset_index(drop=True)
    log(f"common rows across L3/L4/L5: {len(selected)}")

    manifest = selected[
        ["sig_id", "drug", "cell_line", "pert_id", "pert_dose", "pert_time", "split", "n_profiles", "true_targets", "distil_ids"]
    ].copy()
    manifest["row_key"] = manifest["sig_id"]
    manifest["matched_all_layers"] = True
    write_table(manifest, out_root / cfg["outputs"]["profile_layer_manifest"])

    audit_rows = []
    for layer, mat in layer_mats.items():
        audit_rows.extend(write_layer(layer, mat, selected, out_root, cfg, features))
    audit = pd.DataFrame(audit_rows)
    audit["all_layer_common_rows"] = len(selected)
    audit["strict_test_units"] = int(selected[selected["split"].eq("test")][["drug", "cell_line"]].drop_duplicates().shape[0])
    write_table(audit, out_root / cfg["outputs"]["layer_input_audit"])

    write_json(
        {
            "bulk_parquet": str(resolve_path(src["bulk_parquet"])),
            "siginfo_beta": str(resolve_path(src["siginfo_beta"])),
            "l3_gctx": str(resolve_path(src["l3_gctx"])),
            "l4_gctx": str(resolve_path(src["l4_gctx"])),
            "l5_gctx": str(resolve_path(src["l5_gctx"])),
            "truncated_l3_download_candidate": str(resolve_path(src["truncated_l3_download_candidate"])),
            "selected_rows": int(len(selected)),
            "test_rows": int((selected["split"] == "test").sum()),
            "test_drug_cell_units": int(selected[selected["split"].eq("test")][["drug", "cell_line"]].drop_duplicates().shape[0]),
        },
        out_root / "beta_source_audit.json",
    )
    log(str(out_root))


if __name__ == "__main__":
    main()
