#!/usr/bin/env python3
"""Assemble per-cell-line Tahoe AnnData inputs from plate observations and X128 projections."""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import zarr


DEFAULT_RAW_DIR = Path("data/tahoe_raw")
DEFAULT_X128_DIR = Path("data/tahoe_128")
DEFAULT_OUT_DIR = Path("data/processed/tahoe")
DEFAULT_WORK_DIR = Path("results/tahoe_assembly")


def normalize_cell_name(value: str) -> str:
    return str(value).strip()


def slug_cell_name(value: str) -> str:
    text = normalize_cell_name(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    if not text:
        raise ValueError(f"Cannot slug empty cell name from {value!r}")
    return text


def x128_path_for_raw(raw_path: Path, x128_dir: Path) -> Path:
    return x128_dir / f"{raw_path.stem}.X128.zarr"


def source_raw_paths(raw_dir: Path) -> list[Path]:
    files = sorted(raw_dir.glob("*.h5ad"))
    if not files:
        raise FileNotFoundError(f"No raw .h5ad files found in {raw_dir}")
    return files


def read_obs_counts(raw_path: Path) -> pd.Series:
    adata = ad.read_h5ad(raw_path, backed="r")
    try:
        if "cell_name" not in adata.obs.columns:
            raise KeyError(f"{raw_path} has no obs['cell_name']")
        vc = adata.obs["cell_name"].astype(str).value_counts(dropna=False)
        return vc
    finally:
        try:
            adata.file.close()
        except Exception:
            pass
        del adata
        gc.collect()


def count_cell_lines(raw_dir: Path, out_tsv: Path, resume: bool = True) -> pd.DataFrame:
    files = source_raw_paths(raw_dir)
    rows: list[dict] = []
    done = set()
    if resume and out_tsv.exists():
        old = pd.read_csv(out_tsv, sep="\t")
        rows.extend(old.to_dict("records"))
        done = set(old["source_file"].astype(str))

    for raw_path in files:
        if raw_path.name in done:
            continue
        print(f"[count] {time.strftime('%F %T')} {raw_path.name}", flush=True)
        vc = read_obs_counts(raw_path)
        for cell, n in vc.items():
            rows.append({"source_file": raw_path.name, "cell_name": str(cell), "n_cells": int(n)})
        df = pd.DataFrame(rows)
        out_tsv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_tsv, sep="\t", index=False)
        gc.collect()

    return pd.DataFrame(rows)


def select_top_cells(counts: pd.DataFrame, exclude: set[str], n: int) -> list[str]:
    agg = counts.groupby("cell_name", as_index=False)["n_cells"].sum()
    agg = agg[~agg["cell_name"].str.lower().isin({x.lower() for x in exclude})]
    agg = agg.sort_values(["n_cells", "cell_name"], ascending=[False, True])
    return agg.head(n)["cell_name"].tolist()


def read_plate_slice(
    raw_path: Path,
    x128_dir: Path,
    cell_name: str,
    block_size: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    zpath = x128_path_for_raw(raw_path, x128_dir)
    if not zpath.exists():
        raise FileNotFoundError(f"Missing X128 zarr for {raw_path.name}: {zpath}")
    adata = ad.read_h5ad(raw_path, backed="r")
    try:
        obs = adata.obs
        mask = obs["cell_name"].astype(str).eq(cell_name).to_numpy()
        idx = np.flatnonzero(mask).astype(np.int64)
        if idx.size == 0:
            return np.empty((0, 128), dtype=np.float32), pd.DataFrame()
        z = zarr.open(str(zpath), mode="r")
        if z.shape[0] != adata.n_obs or z.shape[1] != 128:
            raise RuntimeError(f"Shape mismatch for {raw_path.name}: raw={adata.shape}, zarr={z.shape}")
        # Tahoe rows are often interleaved by cell line. Reading one contiguous
        # run per target cell can therefore degenerate into many tiny zarr reads.
        # Instead, scan each plate in large sequential row blocks and subselect
        # the target rows in memory.
        x_parts = []
        for start in range(0, z.shape[0], block_size):
            end = min(start + block_size, z.shape[0])
            block_mask = mask[start:end]
            if not block_mask.any():
                continue
            block = np.asarray(z[start:end, :], dtype=np.float32)
            x_parts.append(block[block_mask])
        x = x_parts[0] if len(x_parts) == 1 else np.vstack(x_parts).astype(np.float32, copy=False)
        out_obs = obs.iloc[idx].copy()
        out_obs["batch"] = raw_path.stem
        if "plate" not in out_obs.columns:
            out_obs["plate"] = raw_path.stem.split("_filt_")[0]
        return x, out_obs
    finally:
        try:
            adata.file.close()
        except Exception:
            pass
        del adata
        gc.collect()


def build_cellline_h5ad(
    raw_dir: Path,
    x128_dir: Path,
    out_dir: Path,
    work_dir: Path,
    cell_name: str,
    block_size: int,
) -> dict:
    slug = slug_cell_name(cell_name)
    out_path = out_dir / f"tahoe_{slug}.h5ad"
    tmp_path = work_dir / f"tmp_tahoe_{slug}.h5ad"
    audit_path = work_dir / f"tahoe_{slug}_build_audit.json"
    if out_path.exists():
        print(f"[skip] final exists: {out_path}", flush=True)
        adata = ad.read_h5ad(out_path, backed="r")
        try:
            return {"cell_name": cell_name, "slug": slug, "out_path": str(out_path), "n_obs": int(adata.n_obs), "n_vars": int(adata.n_vars), "skipped_existing": True}
        finally:
            try:
                adata.file.close()
            except Exception:
                pass

    rows = []
    xs = []
    plate_counts: Counter[str] = Counter()
    for raw_path in source_raw_paths(raw_dir):
        print(f"[build:{cell_name}] {time.strftime('%F %T')} reading {raw_path.name}", flush=True)
        x, obs = read_plate_slice(raw_path, x128_dir, cell_name, block_size=block_size)
        if len(obs) == 0:
            continue
        xs.append(x)
        rows.append(obs)
        plate_counts[raw_path.stem] += int(len(obs))
        print(f"[build:{cell_name}] kept {len(obs)} cells from {raw_path.name}", flush=True)
        gc.collect()

    if not rows:
        raise RuntimeError(f"No rows found for cell line {cell_name}")

    X = xs[0] if len(xs) == 1 else np.vstack(xs).astype(np.float32, copy=False)
    obs_all = pd.concat(rows, axis=0)
    obs_all.index = obs_all.index.astype(str)
    var = pd.DataFrame(index=[str(i) for i in range(X.shape[1])])
    out = ad.AnnData(X=X, obs=obs_all, var=var)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    if tmp_path.exists():
        tmp_path.unlink()
    out.write_h5ad(tmp_path, compression="gzip")
    shutil.move(str(tmp_path), str(out_path))
    audit = {
        "cell_name": cell_name,
        "slug": slug,
        "out_path": str(out_path),
        "n_obs": int(out.n_obs),
        "n_vars": int(out.n_vars),
        "dtype": str(out.X.dtype),
        "raw_dir": str(raw_dir),
        "x128_dir": str(x128_dir),
        "block_size": int(block_size),
        "plate_counts": dict(plate_counts),
        "method": "raw Tahoe obs row mask + precomputed Tahoe X128 zarr; one plate at a time; block-wise sequential zarr scan",
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    del out, X, xs, rows, obs_all
    gc.collect()
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--x128-dir", type=Path, default=DEFAULT_X128_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument("--top-n", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=262144)
    parser.add_argument("--exclude-cell", action="append", default=["A549"])
    parser.add_argument("--cell", action="append", default=[])
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    counts_tsv = args.work_dir / "tahoe_cell_line_counts_by_plate.tsv"
    counts = count_cell_lines(args.raw_dir, counts_tsv, resume=True)
    agg = counts.groupby("cell_name", as_index=False)["n_cells"].sum().sort_values(["n_cells", "cell_name"], ascending=[False, True])
    agg.to_csv(args.work_dir / "tahoe_cell_line_counts_total.tsv", sep="\t", index=False)
    print("[counts] top cell lines", flush=True)
    print(agg.head(30).to_string(index=False), flush=True)

    if args.count_only:
        return

    cells = args.cell or select_top_cells(counts, set(args.exclude_cell), args.top_n)
    print(f"[selection] cells={cells}", flush=True)
    audits = []
    for cell in cells:
        audits.append(build_cellline_h5ad(
            args.raw_dir,
            args.x128_dir,
            args.out_dir,
            args.work_dir,
            cell,
            block_size=args.block_size,
        ))
    (args.work_dir / "selected_cellline_build_audits.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    print(json.dumps(audits, indent=2), flush=True)


if __name__ == "__main__":
    main()
