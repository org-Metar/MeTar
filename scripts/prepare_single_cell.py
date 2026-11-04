#!/usr/bin/env python3
"""Reuse the archived Tahoe SVD projection; never fit or normalize raw .X."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import zipfile
from pathlib import Path

ASSET_SHA256 = {
    "hvg_selection.npz": "46e490db54a019554a9d6a57bd5cd2b4bcaa2f3bedd1f19724b8486f6d32f606",
    "tsvd_model.npz": "b6ff2927bdf1f053f5c54a4e8046b77f8bc99d74ff81568e0ea4fc72640e59c3",
}
ORDERED_VAR_SHA256 = "60884d016809b8b70e85e8ea98420e0dc332138099a8e69154d60d2b6823a18d"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def npz_headers(path: Path) -> dict:
    """Read NPY headers, without unpickling HVGs or reading component arrays."""
    headers = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            with archive.open(name) as handle:
                prefix = handle.read(8)
                if prefix[:6] != b"\x93NUMPY" or prefix[6] not in (1, 2, 3):
                    raise ValueError(f"Unsupported NPY header: {path}:{name}")
                length = int.from_bytes(handle.read(2 if prefix[6] == 1 else 4), "little")
                if length > 65536:
                    raise ValueError(f"Excessive NPY header: {path}:{name}")
                headers[name.removesuffix(".npy")] = ast.literal_eval(
                    handle.read(length).decode("utf-8" if prefix[6] == 3 else "latin1").strip()
                )
    return headers


def validate_assets(assets: Path) -> dict:
    headers = {}
    for name, expected in ASSET_SHA256.items():
        path = assets / name
        if sha256_file(path) != expected:
            raise ValueError(f"Archived asset SHA-256 mismatch: {path}")
        headers[name] = npz_headers(path)
    model = headers["tsvd_model.npz"]
    if model["components_"]["shape"] != (128, 6000):
        raise ValueError("The selected projection must have 128 x 6000 components")
    if model["hvg_names"]["shape"] != (6000,) or model["hvg_names"]["descr"] != "<U15":
        raise ValueError("Expected the selected model's non-pickled HVG names")
    return headers


def validate_gene_order(names: list[str]) -> None:
    digest = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
    if len(names) != 62710 or digest != ORDERED_VAR_SHA256:
        raise ValueError(
            "Raw var_names differ from the archived 62,710-gene ordered schema; "
            "the saved SVD projection requires the original column order"
        )


def inspect_plate(path: Path) -> dict:
    """Read only var index and HDF5 shape metadata; never access .X values."""
    import h5py

    with h5py.File(path, "r") as handle:
        var = handle["var"]
        key = var.attrs.get("_index", "_index")
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in var[key][()]]
        validate_gene_order(names)
        matrix = handle["X"]
        encoding = matrix.attrs.get("encoding-type", "")
        if isinstance(encoding, bytes):
            encoding = encoding.decode()
        if not isinstance(matrix, h5py.Group) or encoding != "csr_matrix":
            raise ValueError(f"{path}: raw .X must use the original CSR sparse encoding")
        if not {"data", "indices", "indptr"}.issubset(matrix.keys()):
            raise ValueError(f"{path}: incomplete CSR sparse layout")
        shape = tuple(matrix.attrs["shape"])
        if len(shape) != 2 or shape[1] != len(names):
            raise ValueError(f"Invalid .X shape metadata in {path}: {shape}")
        return {"path": str(path), "shape": [int(x) for x in shape], "encoding": encoding}


def project_block(matrix, components):
    """Equivalent to the archived reloaded TruncatedSVD.transform operation."""
    import numpy as np
    from scipy import sparse

    matrix = sparse.csr_matrix(matrix)
    result = np.asarray(matrix @ components.T, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite values in projected X128 features")
    return result


def transform_plates(files: list[Path], assets: Path, output: Path, batch_rows: int) -> None:
    import anndata as ad
    import numpy as np
    import zarr
    from numcodecs import Blosc
    from scipy import sparse

    # hvg_selection.npz contains pickled object names. The identical fitted model
    # stores names as Unicode, so no object deserialization is necessary here.
    with np.load(assets / "tsvd_model.npz", allow_pickle=False) as state:
        hvg_names = state["hvg_names"]
        components = state["components_"].astype(np.float64)
    output.mkdir(parents=True, exist_ok=False)
    for path in files:
        dataset = ad.read_h5ad(path, backed="r")
        try:
            validate_gene_order(dataset.var_names.astype(str).tolist())
            # Preserve the ORIGINAL mask order, not the variance-ranked HVG order.
            keep = np.isin(dataset.var_names.to_numpy().astype(str), hvg_names)
            if int(keep.sum()) != components.shape[1]:
                raise ValueError(f"{path}: expected 6,000 selected genes")
            destination = output / f"{path.stem}.X128.zarr"
            array = zarr.create(
                shape=(dataset.n_obs, 128), chunks=(20000, 128), dtype="float32",
                store=zarr.DirectoryStore(str(destination)),
                compressor=Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE),
                overwrite=False,
            )
            for start in range(0, dataset.n_obs, batch_rows):
                stop = min(start + batch_rows, dataset.n_obs)
                block = sparse.csr_matrix(dataset.X[start:stop, :])[:, keep]
                array[start:stop, :] = project_block(block, components)
            metadata = {
                "source_file": str(path), "n_rows": int(dataset.n_obs),
                "n_components": 128, "chunk_rows": 20000,
                "compressor": "blosc+zstd", "global_hvg": 6000,
                "model_path": str(assets / "tsvd_model.npz"), "hvg_list_saved": True,
                "method": "TruncatedSVD(randomized)",
                "ordered_var_sha256": ORDERED_VAR_SHA256,
                "asset_sha256": ASSET_SHA256, "normalization": "none; use original .X",
                "component_precision": "archived float32 promoted to float64 before dot",
            }
            (output / f"{path.stem}.X128.meta.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
        finally:
            dataset.file.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=50000)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_rows < 1:
        parser.error("--batch-rows must be positive")
    files = sorted(args.input_dir.glob("*.h5ad"))
    if not files:
        parser.error("--input-dir contains no .h5ad files")
    headers = validate_assets(args.assets)
    plates = [inspect_plate(path) for path in files]
    report = {
        "status": "compatible_metadata", "plates": plates, "asset_headers": headers,
        "check_only": args.check_only, "matrix_values_read": False,
        "output_dir": str(args.output_dir),
    }
    if not args.check_only:
        transform_plates(files, args.assets, args.output_dir, args.batch_rows)
        report.update(status="transformed", matrix_values_read=True)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
