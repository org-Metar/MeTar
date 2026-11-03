#!/usr/bin/env python3
"""Build the article-used h07 cache from hydrated public bulk inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch

from metar.paths import resolve_source


EXPECTED_AUDIT = {
    "n_train_label_genes": 668,
    "n_holdout_val_rows_traincells": 14076,
    "n_holdout_test_rows_traincells": 30735,
    "n_genes_full_universe": 16583,
}
SOURCE_HASHES = {
    "data/processed/lincs/large_data.parquet": (
        "970db40a76f2d43a22b75703838bc8ba315ba52caf61c79946e8f1a06c849d41"
    ),
    "data/processed/lincs/compoundinfo_beta.txt": (
        "a71fca6de41dcc46a5063858be7e04155f3c09832c8b3fb35814f03db8d9fdff"
    ),
    "data/processed/fullspace/pertkge_ppi_graph.csv": (
        "8f1263018811a59498241e19c5fea0939f9ad164fca8862b6ff3cbbdb5bcfd1f"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("metar_h07_prepare", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import h07 preparation module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package = args.package.resolve()
    output = package / "data/processed/fullspace/h07_prepared_data.pt"
    if output.exists():
        raise FileExistsError(f"refusing to replace existing cache: {output}")
    source_hashes = {}
    for relative, expected in SOURCE_HASHES.items():
        path = package / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"public input hash mismatch: {relative}")
        source_hashes[relative] = observed
    protocol_path = package / "data/processed/lincs/bulk_protocol_lock_v2.json"
    repository = Path(__file__).resolve().parents[1]
    wrapper_name = "train_bulk_sharedsample_indunseenlabel_v1_20260330.py"
    base_name = "train_bulk_dualbranch_v2_20260328.py"
    data_dir = package / "data/processed/lincs"
    ppi_source = package / "data/processed/fullspace/pertkge_ppi_graph.csv"
    ppi_target = data_dir / "pertkge_ppi_graph.csv"
    if ppi_target.exists():
        raise FileExistsError(f"refusing to replace compatibility link: {ppi_target}")
    ppi_target.symlink_to(os.path.relpath(ppi_source, data_dir))
    try:
        with tempfile.TemporaryDirectory(prefix="metar_h07_prepare_") as runtime:
            os.environ["METAR_BULK_PROJECT_ROOT"] = runtime
            os.environ["METAR_BULK_PROTOCOL_PATH"] = str(protocol_path)
            scripts = Path(runtime) / "scripts"
            scripts.mkdir()
            for name in (wrapper_name, base_name):
                source = resolve_source(
                    "code/remote_dependencies/h07/" + name, repository, package
                )
                shutil.copy2(source, scripts / name)
            wrapper = import_module(scripts / wrapper_name)
            wrapper.bulk_base.DATA_DIR = data_dir
            wrapper.bulk_base.MANIFEST_PATH = protocol_path
            protocol = wrapper.bulk_base.load_protocol(protocol_path)
            data = wrapper.bulk_base.prepare_data(protocol, torch.device(args.device))
    finally:
        ppi_target.unlink(missing_ok=True)
    observed_audit = {
        key: int(data["audit"].get(key, -1)) for key in EXPECTED_AUDIT
    }
    if observed_audit != EXPECTED_AUDIT:
        raise RuntimeError(
            f"h07 preparation audit mismatch: {observed_audit} != {EXPECTED_AUDIT}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"data": data, "protocol": protocol}, output)
    report = {
        "output": str(output),
        "output_sha256": sha256_file(output),
        "canonical_file_sha256": (
            "4e7af85be1caf15265e4ec7df59b7d79d27fc2dbbda29511bff106b9f4122a07"
        ),
        "source_hashes": source_hashes,
        "audit": observed_audit,
        "status": "PASS",
    }
    audit_path = output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
