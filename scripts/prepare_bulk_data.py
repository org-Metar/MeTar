#!/usr/bin/env python3
"""Build the article-used h07 cache from hydrated public bulk inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

from metar.paths import resolve_source
from metar.runtime import RunReporter, report_progress, report_step, resolve_device
from metar.training import check_training_dependencies

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


def prepare_cache(args: argparse.Namespace) -> dict:
    package = args.package.resolve()
    output = package / "data/processed/fullspace/h07_prepared_data.pt"
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace existing cache: {output}")
    source_hashes = {}
    report_step("Verify public bulk inputs")
    for index, (relative, expected) in enumerate(SOURCE_HASHES.items(), 1):
        path = package / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"public input hash mismatch: {relative}")
        source_hashes[relative] = observed
        report_progress(index, len(SOURCE_HASHES), label="Input checksums")
    protocol_path = package / "data/processed/lincs/bulk_protocol_lock_v2.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    repository = Path(__file__).resolve().parents[1]
    wrapper_name = "train_bulk_sharedsample_indunseenlabel_v1_20260330.py"
    base_name = "train_bulk_dualbranch_v2_20260328.py"
    data_dir = package / "data/processed/lincs"
    ppi_source = package / "data/processed/fullspace/pertkge_ppi_graph.csv"
    ppi_target = data_dir / "pertkge_ppi_graph.csv"
    created_compatibility = False
    if ppi_target.exists() or ppi_target.is_symlink():
        if (
            not ppi_target.is_file()
            or sha256_file(ppi_target)
            != source_hashes["data/processed/fullspace/pertkge_ppi_graph.csv"]
        ):
            raise FileExistsError(f"unexpected compatibility graph: {ppi_target}")
    else:
        try:
            ppi_target.symlink_to(os.path.relpath(ppi_source, data_dir))
        except OSError:
            # Windows may not grant symlink privileges. The input remains
            # read-only to preparation; remove only this temporary copy.
            shutil.copyfile(ppi_source, ppi_target)
        created_compatibility = True
    old_environment = {
        key: os.environ.get(key)
        for key in ("METAR_BULK_PROJECT_ROOT", "METAR_BULK_PROTOCOL_PATH")
    }
    try:
        with tempfile.TemporaryDirectory(prefix="metar_h07_prepare_") as runtime:
            os.environ["METAR_BULK_PROJECT_ROOT"] = runtime
            os.environ["METAR_BULK_PROTOCOL_PATH"] = str(protocol_path)
            scripts = Path(runtime) / "scripts"
            scripts.mkdir()
            # prepare_data writes the label partition before returning data.
            (Path(runtime) / "derived_tables").mkdir()
            for name in (wrapper_name, base_name):
                source = resolve_source(
                    "code/remote_dependencies/h07/" + name, repository, package
                )
                shutil.copy2(source, scripts / name)
            wrapper = import_module(scripts / wrapper_name)
            wrapper.bulk_base.DATA_DIR = data_dir
            wrapper.bulk_base.MANIFEST_PATH = protocol_path
            protocol = wrapper.bulk_base.load_protocol(protocol_path)
            report_step("Build the locked bulk splits, normalization and graph tensors")
            data = wrapper.bulk_base.prepare_data(protocol, torch.device(args.device))
    finally:
        if created_compatibility:
            ppi_target.unlink(missing_ok=True)
        for key, value in old_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    observed_audit = {key: int(data["audit"].get(key, -1)) for key in EXPECTED_AUDIT}
    if observed_audit != EXPECTED_AUDIT:
        raise RuntimeError(
            f"h07 preparation audit mismatch: {observed_audit} != {EXPECTED_AUDIT}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    report_step("Save validated bulk cache")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".h07_", suffix=".pt", dir=output.parent
    )
    os.close(descriptor)
    temporary_output = Path(temporary_name)
    try:
        torch.save({"data": data, "protocol": protocol}, temporary_output)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to replace existing cache: {output}")
        temporary_output.replace(output)
    finally:
        temporary_output.unlink(missing_ok=True)
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
    return report


def main() -> int:
    args = parse_args()
    try:
        args.device = resolve_device(args.device)
        check_training_dependencies("closed-bulk")
        with RunReporter(
            "Prepare bulk data",
            {"package": args.package.resolve(), "device": args.device},
        ) as progress:
            report = prepare_cache(args)
            progress.finish(**report["audit"], output=report["output"])
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
