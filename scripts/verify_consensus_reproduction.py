#!/usr/bin/env python3
"""Recompute the selected multi-cell consensus scores and scientific metrics."""

from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
CURRENT_METHODS = {"cell_zscore_mean", "single_cell_mean"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Downloaded artifact root")
    parser.add_argument("--output", type=Path, default=Path("results/consensus_reproduction"))
    return parser.parse_args()


def decompressed_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_metric_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "mode" not in reader.fieldnames or "method" not in reader.fieldnames:
            raise RuntimeError(f"Missing mode/method columns: {path}")
        rows = [row for row in reader if row["mode"] == "adapted" and row["method"] in CURRENT_METHODS]
    return list(reader.fieldnames), rows


def recompute(source: Path, label_order: Path, scratch: Path) -> str:
    """Run the selected calculation against the published score directory."""
    import yaml

    directory = REPOSITORY / "figures/analysis/fig7/multicell_consensus"
    script = directory / "scripts/run_multicell_consensus.py"
    config = yaml.safe_load((directory / "configs/multicell_consensus.yaml").read_text(encoding="utf-8"))
    config["code_release"] = {"local_root": str(REPOSITORY / "vendor/metar_legacy")}
    config["round429"]["local_root"] = str(source)
    config["round429"]["server_root"] = str(source)
    config["label_order"] = {"local_path": str(label_order)}
    output = scratch / "calculated"
    config["outputs"]["root"] = str(output)
    configs = scratch / "configs"
    configs.mkdir()
    config_path = configs / "consensus.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    # The published archive stores each seed directly below scores/, while the
    # original runner also supports the training run's nested score layout.
    previous_paths = sys.path[:]
    previous_argv = sys.argv[:]
    previous_common = sys.modules.pop("common", None)
    captured = io.StringIO()
    try:
        sys.path.insert(0, str(directory / "scripts"))
        spec = importlib.util.spec_from_file_location("metar_consensus_calculation", script)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load calculation: {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.score_path = lambda cfg, root, mode, seed: root / "scores" / f"seed_{seed}" / cfg["modes"][mode]
        sys.argv = [str(script), "--config", str(config_path)]
        with contextlib.redirect_stdout(captured):
            module.main()
    finally:
        sys.path[:] = previous_paths
        sys.argv[:] = previous_argv
        sys.modules.pop("common", None)
        if previous_common is not None:
            sys.modules["common"] = previous_common
    return captured.getvalue()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    output.mkdir(parents=True)
    figure_root = root / "figures/reproduction/fig7"
    source = figure_root / "round429_L5_current_signature"
    expected_root = figure_root / "round430_outputs"
    label_order = root / "data/processed/lincs/r429_runtime/seen_668_genes.tsv"
    report: dict[str, object] = {"status": "running", "score_tables": [], "metric_tables": []}
    try:
        for required in (source, expected_root, label_order):
            if not required.exists():
                raise FileNotFoundError(required)
        with tempfile.TemporaryDirectory(prefix="metar_consensus_") as directory:
            scratch = Path(directory)
            report["calculation_output"] = recompute(source, label_order, scratch)[-4000:]
            calculated = scratch / "calculated"
            score_rows = []
            for seed in (1, 2, 3):
                for method in ("raw_score_mean", "cell_zscore_mean"):
                    rel = Path("scores/drug_level/adapted") / f"seed_{seed}" / f"{method}_drug_scores.tsv.gz"
                    actual = calculated / rel
                    expected = expected_root / rel
                    if not actual.is_file() or not expected.is_file():
                        raise FileNotFoundError(f"Missing score comparison pair: {rel}")
                    actual_sha = decompressed_sha256(actual)
                    expected_sha = decompressed_sha256(expected)
                    if actual_sha != expected_sha:
                        raise RuntimeError(f"Consensus score content differs: {rel}")
                    score_rows.append({"path": rel.as_posix(), "decompressed_sha256": actual_sha})
            report["score_tables"] = score_rows
            metric_rows = []
            for name in ("multicell_consensus_per_drug.tsv", "multicell_consensus_summary.tsv"):
                actual_fields, actual_rows = current_metric_rows(calculated / "metrics" / name)
                expected_fields, expected_rows = current_metric_rows(expected_root / "metrics" / name)
                if actual_fields != expected_fields or actual_rows != expected_rows:
                    raise RuntimeError(f"Consensus scientific metric values differ: {name}")
                metric_rows.append({"path": name, "rows": len(actual_rows)})
            report["metric_tables"] = metric_rows
        report["status"] = "pass"
        return 0
    except Exception as exc:
        report["status"] = "fail"
        report["error"] = str(exc)
        raise
    finally:
        (output / "reproduction.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
