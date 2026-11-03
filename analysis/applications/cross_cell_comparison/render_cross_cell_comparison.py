#!/usr/bin/env python3
"""Recompute current cross-cell numeric source tables without historical artwork."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reproduction_runs/cross_cell_comparison"))
    parser.add_argument("--direct-source", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[3]
    script = repository / "figures/analysis/fig3/direct_fair_comparison/scripts/summarize_cross_cell_results.py"
    spec = importlib.util.spec_from_file_location("cross_cell_results", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    package = args.package_root.resolve()
    paths = module.input_paths(package, args.direct_source)
    if args.check_only:
        print(f"Cross-cell inputs ready: {len(paths)} tables")
        return 0
    output = args.output_dir
    if not output.is_absolute():
        output = package / output
    outputs = module.generate_sources(package, output.resolve() / "source_data", args.direct_source)
    print(f"Wrote {len(outputs)} cross-cell numeric source tables to {output / 'source_data'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
