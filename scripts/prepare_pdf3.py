#!/usr/bin/env python3
"""Export the published PDF-3 response as a prediction-ready CSV example."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=os.environ.get("METAR_ARTIFACT_ROOT"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--from-raw", action="store_true",
                        help="Recompute from JHH7.csv; also requires the original geneinfo schema")
    args = parser.parse_args()
    if args.package is None:
        parser.error("Set METAR_ARTIFACT_ROOT or provide --package")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if args.from_raw:
        script = Path(__file__).resolve().parents[1] / "analysis/applications/pdf3/predict_pdf3.py"
        spec = importlib.util.spec_from_file_location("pdf3_response", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        inputs = args.package / "data/application_inputs/pdf3"
        aligned, values = module.build_aligned_query(inputs / "JHH7.csv", inputs / "geneinfo_beta.txt")
        symbols = aligned["gene_symbol"].astype(str).tolist()
        names = ["PDF-3_JHH7"]
    else:
        source = args.package / "figures/reproduction/pdf3/round291_current_668/log2fc_fpkm_pseudocount1.npz"
        if not source.is_file():
            parser.error("Missing published PDF-3 input. Download/extract result_data.tar.gz first.")
        with np.load(source, allow_pickle=False) as payload:
            values = payload["query_matrix"]
            symbols = payload["gene_symbols"].astype(str).tolist()
            names = payload["query_names"].astype(str).tolist()
    if values.shape != (1, 978) or len(set(symbols)) != 978 or not np.isfinite(values).all():
        raise ValueError("Expected one finite, ordered 978-feature PDF-3 response")
    frame = pd.DataFrame(values, columns=symbols)
    frame.insert(0, "query_id", names)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Prepared {args.output}: 1 query, 978 ordered landmark log2FC features")


if __name__ == "__main__":
    main()
