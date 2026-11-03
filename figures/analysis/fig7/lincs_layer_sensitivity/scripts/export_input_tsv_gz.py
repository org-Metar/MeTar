#!/usr/bin/env python
"""Export parquet layer inputs to the TSV.GZ filenames listed in the Round429 plan."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import parse_args_with_config, read_yaml, write_table


def main() -> None:
    args = parse_args_with_config("Export Round429 input parquet files to TSV.GZ")
    cfg_path = Path(args.config).resolve()
    cfg = read_yaml(cfg_path)
    out_root = cfg_path.parents[1] / cfg["outputs"]["root"]
    for layer in cfg["layers"]:
        layer_dir = out_root / "inputs" / layer
        for split, filename in cfg["splits"].items():
            src = layer_dir / f"{split}_profiles.parquet"
            dst = layer_dir / filename
            if not src.exists():
                raise FileNotFoundError(src)
            if dst.exists():
                continue
            df = pd.read_parquet(src)
            write_table(df, dst)
            print(dst, flush=True)


if __name__ == "__main__":
    main()
