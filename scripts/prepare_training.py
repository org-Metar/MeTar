#!/usr/bin/env python3
"""Prepare a private MeTar training workspace for a full run or one stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from metar.training_workspace import (
    SINGLE_CELL_FILES,
    TRAINING_STAGES,
    prepare_training_workspace,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--code-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--stage", choices=TRAINING_STAGES, default="all")
    parser.add_argument("--cell-line", choices=tuple(SINGLE_CELL_FILES))
    parser.add_argument(
        "--reuse-pretrained",
        action="store_true",
        help="Copy only this stage's required upstream weights or target embeddings",
    )
    parser.add_argument(
        "--copy-inputs",
        action="store_true",
        help="Copy large inputs instead of symlinking; useful on Windows without symlink privileges",
    )
    args = parser.parse_args()
    report = prepare_training_workspace(
        args.code_root,
        args.artifacts,
        args.output,
        check_only=args.check_only,
        copy_inputs=args.copy_inputs,
        stage=args.stage,
        reuse_pretrained=args.reuse_pretrained,
        cell_line=args.cell_line,
    )
    print(json.dumps(report, indent=2))
    return 2 if report["missing_inputs"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
