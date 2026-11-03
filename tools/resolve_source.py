#!/usr/bin/env python3
"""Locate a source file for the compatibility scripts using only the stdlib."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metar.paths import resolve_source

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repository", type=Path, required=True)
parser.add_argument("--artifacts", type=Path, required=True)
parser.add_argument("relative")
args = parser.parse_args()
try:
    print(resolve_source(args.relative, args.repository, args.artifacts))
except (ValueError, FileNotFoundError) as exc:
    parser.exit(1, f"{exc}\n")
