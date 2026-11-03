#!/usr/bin/env python3
"""Verify pretrained public APIs against canonical article inputs and outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from benchmark_bulk import verify_full_space, verify_proteomics
from benchmark_utils import critical_hashes
from benchmark_adaptation import verify_cross_cell, verify_single_cell


Check = Callable[[argparse.Namespace], dict[str, Any]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("proteomics", "full-space", "cross-cell", "single-cell", "all"),
        default="all",
    )
    parser.add_argument("--repo", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def prepare_args(args: argparse.Namespace) -> argparse.Namespace:
    args.repo = args.repo.resolve()
    args.package = args.package.resolve()
    args.output = args.output.resolve()
    args.compat_write = args.package / "compat/home/lwn/claude/manuscript/write"
    args.round252 = (
        args.compat_write / "claude_round252_epiqgate_pseudoval_sweep_20260520"
    )
    args.round259 = args.compat_write / (
        "claude_round259_crosscell_fullspace_round252query_then_sc_proteomics_"
        "20260522"
    )
    args.archive259 = args.package / "code/original_runs/r259_261"
    args.critical_hashes = critical_hashes(args.package)
    if not args.round252.is_dir() or not args.round259.is_dir():
        raise RuntimeError(
            "article-exact compatibility layout is missing; run "
            "bash tools/prepare_prediction_runtime.sh "
            "$METAR_ARTIFACT_ROOT from the GitHub checkout first"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    return args


def git_identity(repo: Path) -> tuple[str, bool]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True
        ).strip()
    )
    return commit, dirty


def main(argv: list[str] | None = None) -> int:
    args = prepare_args(parse_args(argv))
    checks: dict[str, Check] = {
        "proteomics": verify_proteomics,
        "full-space": verify_full_space,
        "cross-cell": verify_cross_cell,
        "single-cell": verify_single_cell,
    }
    selected = checks if args.mode == "all" else {args.mode: checks[args.mode]}
    commit, dirty = git_identity(args.repo)
    report: dict[str, Any] = {
        "repo": str(args.repo),
        "git_commit": commit,
        "git_dirty": dirty,
        "package": str(args.package),
        "device": args.device,
        "interfaces": {},
    }
    report_path = args.output / f"{args.mode}_article_exact_report.json"
    for name, check in selected.items():
        report["interfaces"][name] = check(args)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["status"] = (
        "PASS"
        if all(item["status"] == "PASS" for item in report["interfaces"].values())
        else "FAIL"
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
