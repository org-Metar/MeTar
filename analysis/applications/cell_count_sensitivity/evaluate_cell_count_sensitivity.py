#!/usr/bin/env python3
"""Run only the manuscript-used Round393 prediction-time cell-count grid."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd

from metar.paths import resolve_source


CELL_COUNTS = (1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 300)
LOCKED_OUTPUTS = {
    "Figure5_R393_cellnumber_repeat_level_metrics.tsv":
        "39b14ecb1b12afb7f76e7aa3623674a2c03396ff77abcd9a088f788a23d9a3e9",
    "Figure5_R393_cellnumber_sensitivity_curve.tsv":
        "cbb4ea2d537c59d9ff691ff2415db7281061f8105f893a76936b8f6efec99fef",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_current(root: Path) -> None:
    source = root / "figures" / "reproduction" / "fig5" / "round393_current" / "source_data"
    for name, expected in LOCKED_OUTPUTS.items():
        path = source / name
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Round393 current output mismatch: {path}")
        frame = pd.read_csv(path, sep="\t")
        observed = tuple(sorted({int(value) for value in frame["cell_count"]}))
        if observed != CELL_COUNTS:
            raise RuntimeError(f"Wrong current cell-count grid in {path}: {observed}")


def run_current(root: Path, output_dir: Path, device: str, repeats: int) -> None:
    target = (
        root
        / "compat" / "home" / "lwn" / "claude" / "manuscript" / "write"
        / "claude_round393_fig5ef_refine_log2_20260611" / "scripts"
        / "eval_round393_extra_cellcounts.py"
    )
    runner = resolve_source(
        "code/portable/run_metar_portable.py", Path(__file__).resolve().parents[3], root
    )
    if not target.is_file() or not runner.is_file():
        raise FileNotFoundError(target if not target.is_file() else runner)
    subprocess.run(
        [
            sys.executable,
            str(runner),
            "--package-root", str(root),
            str(target),
            "--out-dir", str(output_dir),
            "--device", device,
            "--repeats", str(repeats),
            "--cell-counts", ",".join(map(str, CELL_COUNTS)),
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.package_root.resolve()
    check_current(root)
    if args.check_only:
        print("FIG5_CELLNUMBER_CURRENT_CHECK_OK")
        return 0
    if args.output_dir is None:
        raise RuntimeError("--output-dir is required unless --check-only is used")
    run_current(root, args.output_dir.resolve(), args.device, args.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
