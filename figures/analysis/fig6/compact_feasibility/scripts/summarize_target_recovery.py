"""Compute manuscript result tables without historical figure layouts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

FULL_SPACE=16583
TOP_FRACTIONS=[("top 1%",0.01),("top 5%",0.05),("top 10%",0.10)]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))

def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def fnum(value: str | object) -> float:
    if value == "" or value is None:
        return float("nan")
    return float(value)

def parse_ranks(value: str) -> list[int]:
    return [int(part) for part in value.split(";") if part.strip()]

def derive_fullspace_hit_recall(
    holdout_summary: list[dict[str, str]],
    holdout_detail: list[dict[str, str]],
) -> list[dict[str, object]]:
    summary_by_threshold = {row["threshold"]: row for row in holdout_summary}
    all_ranks = [rank for row in holdout_detail for rank in parse_ranks(row["all_positive_ranks"])]
    total_positive_targets = len(all_ranks)
    rows: list[dict[str, object]] = []
    for label, fraction in TOP_FRACTIONS:
        key = label.replace(" ", "")
        k = int(round(fraction * FULL_SPACE))
        hit_row = summary_by_threshold[key]
        recovered_targets = sum(1 for rank in all_ranks if rank <= k)
        rows.append(
            {
                "threshold": label,
                "k": k,
                "universe": FULL_SPACE,
                "n_drugs": int(fnum(hit_row["n_evaluable_drugs"])),
                "n_drugs_hit": int(fnum(hit_row["n_recovered"])),
                "hit": fnum(hit_row["MeTar"]),
                "hit_random": fnum(hit_row["matched_random"]),
                "n_positive_targets": total_positive_targets,
                "n_targets_recovered": recovered_targets,
                "recall": recovered_targets / total_positive_targets,
                "recall_random": fraction,
            }
        )
    return rows



def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-summary",type=Path,required=True)
    parser.add_argument("--holdout-detail",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows=derive_fullspace_hit_recall(read_tsv(args.holdout_summary),read_tsv(args.holdout_detail))
    write_tsv(args.output,rows,list(rows[0]))


if __name__ == "__main__":
    main()
