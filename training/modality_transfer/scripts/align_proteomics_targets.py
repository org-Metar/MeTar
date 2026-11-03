#!/usr/bin/env python
"""Align Round259 target embeddings to the locked proteomics gene order."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ALIAS_TO_LOCKED = {
    "15-Sep": "SEPT15",
    "3-Sep": "SEPT3",
    "4-Sep": "SEPT4",
    "5-Sep": "SEPT5",
    "6-Sep": "SEPT6",
    "7-Sep": "SEPT7",
    "9-Sep": "SEPT9",
    "8-Mar": "MARCHF8",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-label-repr", required=True)
    parser.add_argument("--round259-embedding", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--audit-json", required=True)
    args = parser.parse_args()

    old_path = Path(args.old_label_repr)
    new_path = Path(args.round259_embedding)
    out_csv = Path(args.out_csv)
    audit_json = Path(args.audit_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    audit_json.parent.mkdir(parents=True, exist_ok=True)

    old_df = pd.read_csv(old_path, index_col=0)
    new_df = pd.read_csv(new_path, index_col=0)
    renamed_new_df = new_df.rename(index=ALIAS_TO_LOCKED)

    duplicated_after_alias = renamed_new_df.index[renamed_new_df.index.duplicated()].tolist()
    missing_after_alias = sorted(set(old_df.index) - set(renamed_new_df.index))
    extra_after_alias = sorted(set(renamed_new_df.index) - set(old_df.index))
    if duplicated_after_alias or missing_after_alias:
        raise ValueError(
            "Cannot align Round259 embedding to locked proteomics order: "
            f"duplicates={duplicated_after_alias}, missing={missing_after_alias[:20]}"
        )

    aligned = renamed_new_df.loc[old_df.index]
    aligned.to_csv(out_csv)

    old_norm = np.linalg.norm(old_df.values.astype(np.float32), axis=1)
    new_norm = np.linalg.norm(aligned.values.astype(np.float32), axis=1)
    audit = {
        "old_label_repr_path": str(old_path),
        "old_label_repr_sha256": sha256(old_path),
        "round259_embedding_path": str(new_path),
        "round259_embedding_sha256": sha256(new_path),
        "aligned_embedding_path": str(out_csv),
        "aligned_embedding_sha256": sha256(out_csv),
        "old_shape": list(old_df.shape),
        "round259_shape": list(new_df.shape),
        "aligned_shape": list(aligned.shape),
        "locked_gene_order_preserved": bool(aligned.index.equals(old_df.index)),
        "aliases_applied": ALIAS_TO_LOCKED,
        "missing_after_alias": missing_after_alias,
        "extra_after_alias_count": len(extra_after_alias),
        "extra_after_alias_preview": extra_after_alias[:20],
        "old_norm_min": float(old_norm.min()),
        "old_norm_max": float(old_norm.max()),
        "aligned_norm_min": float(new_norm.min()),
        "aligned_norm_max": float(new_norm.max()),
    }
    audit_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
