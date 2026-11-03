#!/usr/bin/env python3
"""Launch the locked Round261 proteomics run with package-local paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


EXPECTED_PROTEOMICS_SHA = "f8a1da7355190aaa32ab9f13334efa09562cd43cb68b30510a5a3a405304da2b"
EXPECTED_LABEL_SHA = "fd628af6f43d416b890cff031b1e80aa0460761322520a87cdfc62311839d116"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch for {path}: expected={expected} actual={actual}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--output-dir", default="reproduction_runs/proteomics")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    root = Path(args.package_root).resolve()
    script = (
        root
        / "compat/home/lwn/claude/manuscript/write"
        / "claude_round259_crosscell_fullspace_round252query_then_sc_proteomics_20260522"
        / "scripts/run_round261_proteomics_round259_embedding.py"
    )
    data_dir = root / "data/processed/proteomics"
    proteomics = data_dir / "proteomicsdb_data.csv"
    label_table = root / "data/processed/fullspace/round259_target_embedding_aligned_to_proteomics_gene_order.csv"
    method_lock = (
        root
        / "code/original_runs/r259_261/proteomics_results"
        / "round261_proteomics_round259_embedding_method_lock.json"
    )
    for path in (script, method_lock):
        if not path.is_file():
            raise FileNotFoundError(path)
    require_hash(proteomics, EXPECTED_PROTEOMICS_SHA)
    require_hash(label_table, EXPECTED_LABEL_SHA)

    lock = json.loads(method_lock.read_text(encoding="utf-8-sig"))
    config = lock["config"]
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir = output_dir.resolve()
    output_dir.relative_to(root)
    stem = output_dir / "round261_proteomics_round259_embedding"

    env = os.environ.copy()
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PT_MAIN_DIR": str(root / "compat/home/lwn/claude/pt/main"),
            "PT_DATA_DIR": str(data_dir),
            "LABEL_REPR_PATH": str(label_table),
            "OUT_PATH": str(stem.with_suffix(".json")),
            "SAVE_CKPT_PATH": str(stem.with_suffix(".pth")),
            "PER_DRUG_PATH": str(output_dir / f"{stem.name}_per_drug.tsv"),
            "HISTORY_TSV_PATH": str(output_dir / f"{stem.name}_training_history.tsv"),
            "METHOD_LOCK_PATH": str(output_dir / f"{stem.name}_method_lock.json"),
            "METRIC_SUMMARY_PATH": str(output_dir / f"{stem.name}_metric_summary.tsv"),
            "LEAKAGE_AUDIT_PATH": str(output_dir / f"{stem.name}_leakage_and_denominator_audit.md"),
            "SELECTED_RESULT_PATH": str(output_dir / f"{stem.name}_selected_primary_result.json"),
            "DEVICE": args.device,
            "INIT_SCALE": "10.0",
        }
    )
    key_map = {
        "epochs": "EPOCHS",
        "eval_every": "EVAL_EVERY",
        "lr": "LR",
        "weight_decay": "WEIGHT_DECAY",
        "val_frac": "VAL_FRAC",
        "holdout_k_percent": "HOLDOUT_K_PERCENT",
        "seed": "SEED",
        "proj_dim": "PROJ_DIM",
        "n_res": "N_RES",
        "dropout": "DROPOUT",
        "batch_size": "BATCH_SIZE",
        "topk_feat": "TOPK_FEAT",
        "label_adapter_mode": "LABEL_ADAPTER_MODE",
        "label_bottleneck": "LABEL_BOTTLENECK",
        "adapter_l2": "ADAPTER_L2",
        "ortho_lambda": "ORTHO_LAMBDA",
    }
    for key, env_name in key_map.items():
        env[env_name] = str(config[key])

    audit = {
        "script": str(script),
        "proteomics": str(proteomics),
        "label_table": str(label_table),
        "output_dir": str(output_dir),
        "device": args.device,
        "config": config,
    }
    if args.check_only:
        print(json.dumps(audit, indent=2))
        return 0

    output_dir.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, str(script)], cwd=root, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
