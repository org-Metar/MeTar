#!/usr/bin/env python3
"""Rebuild or retrain only the manuscript-used Fig. 5 pseudo-bulk control."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from metar.paths import resolve_source


SELECTION_POLICY = "matched_singlecell_objective"
METHOD_SC = "Single-cell adapter"
METHOD_PB = "Pseudo-bulk adapter"
THRESHOLDS = (0.01, 0.05, 0.10, 0.20)
SELECTIONS = {
    "A549": ("drug_normalized_bce", 2043),
    "SW480": ("label_bag_retrieval", 2041),
    "NCI-H460": ("drug_normalized_bce", 2042),
}
LOCKED_HASHES = {
    "train_A549/drug_normalized_bce__seed2043/checkpoint.pth":
        "3821ab0c312f77775159a602a22463c81fd4207ecf884a465d4874f03e720335",
    "train_A549/drug_normalized_bce__seed2043/holdout_per_drug.tsv":
        "8a012ad95fc01f8d5d60ff771ade6872363e47c0404ff3e93f6975a084c28a32",
    "train_SW480/label_bag_retrieval__seed2041/checkpoint.pth":
        "62917a738099f4de6cfcf98f3ca9d671996e595bb787939c51837c5eecdb9640",
    "train_SW480/label_bag_retrieval__seed2041/holdout_per_drug.tsv":
        "f2e1560afa3cf2d7c1abd74c5f0faeee82f1fdfd419c4c28e5d8ff9c3642ce7d",
    "train_NCI_H460/drug_normalized_bce__seed2042/checkpoint.pth":
        "2ccf2b24cfc82d767bd51f373029f6c95bf588f88aaaac57b42577b0a4bd4b60",
    "train_NCI_H460/drug_normalized_bce__seed2042/holdout_per_drug.tsv":
        "152353bb633d80c56e8d52c25c3d2ab045ae0d0c0fd160031caf288773dee759",
    "source_data/Round386_x128_mean_pseudobulk_profiles.npz":
        "e106fdcbd4a881228b1212dbe60740a0f8ab1744985e9f55b9f3e9ad944b1ae9",
    "scripts/run_round386_x128_pseudobulk_adapter.py":
        "c6d7636f29a8be2dcdfdd3fea0326a33936b6ab3fc10983b4b2e7f838ee901c0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_paths(root: Path) -> tuple[Path, Path, Path]:
    round386 = root / "figures" / "reproduction" / "fig5" / "round386_current"
    reference = root / "figures" / "reproduction" / "fig5" / "round398_reference"
    current = root / "figures" / "reproduction" / "round523" / "source_data"
    return round386, reference, current


def verify_locked_inputs(round386: Path) -> None:
    for relative, expected in LOCKED_HASHES.items():
        path = round386 / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Locked Fig. 5f input mismatch: {relative}: {actual}")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
    return float(center - half), float(center + half)


def load_selected_holdouts(round386: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    selections: list[dict[str, object]] = []
    for cell, (loss_mode, seed) in SELECTIONS.items():
        run_dir = round386 / f"train_{cell.replace('-', '_')}" / f"{loss_mode}__seed{seed}"
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        if result.get("cell_line") != cell or result.get("loss_mode") != loss_mode or int(result.get("seed")) != seed:
            raise RuntimeError(f"Selection metadata mismatch: {run_dir}")
        frame = pd.read_csv(run_dir / "holdout_per_drug.tsv", sep="\t")
        frame["cell_line"] = cell
        frame["loss_mode"] = loss_mode
        frame["seed"] = seed
        frame["selection_policy"] = SELECTION_POLICY
        frames.append(frame)
        selections.append(
            {
                "selection_policy": SELECTION_POLICY,
                "cell_line": cell,
                "loss_mode": loss_mode,
                "seed": seed,
                "checkpoint_sha256": sha256(run_dir / "checkpoint.pth"),
                "result_sha256": sha256(run_dir / "result.json"),
                "holdout_sha256": sha256(run_dir / "holdout_per_drug.tsv"),
                "holdout_units": len(frame),
            }
        )
    combined = pd.concat(frames, ignore_index=True)
    if len(combined) != 90 or set(combined["selection_policy"]) != {SELECTION_POLICY}:
        raise RuntimeError("Current pseudo-bulk denominator must be 30 drugs x 3 cells")
    return combined, pd.DataFrame(selections)


def summarize_curve(frame: pd.DataFrame, random_by_threshold: dict[float, float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cell in (*SELECTIONS, "pooled"):
        subset = frame if cell == "pooled" else frame[frame["cell_line"] == cell]
        for threshold in THRESHOLDS:
            column = f"hit_top{int(threshold * 100)}pct"
            hits = subset[column].astype(int) if column in subset else (subset["best_percentile"] <= threshold).astype(int)
            n_units = int(len(subset))
            n_success = int(hits.sum())
            low, high = wilson(n_success, n_units)
            rows.append(
                {
                    "method": METHOD_PB,
                    "selection_policy": SELECTION_POLICY,
                    "cell_line": cell,
                    "threshold_percent": threshold,
                    "n_units": n_units,
                    "n_success": n_success,
                    "recovery": n_success / n_units,
                    "wilson95_low": low,
                    "wilson95_high": high,
                    "candidate_space": 16583,
                    "matched_random_anyhit_mean": random_by_threshold[threshold],
                }
            )
    return pd.DataFrame(rows)


def compare_current_table(actual: pd.DataFrame, reference_path: Path, columns: list[str]) -> None:
    expected = pd.read_csv(reference_path, sep="\t")
    pd.testing.assert_frame_equal(
        actual[columns].reset_index(drop=True),
        expected[columns].reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def rebuild_tables(root: Path, output_dir: Path) -> dict[str, object]:
    round386, reference, _current = package_paths(root)
    verify_locked_inputs(round386)
    output_dir.mkdir(parents=True, exist_ok=True)

    holdouts, selections = load_selected_holdouts(round386)
    single_cell_reference = pd.read_csv(
        reference / "Figure5_R393_pseudobulk_compact_curve.tsv", sep="\t"
    )
    single_cell = single_cell_reference[single_cell_reference["method"] == METHOD_SC].copy()
    random_by_threshold = (
        single_cell.drop_duplicates("threshold_percent")
        .set_index("threshold_percent")["matched_random_anyhit_mean"]
        .astype(float)
        .to_dict()
    )
    curve = summarize_curve(holdouts, random_by_threshold)
    figure_curve = pd.concat([single_cell, curve], ignore_index=True, sort=False)
    compact = pd.concat(
        [
            curve[curve["cell_line"] == "pooled"],
            single_cell,
        ],
        ignore_index=True,
        sort=False,
    )
    compact_columns = [
        "method", "cell_line", "threshold_percent", "n_units", "n_success",
        "recovery", "wilson95_low", "wilson95_high", "candidate_space",
        "matched_random_anyhit_mean",
    ]
    figure_columns = [*compact_columns, "selection_policy"]
    compact = compact[compact_columns]
    figure_curve = figure_curve[figure_columns]

    compare_current_table(
        figure_curve,
        reference / "Figure5_R398_fig5e_corrected_curve.tsv",
        figure_columns,
    )
    compare_current_table(
        compact,
        reference / "Figure5_R398_pseudobulk_compact_curve.tsv",
        compact_columns,
    )

    outputs = {
        "Figure5_R398_pseudobulk_candidate_holdout_long.tsv": holdouts,
        "Figure5_R398_pseudobulk_selection_current.tsv": selections,
        "Figure5_R398_pseudobulk_policy_recovery_current.tsv": curve,
        "Figure5_R398_fig5e_corrected_curve.tsv": figure_curve,
    }
    hashes: dict[str, str] = {}
    for name, frame in outputs.items():
        path = output_dir / name
        frame.to_csv(path, sep="\t", index=False, lineterminator="\r\n")
        hashes[name] = sha256(path)

    # Keep the exact article-consumed decimal serialization after proving that
    # the current-only rebuild is numerically identical to it.
    compact_name = "Figure5_R398_pseudobulk_compact_curve.tsv"
    compact_path = output_dir / compact_name
    shutil.copy2(reference / compact_name, compact_path)
    hashes[compact_name] = sha256(compact_path)

    audit = {
        "scope": "manuscript-used processed-feature pseudo-bulk control only",
        "selection_policy": SELECTION_POLICY,
        "selected_branches": {
            cell: {"loss_mode": loss_mode, "seed": seed}
            for cell, (loss_mode, seed) in SELECTIONS.items()
        },
        "holdout_units": len(holdouts),
        "candidate_space": 16583,
        "excluded": ["validation_selected", "holdout_oracle", "raw UMI pseudo-bulk"],
        "output_sha256": hashes,
    }
    audit_path = output_dir / "Figure5_R398_pseudobulk_current_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    hashes[audit_path.name] = sha256(audit_path)
    return audit


def load_trainer(path: Path):
    spec = importlib.util.spec_from_file_location("round386_current_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train_current(root: Path, output_dir: Path, device_name: str, force: bool) -> None:
    import torch

    round386, _reference, _current = package_paths(root)
    verify_locked_inputs(round386)
    trainer_path = resolve_source(
        "figures/reproduction/fig5/round386_current/scripts/run_round386_x128_pseudobulk_adapter.py",
        Path(__file__).resolve().parents[3], root,
    )
    trainer = load_trainer(trainer_path)
    trainer.TARGET_EMBEDDING_PATH = root / "data" / "processed" / "fullspace" / "round259_target_embedding.csv"
    trainer.PROTOCOL_JSON = root / "data" / "processed" / "tahoe" / "sc_protocol_lock_v3_20260326.json"
    trainer.DTI_PATH = root / "data" / "processed" / "tahoe" / "tahoe_dti.csv"
    output_source = output_dir / "source_data"
    output_source.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        round386 / "source_data" / "Round386_x128_mean_pseudobulk_profiles.npz",
        output_source / "Round386_x128_mean_pseudobulk_profiles.npz",
    )
    device = torch.device(device_name if torch.cuda.is_available() and device_name.startswith("cuda") else "cpu")
    for cell, (loss_mode, seed) in SELECTIONS.items():
        run_dir = output_dir / f"train_{cell.replace('-', '_')}" / f"{loss_mode}__seed{seed}"
        if run_dir.exists() and not force:
            raise FileExistsError(run_dir)
        context = trainer.TrainContext(output_dir, cell, device)
        trainer.train_one(context, loss_mode, seed, run_dir)


def check_current(root: Path) -> None:
    round386, _reference, current = package_paths(root)
    verify_locked_inputs(round386)
    matched = pd.read_csv(current / "Figure5_R398_pseudobulk_candidate_holdout_long.tsv", sep="\t")
    if len(matched) != 90 or set(matched["selection_policy"]) != {SELECTION_POLICY}:
        raise RuntimeError("Fig. 5 current table contains non-current selection policies")
    audit = json.loads((current / "Figure5_R398_pseudobulk_current_audit.json").read_text(encoding="utf-8"))
    if audit.get("selection_policy") != SELECTION_POLICY:
        raise RuntimeError("Fig. 5 current audit has the wrong selection policy")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("rebuild", "check", "train-current"))
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = args.package_root.resolve()
    _round386, _reference, current = package_paths(root)
    if args.command == "check":
        check_current(root)
        print("FIG5F_CURRENT_CHECK_OK")
    elif args.command == "rebuild":
        audit = rebuild_tables(root, (args.output_dir or current).resolve())
        print(json.dumps(audit, sort_keys=True))
    else:
        if args.output_dir is None:
            raise RuntimeError("--output-dir is required for train-current")
        train_current(root, args.output_dir.resolve(), args.device, args.force)
        print("FIG5F_CURRENT_TRAIN_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
