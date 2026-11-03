#!/usr/bin/env python3
"""Recompute the scientific LINCS/Tahoe recall tables used for Figure7."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPOSITORY = Path(__file__).resolve().parents[1]
EXPECTED = {
    "round493/Figure7_R493_tahoe_macro_recall_source.tsv": "52dc2807ae179e42f5db99c93ee5fd4b8624942d1c025f75ba1c87454153d6db",
    "round493/Figure7_R493_tahoe_macro_recall_summary.tsv": "96fdd051cd99f2a836f400ed65a4618f7889520c05d85e418ad7c148bc474be2",
    "round494/Figure7_R494_lincs_checkpoint_macro_recall_source.tsv": "a33e475604180cd6e8b4f646d7922d869fd16a2a8c38e6fb548b50fa29b5d4b3",
    "round494/Figure7_R494_lincs_checkpoint_macro_recall_summary.tsv": "553c38011b1dddde5b8a8ce0c3a32c3da524051bc32e45eb4d7eb1a7987f4a1d",
}


def load_module(name: str, relative: str):
    path = REPOSITORY / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import scientific source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def text_digest(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Downloaded artifact root")
    parser.add_argument("--output", type=Path, default=Path("results/figure7_reproduction"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    output.mkdir(parents=True)
    figure_root = root / "figures/reproduction/fig7"
    report = {"status": "running", "files": [], "renderer": "scripts/plot_figures.py render fig7"}
    try:
        with tempfile.TemporaryDirectory(prefix="metar_figure7_") as directory:
            scratch = Path(directory)
            tahoe = load_module("tahoe_recall", "figures/analysis/fig7/macro_recall/scripts/build_macro_recall_sources.py")
            tahoe.SOURCE_OUT = scratch / "round493"
            tahoe.SOURCE_OUT.mkdir()
            source = figure_root / "round492/source_data"
            tahoe.SC_METRICS = source / "round260_singlecell_score_metrics.tsv"
            tahoe.SC_SCORES = source / "round260_singlecell_fullspace_scores.tsv.gz"
            tahoe.SC_LABELS = source / "round260_singlecell_target_labels.tsv"
            values = tahoe.compute_tahoe_recall()
            tahoe.summarize_recall(values).to_csv(tahoe.SOURCE_OUT / "Figure7_R493_tahoe_macro_recall_summary.tsv", sep="\t", index=False)

            lincs = load_module("lincs_recall", "figures/analysis/fig7/checkpoint_recall/scripts/build_lincs_checkpoint_recall.py")
            lincs.SOURCE_OUT = scratch / "round494"
            lincs.QA_OUT = scratch / "calculation_details"
            lincs.seed_paths = lambda source, seed: {
                "checkpoint": source / "runs" / f"seed_{seed}" / "checkpoint.pt",
                "adapter": source / "support_adapter" / f"seed_{seed}" / "support_adapter.pt",
                "score": source / "scores" / f"seed_{seed}" / "support_adapted_scores.tsv.gz",
            }
            lincs.compute_lincs(SimpleNamespace(
                round429_root=figure_root / "round429_L5_current_signature",
                label_order=root / "data/processed/lincs/r429_runtime/seen_668_genes.tsv",
                seeds=[1, 2, 3], min_cell_lines=2,
            ))
            for key, expected in EXPECTED.items():
                stage, name = key.split("/", 1)
                actual = scratch / stage / name
                if text_digest(actual) != expected:
                    raise RuntimeError(f"Recomputed values differ from the selected scientific table: {key}")
                destination = output / name
                destination.write_bytes(actual.read_bytes())
                report["files"].append({"path": name, "matches_reference": True})

            # The current rendering tables are checked directly against their
            # scientific calculations, independent of historical layout scripts.
            recovery = load_module("consensus_recovery", "figures/analysis/fig7/hitk_recovery/scripts/summarize_consensus_recovery.py")
            recovery.SOURCE_OUT = scratch / "consensus"
            recovery.SOURCE_OUT.mkdir()
            recovery.ROUND430_PER_DRUG = figure_root / "round430_outputs/metrics/multicell_consensus_per_drug.tsv"
            recovery.ROUND430_SUMMARY = figure_root / "round430_outputs/metrics/multicell_consensus_summary.tsv"
            recovery.SC_METRICS = tahoe.SC_METRICS
            recovery.SC_SCORES = tahoe.SC_SCORES
            recovery.SC_LABELS = tahoe.SC_LABELS
            recovery.compute_lincs_recovery_curve()
            recovery.compute_lincs_rank_pairs()
            recovery.compute_lincs_state_summary()
            recovery.compute_tahoe_consensus()
            import pandas as pd
            pairs = {
                "Figure7_lincs_recovery_curve_source.tsv": "Figure7_lincs_hitk_curve_source.tsv",
                "Figure7_lincs_rank_pair_source.tsv": "Figure7_lincs_rank_pair_source.tsv",
                "Figure7_lincs_state_summary_source.tsv": "Figure7_lincs_state_summary_source.tsv",
                "Figure7_tahoe_recovery_curve_source.tsv": "Figure7_tahoe_hitk_curve_source.tsv",
                "Figure7_tahoe_rank_pair_source.tsv": "Figure7_tahoe_rank_pair_source.tsv",
                "Figure7_tahoe_state_summary_source.tsv": "Figure7_tahoe_state_summary_source.tsv",
            }
            for generated, reference in pairs.items():
                actual_frame = pd.read_csv(recovery.SOURCE_OUT / generated, sep="\t")
                reference_frame = pd.read_csv(REPOSITORY / "data/plot_sources/figure7" / reference, sep="\t")
                columns = [column for column in reference_frame.columns if column in actual_frame.columns]
                left = actual_frame[columns].sort_values(columns).reset_index(drop=True)
                right = reference_frame[columns].sort_values(columns).reset_index(drop=True)
                pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=False, rtol=1e-11, atol=1e-12)
                report["files"].append({"path": reference, "matches_reference": True})
        report["status"] = "pass"
        return 0
    except Exception as exc:
        report["status"] = "fail"
        report["error"] = str(exc)
        raise
    finally:
        (output / "reproduction.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
