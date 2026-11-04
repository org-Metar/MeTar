"""Article-exact cross-cell and single-cell verification."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

from benchmark_utils import hash_matches_manifest, import_module, sha256_file
from metar.io import load_target_table
from metar.predict import predict_cross_cell, predict_single_cell
from metar.runtime import report_metrics, report_step, resolve_device


SELECTED_CROSS_VARIANT = (
    "r252_epiqgate_lm1_lr0p00015_sc0p18_mn2p5_ba0p1_"
    "s32_q32_diverse_label_seed20260524_official_unseen_diverse_label_s64"
)
CELL_SELECTIONS = {
    "A549": (
        "A549_selected_primary_result.portable.json",
        "drug_normalized_bce",
        2043,
        "tahoe_a549.h5ad",
    ),
    "SW480": (
        "SW480_selected_primary_result.portable.json",
        "label_bag_retrieval",
        2041,
        "tahoe_sw480.h5ad",
    ),
    "NCI-H460": (
        "NCI_H460_selected_primary_result.portable.json",
        "drug_normalized_bce",
        2042,
        "tahoe_nci_h460.h5ad",
    ),
}


def build_round252_context(
    args: argparse.Namespace, device: torch.device
) -> tuple[Any, Any, dict[str, Any]]:
    r252 = import_module(
        "article_exact_r252",
        args.round252 / "scripts/run_round252_epiqgate_pseudoval_sweep.py",
    )
    scratch_root = args.output / "round252_runtime"
    r252.ROUND_ROOT = scratch_root
    r252.SOURCE = scratch_root / "source_data"
    r252.QA = scratch_root / "qa"
    r252.CHECKPOINTS = scratch_root / "checkpoints"
    r252.LOGS = scratch_root / "logs"
    r252.ensure_dirs()
    r250 = r252.load_round250_module()
    namespace = SimpleNamespace(
        kg_checkpoint=None,
        pseudo_unseen_cells="HA1E,HEK293,MCF10A,MDAMB231,PHH,THP1",
        official_unseen_cells="round215_heldout_panel",
        seen_diagnostic_cells="PC3,A375,MCF7",
        support_baseline_variant="r246_joint_lm0p5_sc0p15_mn2p5",
        support_checkpoint=str(
            args.package / "checkpoints/selected/round246_whole_model_checkpoint.pth"
        ),
    )
    r243, r211, r208, labels, kg_path, kg_ckpt, tables, data_audit = (
        r250.load_context(namespace, device)
    )
    support_model, support_adapter, support_payload = (
        r250.load_current_support_baseline(
            r243,
            r211,
            kg_ckpt,
            labels,
            Path(namespace.support_checkpoint),
            device,
        )
    )
    for key in ("official_support", "official_test", "official_test_profile"):
        r243.precompute_q(support_model, tables[key], device)
    context = {
        "r208": r208,
        "labels": labels,
        "tables": tables,
        "data_audit": data_audit,
        "support_model": support_model,
        "support_adapter": support_adapter,
        "support_payload": support_payload,
        "kg_path": kg_path,
    }
    return r252, r252.load_round247_module(), context


def verify_cross_cell(args: argparse.Namespace) -> dict[str, Any]:
    args.device = resolve_device(getattr(args, "device", "auto"))
    device = torch.device(args.device)
    report_metrics(
        {"setting": "cross-cell verification", "device": args.device}, label="Setting"
    )
    report_step("Load locked cross-cell inputs and support context")
    r252, r247, context = build_round252_context(args, device)
    support_item = context["tables"]["official_support"]
    query_item = context["tables"]["official_test"]
    base_checkpoint = (
        args.package / "checkpoints/selected/round246_whole_model_checkpoint.pth"
    )
    adapter_checkpoint = (
        args.package / "checkpoints/selected/round252_adapter_seed20260524.pth"
    )
    per_parts = []
    score_parts = []
    score_cells = []
    score_drugs = []
    report_step("Score cross-cell queries and compare existing reference metrics")
    for cell, raw_indices in query_item["by_cell"].items():
        query_indices = np.asarray(raw_indices, dtype=np.int64)
        support_all = np.asarray(support_item["by_cell"][cell], dtype=np.int64)
        support_rows = support_item["rows"].iloc[support_all].reset_index(drop=True)
        support_y = support_item["y"][support_all]
        local = r252.select_support_indices(
            r247,
            support_rows,
            support_y,
            64,
            "diverse_label",
            r252.stable_seed(r252.SEED, "r251", cell, "diverse_label", 64),
        )
        support_indices = support_all[local]
        query_rows = query_item["rows"].iloc[query_indices].reset_index(drop=True)
        query_ids = tuple(query_rows["drug"].astype(str))
        result = predict_cross_cell(
            query_item["x"][query_indices],
            query_ids,
            support_item["x"][support_indices],
            support_item["y"][support_indices],
            base_checkpoint,
            adapter_checkpoint,
            support_weights=support_item["weights"][support_indices],
            device=args.device,
        )
        _summary, per = context["r208"].compute_metrics(
            result.scores,
            query_item["y"][query_indices],
            query_rows,
            "Round252_EpiQGate",
            SELECTED_CROSS_VARIANT,
            "drug-mean",
        )
        per["method_family"] = "epiqgate"
        per["split_role"] = "official_unseen"
        per["variant"] = SELECTED_CROSS_VARIANT
        per["adapt_support_size"] = 64
        per["used_support_rows"] = len(support_indices)
        per["requested_support_size"] = 64
        per["selector"] = "diverse_label"
        per_parts.append(per)
        score_parts.append(result.scores.astype(np.float32))
        score_cells.extend([str(cell)] * len(result.scores))
        score_drugs.extend(query_ids)
    observed = pd.concat(per_parts, ignore_index=True, sort=False)
    expected = pd.read_csv(
        args.package
        / "data/derived/article_exact/round252_selected_public_api_per_unit.tsv.gz",
        sep="\t",
        low_memory=False,
    )
    expected = expected[
        expected["method_family"].eq("epiqgate")
        & expected["variant"].eq(SELECTED_CROSS_VARIANT)
        & expected["split"].eq("official_test")
        & expected["evaluation_unit"].eq("drug-mean")
    ].reset_index(drop=True)
    keys = ["cell_line", "drug"]
    observed = observed.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    if not np.array_equal(
        observed[keys].astype(str).to_numpy(),
        expected[keys].astype(str).to_numpy(),
    ):
        raise RuntimeError("Round252 public prediction rows differ from archived rows")
    numeric = [
        "n_profiles",
        "n_true_targets",
        "Hit@1",
        "Hit@5",
        "Hit@10",
        "Hit@20",
        "Hit@30",
        "Hit@50",
        "AUROC",
        "NDCG@10",
        "MeanAP",
    ]
    max_delta = float(
        np.nanmax(
            np.abs(
                observed[numeric].to_numpy(dtype=float)
                - expected[numeric].to_numpy(dtype=float)
            )
        )
    )
    observed.to_csv(
        args.output / "round252_public_api_per_unit.tsv.gz", sep="\t", index=False
    )
    np.savez_compressed(
        args.output / "round252_public_api_scores.npz",
        scores=np.vstack(score_parts),
        cell_lines=np.asarray(score_cells),
        drugs=np.asarray(score_drugs),
        targets=np.asarray(context["labels"]),
    )
    summary = {
        "status": "PASS"
        if len(observed) == len(expected) and max_delta <= 1.0e-7
        else "FAIL",
        "adapter_matches_manifest": hash_matches_manifest(
            args, "checkpoints/selected/round252_adapter_seed20260524.pth"
        ),
        "rows": len(observed),
        "cells": int(observed["cell_line"].nunique()),
        "max_metric_abs_delta": max_delta,
    }
    report_metrics(
        {"status": summary["status"], "rows": len(observed),
         "cells": summary["cells"], "max_metric_abs_delta": max_delta},
        label="Summary",
    )
    return summary


def verify_single_cell(args: argparse.Namespace) -> dict[str, Any]:
    args.device = resolve_device(getattr(args, "device", "auto"))
    report_metrics(
        {"setting": "single-cell verification", "device": args.device}, label="Setting"
    )
    report_step("Load locked single-cell evaluation inputs")
    base = import_module(
        "article_exact_round260_base",
        args.round259 / "scripts/run_sc_label_bag_calibration_multicell_base.py",
    )
    target_table = args.package / "data/processed/fullspace/round259_target_embedding.csv"
    target_names, target_values = load_target_table(target_table)
    reference = pd.read_csv(
        args.package
        / "figures/reproduction/fig7/round492/source_data/"
        "round260_singlecell_fullspace_scores.tsv.gz",
        sep="\t",
    )
    if tuple(reference.columns[2:].astype(str)) != target_names:
        raise RuntimeError("Round492 target order differs from public target table")
    results: dict[str, Any] = {}
    base.PROTOCOL_JSON = (
        args.package / "data/processed/tahoe/sc_protocol_lock_v3_20260326.json"
    )
    base.DTI_PATH = args.package / "data/processed/tahoe/tahoe_dti.csv"
    base.FORBIDDEN_TARGET_REPR = args.package / "data/processed/tahoe/label_repr_all.csv"
    selection_root = args.package / "data/processed/tahoe/selected_runs"
    for cell_line, (selection, loss_mode, seed, filename) in CELL_SELECTIONS.items():
        report_step(f"Score {cell_line} cells and compare the existing reference")
        selected = json.loads((selection_root / selection).read_text())
        if str(selected["loss_mode"]) != loss_mode or int(selected["seed"]) != seed:
            raise RuntimeError(f"unexpected selected Round260 run for {cell_line}")
        checkpoint = args.package / "checkpoints/selected" / (
            f"round260_{cell_line.replace('-', '_')}_seed{seed}.pth"
        )
        base.TARGET_EMBEDDING_PATH = target_table
        adata_path = args.package / "data/processed/tahoe" / filename
        runtime_dir = args.output / f"round260_runtime_{cell_line.replace('-', '_')}"
        context = base.SweepContext(
            torch.device("cpu"), runtime_dir, adata_path, cell_line
        )
        expected = reference[reference["cell_line"].eq(cell_line)].reset_index(drop=True)
        blocks = []
        group_ids = []
        for drug in expected["drug"].astype(str):
            indices = context.eval_cells_by_drug[drug]
            values = base.fetch_x_batch(
                context.adata, indices, torch.device("cpu")
            ).numpy()
            blocks.append(values)
            group_ids.extend([drug] * len(values))
        observed = predict_single_cell(
            np.vstack(blocks),
            tuple(group_ids),
            checkpoint,
            target_names,
            target_values,
            cell_line=cell_line,
            max_cells=0,
            device=args.device,
        )
        if observed.query_ids != tuple(expected["drug"].astype(str)):
            raise RuntimeError(f"Round260 query order differs for {cell_line}")
        expected_scores = expected.iloc[:, 2:].to_numpy(dtype=np.float32)
        delta = np.abs(observed.scores - expected_scores)
        results[cell_line] = {
            "status": "PASS" if float(delta.max()) <= 1.0e-6 else "FAIL",
            "rows": len(expected),
            "cells_min": min(len(block) for block in blocks),
            "cells_max": max(len(block) for block in blocks),
            "score_max_abs_delta": float(delta.max()),
            "score_mean_abs_delta": float(delta.mean()),
            "checkpoint_matches_manifest": hash_matches_manifest(
                args,
                "checkpoints/selected/"
                f"round260_{cell_line.replace('-', '_')}_seed{seed}.pth",
            ),
            "adata_sha256": sha256_file(adata_path),
        }
        del context, blocks, observed, expected_scores, delta
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = {
        "status": "PASS"
        if all(item["status"] == "PASS" for item in results.values())
        else "FAIL",
        "cell_lines": results,
        "reference_rows": len(reference),
    }
    report_metrics(
        {"status": summary["status"], "cell_lines": len(results),
         "reference_rows": len(reference)},
        label="Summary",
    )
    return summary
