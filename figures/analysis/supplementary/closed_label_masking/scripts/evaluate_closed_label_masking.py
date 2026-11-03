#!/usr/bin/env python3
"""Evaluate Round259 closed-label-masked recovery without retraining."""

from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch


REMOTE_WRITE_ROOT = Path("/home/lwn/claude/manuscript/write")
ROUND259_ROOT = REMOTE_WRITE_ROOT / "claude_round259_crosscell_fullspace_round252query_then_sc_proteomics_20260522"
ROUND259_SCRIPT = ROUND259_ROOT / "scripts" / "run_round259_crosscell_fullspace_round252query.py"
ROUND230_ROOT = REMOTE_WRITE_ROOT / "claude_round230_scratch_mlp_metar_target_tower_pc3_20260518"
ROUND246_CKPT = (
    REMOTE_WRITE_ROOT
    / "claude_round246_joint_seen_target_train_pc3_a375_mcf7_20260519"
    / "checkpoints"
    / "r246_joint_lm0p5_sc0p15_mn2p5"
    / "whole_model_checkpoint.pth"
)
ROUND252_PROTOCOL_AUDIT = (
    REMOTE_WRITE_ROOT
    / "claude_round252_epiqgate_pseudoval_sweep_20260520"
    / "source_data"
    / "cell_splits.json"
)
ROUND382_ROOT = REMOTE_WRITE_ROOT / "claude_round382_supp_fig6_closed_label_removal_control_20260610"
SOURCE = ROUND382_ROOT / "source_data"
QA = ROUND382_ROOT / "qa"
THRESHOLDS = [0.01, 0.05, 0.10, 0.20]


def load_round259_module():
    spec = importlib.util.spec_from_file_location("round259_eval_source", ROUND259_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import Round259 script from {ROUND259_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROUND_ROOT = ROUND259_ROOT
    module.REMOTE_WRITE_ROOT = REMOTE_WRITE_ROOT
    module.ROUND230_ROOT = ROUND230_ROOT
    module.ROUND230_SCRIPT = ROUND230_ROOT / "scripts" / "run_round230_scratch_mlp_metar_target_tower_pc3.py"
    module.ROUND246_CKPT = ROUND246_CKPT
    module.ROUND252_PROTOCOL_AUDIT = ROUND252_PROTOCOL_AUDIT
    return module


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def main() -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)

    module = load_round259_module()
    base = module.load_round230_module()
    base.ROUND_ROOT = module.ROUND230_ROOT
    base.SOURCE = base.ROUND_ROOT / "source_data"
    base.QA = base.ROUND_ROOT / "qa"
    base.CHECKPOINTS = base.ROUND_ROOT / "checkpoints"
    base.H07_WRAPPER = module.REMOTE_WRITE_ROOT / "bulk_sharedsample_indunseenlabel_20260330_h07" / "scripts" / "train_bulk_sharedsample_indunseenlabel_v1_20260330.py"
    base.H07_CACHE = module.REMOTE_WRITE_ROOT / "claude_round209_fullspace_a01_extrapolation_20260517" / "cache" / "h07_prepared_data.pt"
    base.H07_CKPT = (
        module.REMOTE_WRITE_ROOT
        / "bulk_sharedsample_indunseenlabel_20260330_h07_w060_u8e4_med"
        / "checkpoints"
        / "best_unseen_h07_indlabel_linear_e04init_w060_u8e4_med.pth"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base.set_seed(base.SEED)
    _, data, _ = base.load_h07_data()
    protocol_audit = module.load_round252_protocol_audit()
    _, holdouts, split_counts = module.build_round259_splits(base, data, protocol_audit)
    train_idx = data["train_gene_global_idx"].detach().cpu().numpy().astype("int64")
    query_state, query_audit = module.load_round246_query_state()

    selected_variant = "e13_bce001"
    cfg = deepcopy(module.VARIANTS[selected_variant])
    cfg["target_mode"] = "round259_round246query_crosscell_round240_fullspace_ppi_t4"
    cfg["ablation"] = f"round259_{selected_variant}_round246query_crosscell_fullspace_ppi_t4"
    model = module.build_model(base, data, device, train_idx, cfg, query_state).to(device)
    ckpt_path = ROUND259_ROOT / "checkpoints" / selected_variant / "selected_checkpoint.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    best_epoch = int(ckpt.get("best_epoch", -1))

    rows = []
    per_threshold_rows = []
    for exclude_seen in [False, True]:
        for threshold in THRESHOLDS:
            metrics, per = module.evaluate_holdout_percent(
                model,
                holdouts["holdout_test"],
                device,
                batch_size=256,
                percent=threshold,
                exclude_seen=exclude_seen,
                return_rows=True,
            )
            key = f"top{int(threshold * 100)}pct_success"
            n_profiles = int(metrics["n_profiles"])
            n_success = int(round(float(metrics[key]) * n_profiles))
            low, high = wilson(n_success, n_profiles)
            rows.append(
                {
                    "analysis": "full_universe" if not exclude_seen else "closed_labels_masked",
                    "split": "holdout_test",
                    "cell_split_role": "official_unseen_final",
                    "evaluation_unit": "profile",
                    "threshold_percent": threshold,
                    "top_k_rank_cutoff": int(16583 * threshold),
                    "n_profiles": n_profiles,
                    "n_success": n_success,
                    "recovery": float(metrics[key]),
                    "wilson95_low": low,
                    "wilson95_high": high,
                    "candidate_space": int(metrics["candidate_space"]),
                    "exclude_seen_labels": bool(metrics["exclude_seen_labels"]),
                    "n_masked_closed_labels": 668 if exclude_seen else 0,
                    "selected_variant": selected_variant,
                    "best_epoch": best_epoch,
                }
            )
            if per is not None:
                per = per.copy()
                per["analysis"] = "full_universe" if not exclude_seen else "closed_labels_masked"
                per["threshold_percent"] = threshold
                per["success"] = per[f"success_top{int(threshold * 100)}pct"].astype(bool)
                per_threshold_rows.append(per)

    curve = pd.DataFrame(rows)
    if split_counts["holdouts"]["holdout_test"]["n_profiles"] != 4463:
        raise RuntimeError(split_counts["holdouts"]["holdout_test"])
    if split_counts["n_genes_full_universe"] != 16583 or split_counts["n_train_label_genes"] != 668:
        raise RuntimeError(split_counts)
    full5 = curve[(curve.analysis == "full_universe") & (curve.threshold_percent == 0.05)]["recovery"].iloc[0]
    full10 = curve[(curve.analysis == "full_universe") & (curve.threshold_percent == 0.10)]["recovery"].iloc[0]
    masked5 = curve[(curve.analysis == "closed_labels_masked") & (curve.threshold_percent == 0.05)]["recovery"].iloc[0]
    if abs(full5 - 0.3269101501232355) > 1e-12:
        raise RuntimeError(f"Full top5 mismatch: {full5}")
    if abs(full10 - 0.6139368138023751) > 1e-12:
        raise RuntimeError(f"Full top10 mismatch: {full10}")
    if abs(masked5 - 0.5343939054447681) > 1e-12:
        raise RuntimeError(f"Masked top5 mismatch: {masked5}")

    curve.to_csv(SOURCE / "Supplementary_Figure6_closed_label_masked_recovery_curve.tsv", sep="\t", index=False)
    if per_threshold_rows:
        pd.concat(per_threshold_rows, ignore_index=True).to_csv(
            SOURCE / "Supplementary_Figure6_per_profile_threshold_calls.tsv.gz",
            sep="\t",
            index=False,
            compression="gzip",
        )
    audit = {
        "round": "Round382",
        "source_round": "Round259",
        "selected_variant": selected_variant,
        "selected_checkpoint": str(ckpt_path),
        "selected_checkpoint_sha256": module.sha256_file(ckpt_path),
        "round246_checkpoint": str(module.ROUND246_CKPT),
        "round246_checkpoint_sha256": query_audit["round246_checkpoint_sha256"],
        "thresholds": THRESHOLDS,
        "device": str(device),
        "counts": split_counts,
        "mask_rule": "Set the 668 closed-set benchmark labels to -inf before ranking; retain full-universe percentile cutoffs.",
    }
    (SOURCE / "Supplementary_Figure6_evaluation_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (ROUND382_ROOT / "method_lock.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(curve.to_string(index=False))


if __name__ == "__main__":
    main()
