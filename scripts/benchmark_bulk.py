"""Article-exact proteomics and full-space verification."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
from typing import Any

import numpy as np
import pandas as pd
import torch

from benchmark_utils import hash_matches_manifest, import_module, sha256_file
from metar.io import load_target_table, split_targets
from metar.predict import predict_full_space, predict_proteomics
from metar.preprocessing import fit_round261_preprocessor
from metar.runtime import report_metrics, report_step, resolve_device


METRIC_COLUMNS = [
    f"{metric}@{k}"
    for metric in ("Hit", "NDCG", "MAP")
    for k in (1, 3, 5, 10)
]


def dcg_at_k(relevance: np.ndarray, k: int) -> float:
    values = relevance[:k]
    return float((values / np.log2(np.arange(2, len(values) + 2))).sum())


def ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    ranked = labels[np.argsort(scores)[::-1]]
    ideal = np.sort(labels)[::-1]
    denominator = dcg_at_k(ideal, k)
    return dcg_at_k(ranked, k) / denominator if denominator > 0 else 0.0


def average_precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    relevance = labels[np.argsort(scores)[::-1][:k]]
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    denominator = max(1, min(positives, k))
    precisions = np.cumsum(relevance) / (np.arange(len(relevance)) + 1)
    return float((precisions * relevance).sum() / denominator)


def seen_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    rows = {name: [] for name in METRIC_COLUMNS}
    for vector, truth in zip(scores, labels):
        if truth.sum() == 0:
            continue
        for k in (1, 3, 5, 10):
            top = np.argsort(vector)[::-1][:k]
            rows[f"Hit@{k}"].append(float(truth[top].any()))
            rows[f"NDCG@{k}"].append(ndcg_at_k(vector, truth, k))
            rows[f"MAP@{k}"].append(average_precision_at_k(vector, truth, k))
    return {
        name: float(np.mean(values)) if values else 0.0
        for name, values in rows.items()
    }


def verify_proteomics(args: argparse.Namespace) -> dict[str, Any]:
    args.device = resolve_device(getattr(args, "device", "auto"))
    report_metrics(
        {"setting": "proteomics verification", "device": args.device}, label="Setting"
    )
    report_step("Load locked proteomics inputs and fit training-derived preprocessing")
    reference = args.package / "data/processed/proteomics/proteomicsdb_data.csv"
    target_table = args.package / (
        "data/processed/fullspace/"
        "round259_target_embedding_aligned_to_proteomics_gene_order.csv"
    )
    checkpoint = args.package / "checkpoints/selected/round261_proteomics_epoch070.pth"
    archived = json.loads(
        (
            args.package
            / "data/derived/article_exact/round261_archived_result.json"
        ).read_text()
    )
    target_names, target_values = load_target_table(target_table)
    target_set = set(target_names)
    preprocessor = fit_round261_preprocessor(reference, target_names)
    preprocessor_path = args.output / "round261_canonical_preprocessor.json"
    preprocessor.write(preprocessor_path)

    frame = pd.read_csv(reference, index_col=0)
    frame["target_filt"] = frame["target"].map(
        lambda value: [target for target in split_targets(value) if target in target_set]
    )
    frame = frame[frame["target_filt"].map(bool)].copy().reset_index(drop=True)
    frame["drug"] = frame["drug"].astype(str).str.strip().str.lower()
    drug_targets = {
        drug: set().union(*group["target_filt"].map(set))
        for drug, group in frame.groupby("drug")
    }
    shuffled = np.asarray(sorted(drug_targets))
    np.random.RandomState(0).shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * 0.2))
    raw_test = shuffled[:n_test]
    training = shuffled[n_test:]
    training_targets = set().union(*(drug_targets[drug] for drug in training))
    test_drugs = np.asarray(
        [drug for drug in raw_test if drug_targets[drug] & training_targets]
    )
    test_frame = frame[frame["drug"].isin(test_drugs)].copy()

    report_step("Score proteomics queries and compare existing reference metrics")
    result = predict_proteomics(
        test_frame,
        tuple(test_frame["drug"].astype(str)),
        checkpoint,
        target_names,
        target_values,
        preprocessor,
        device=args.device,
    )
    target_to_index = {target: index for index, target in enumerate(target_names)}
    all_labeled_targets = sorted(set().union(*drug_targets.values()))
    random.seed(42)
    holdout_set = set(
        random.sample(all_labeled_targets, min(50, len(all_labeled_targets)))
    )
    seen_targets = sorted(set(all_labeled_targets) - holdout_set)
    seen_indices = np.asarray([target_to_index[target] for target in seen_targets])
    seen_labels = np.zeros((len(test_frame), len(seen_targets)), dtype=np.float32)
    holdout_indices: list[list[int]] = []
    seen_position = {target: index for index, target in enumerate(seen_targets)}
    for row, targets in enumerate(test_frame["target_filt"]):
        for target in targets:
            if target in seen_position:
                seen_labels[row, seen_position[target]] = 1.0
        holdout_indices.append(
            [target_to_index[target] for target in targets if target in holdout_set]
        )
    observed = seen_metrics(result.scores[:, seen_indices], seen_labels)
    for percent in (0.01, 0.05, 0.10):
        k = max(1, int(len(target_names) * percent))
        successes = 0
        denominator = 0
        for vector, positives in zip(result.scores, holdout_indices):
            if not positives:
                continue
            top = np.argpartition(vector, -k)[-k:]
            successes += int(np.intersect1d(top, positives).size > 0)
            denominator += 1
        observed[f"holdout_top{int(percent * 100)}pct"] = successes / denominator
    expected = archived["final_test"]
    deltas = {key: abs(float(observed[key]) - float(expected[key])) for key in expected}
    result.to_wide().reset_index(names="drug").to_csv(
        args.output / "round261_article_test_scores.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    summary = {
        "status": "PASS" if max(deltas.values()) <= 1.0e-12 else "FAIL",
        "reference_sha256": sha256_file(reference),
        "checkpoint_matches_manifest": hash_matches_manifest(
            args, "checkpoints/selected/round261_proteomics_epoch070.pth"
        ),
        "preprocessor": str(preprocessor_path),
        "preprocessor_sha256": sha256_file(preprocessor_path),
        "selected_features": len(preprocessor.feature_names),
        "test_rows": len(test_frame),
        "observed": observed,
        "expected": expected,
        "max_metric_abs_delta": max(deltas.values()),
    }
    report_metrics(
        {"status": summary["status"], "test_rows": len(test_frame),
         "max_metric_abs_delta": summary["max_metric_abs_delta"]},
        label="Summary",
    )
    return summary


def verify_full_space(args: argparse.Namespace) -> dict[str, Any]:
    args.device = resolve_device(getattr(args, "device", "auto"))
    report_metrics(
        {"setting": "full-space verification", "device": args.device}, label="Setting"
    )
    report_step("Load locked full-space cache, splits and checkpoints")
    cache = args.package / "data/processed/fullspace/h07_prepared_data.pt"
    target_table = args.package / "data/processed/fullspace/round259_target_embedding.csv"
    checkpoint = args.package / "checkpoints/selected/round259_e13_bce001.pth"
    r230 = import_module(
        "article_exact_r230",
        args.compat_write
        / "claude_round230_scratch_mlp_metar_target_tower_pc3_20260518/scripts/"
        "run_round230_scratch_mlp_metar_target_tower_pc3.py",
    )
    r259 = import_module(
        "article_exact_r259",
        args.round259 / "scripts/run_round259_crosscell_fullspace_round252query.py",
    )
    payload = torch.load(cache, map_location="cpu", weights_only=False)
    protocol = r259.load_round252_protocol_audit()
    _splits, holdouts, counts = r259.build_round259_splits(
        r230, payload["data"], protocol
    )
    bundle = holdouts["holdout_test"]
    target_names, target_values = load_target_table(target_table)
    if tuple(map(str, payload["data"]["gene_list"])) != target_names:
        raise RuntimeError("target embedding table order differs from h07 cache")

    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_state = checkpoint_payload["model_state_dict"]
    query_state = {
        key.removeprefix("a01_encoder."): value
        for key, value in checkpoint_state.items()
        if key.startswith("a01_encoder.")
    }
    train_indices = (
        payload["data"]["train_gene_global_idx"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )
    report_step("Run the existing archived full-space evaluation")
    article_model = r259.build_model(
        r230,
        payload["data"],
        torch.device(args.device),
        train_indices,
        checkpoint_payload["cfg"],
        query_state,
    ).to(torch.device(args.device))
    article_model.load_state_dict(checkpoint_state, strict=True)
    article_model.eval()
    _summary, article_per = r259.evaluate_holdout_percent(
        article_model,
        bundle,
        torch.device(args.device),
        batch_size=256,
        percent=0.05,
        exclude_seen=False,
        return_rows=True,
    )
    if article_per is None:
        raise RuntimeError("original Round259 evaluation did not return per-profile rows")
    article_per.to_csv(
        args.output / "round259_original_code_rerun.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    del article_model
    torch.cuda.empty_cache()

    report_step("Score public full-space queries and compare the existing reference")
    query_ids = tuple(f"holdout_test:{index}" for index in range(len(bundle["x"])))
    result = predict_full_space(
        bundle["x"],
        query_ids,
        checkpoint,
        target_names,
        target_values,
        device=args.device,
    )
    k = max(1, int(math.floor(len(target_names) * 0.05)))
    rows = []
    for index, (vector, positives) in enumerate(
        zip(result.scores, bundle["true_indices"])
    ):
        true_indices = [int(value) for value in positives]
        top = np.argpartition(vector, -k)[-k:]
        ranks = [int((vector > vector[target]).sum()) + 1 for target in true_indices]
        rows.append(
            {
                "variant": "e13_bce001",
                "split": "holdout_test",
                "cell_split_role": "official_unseen_final",
                "row_index": index,
                "drug": str(bundle["df"].iloc[index]["drug"]),
                "cell_line": str(bundle["df"].iloc[index]["cell_line"]),
                "success_top5pct": bool(
                    np.intersect1d(top, true_indices).size > 0
                ),
                "best_rank": min(ranks) if ranks else -1,
                "n_true_targets": len(true_indices),
                "target_mode": (
                    "round259_round246query_crosscell_round240_fullspace_ppi_t4"
                ),
            }
        )
    observed = pd.DataFrame(rows)
    expected = pd.read_csv(
        args.package
        / "data/derived/article_exact/round259_archived_per_profile_holdout_metrics.tsv.gz",
        sep="\t",
    )
    expected = expected[
        expected["variant"].eq("e13_bce001")
        & expected["split"].eq("holdout_test")
    ].reset_index(drop=True)
    columns = [
        "row_index",
        "drug",
        "cell_line",
        "success_top5pct",
        "best_rank",
        "n_true_targets",
    ]
    exact = observed[columns].equals(expected[columns])
    success_exact = observed["success_top5pct"].equals(expected["success_top5pct"])
    public_rank_delta = (
        observed["best_rank"].astype(int) - expected["best_rank"].astype(int)
    ).abs()
    original_rerun_exact = article_per[columns].equals(expected[columns])
    original_rank_mismatches = int(
        (~article_per["best_rank"].eq(expected["best_rank"])).sum()
    )
    observed.to_csv(
        args.output / "round259_holdout_recomputed.tsv.gz", sep="\t", index=False
    )
    success_rate = float(observed["success_top5pct"].mean())
    expected_rate = float(expected["success_top5pct"].mean())
    article_rate = 0.3269101501232355
    del result, payload, bundle
    gc.collect()
    summary = {
        "status": "PASS"
        if success_exact and success_rate == expected_rate == article_rate
        else "FAIL",
        "cache_sha256": sha256_file(cache),
        "checkpoint_matches_manifest": hash_matches_manifest(
            args, "checkpoints/selected/round259_e13_bce001.pth"
        ),
        "counts": counts,
        "rows": len(observed),
        "per_profile_exact": exact,
        "per_profile_top5_outcomes_exact": success_exact,
        "public_rank_tie_mismatches": int((public_rank_delta != 0).sum()),
        "public_rank_tie_max_abs_delta": int(public_rank_delta.max()),
        "original_code_per_profile_exact": original_rerun_exact,
        "original_code_rank_mismatches": original_rank_mismatches,
        "observed_top5pct_success": success_rate,
        "expected_top5pct_success": expected_rate,
        "article_reported_metric_exact": success_rate == article_rate,
    }
    report_metrics(
        {"status": summary["status"], "rows": len(observed),
         "observed_top5pct_success": success_rate,
         "public_rank_tie_mismatches": summary["public_rank_tie_mismatches"]},
        label="Summary",
    )
    return summary
