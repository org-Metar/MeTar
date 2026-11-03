"""Compute cross-cell similarity, target overlap and recovery from direct model scores."""
from __future__ import annotations

import argparse
import math
import random
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(mat, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return mat / denom

def sem(values: np.ndarray | pd.Series) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) <= 1:
        return float("nan")
    return float(arr.std(ddof=1) / math.sqrt(len(arr)))

def load_inputs() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows = pd.read_csv(ROWS_PATH, sep="\t")
    score_npz = np.load(SCORE_NPZ, allow_pickle=True)
    expression_npz = np.load(EXPRESSION_NPZ, allow_pickle=True)
    score = score_npz["noadapt_scores"].astype(np.float64)
    expression = expression_npz["x_standardized"].astype(np.float64)
    if len(rows) != score.shape[0] or score.shape[1] != 668:
        raise RuntimeError(f"Unexpected no-adapt score shape: rows={len(rows)}, score={score.shape}")
    if len(rows) != expression.shape[0] or expression.shape[1] != 978:
        raise RuntimeError(f"Unexpected standardized expression shape: rows={len(rows)}, expression={expression.shape}")
    return rows, score, expression

def collapse_to_drugcell(rows: pd.DataFrame, score_profile: np.ndarray, expression_profile: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    meta_rows: list[dict[str, object]] = []
    score_parts: list[np.ndarray] = []
    expression_parts: list[np.ndarray] = []
    for (drug, cell), idx in rows.groupby(["drug", "cell_line"], sort=True).indices.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        first = rows.iloc[idx_arr[0]]
        meta_rows.append(
            {
                "drug": str(drug),
                "cell_line": str(cell),
                "n_profiles": int(len(idx_arr)),
                "true_targets": str(first["true_targets"]),
            }
        )
        score_parts.append(score_profile[idx_arr].mean(axis=0))
        expression_parts.append(expression_profile[idx_arr].mean(axis=0))
    return pd.DataFrame(meta_rows), np.vstack(score_parts), np.vstack(expression_parts)

def attach_noadapt_metrics(meta: pd.DataFrame) -> pd.DataFrame:
    metrics = pd.read_csv(METRICS_PATH, sep="\t")
    metrics = metrics[metrics["method"].astype(str).eq("Round246 no-adapt")].copy()
    keep = [
        "drug",
        "cell_line",
        "Hit@10",
        "Hit@30",
        "best_true_target_rank",
        "best_true_target_percentile",
        "AUROC",
        "NDCG@10",
        "MeanAP",
    ]
    merged = meta.merge(metrics[keep], on=["drug", "cell_line"], how="left", validate="one_to_one")
    if merged[["Hit@10", "Hit@30", "best_true_target_percentile"]].isna().any().any():
        raise RuntimeError("Missing no-adapt metrics after merge")
    return merged

def make_pairs(meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(SEED)
    by_cell = {cell: sub.index.tolist() for cell, sub in meta.groupby("cell_line")}
    drug_to_indices = {drug: sub.index.tolist() for drug, sub in meta.groupby("drug")}
    same_records: list[dict[str, object]] = []
    control_records: list[dict[str, object]] = []
    for drug, idxs in drug_to_indices.items():
        if len(idxs) < 2:
            continue
        ordered = sorted(idxs, key=lambda idx: str(meta.loc[idx, "cell_line"]))
        for i, j in combinations(ordered, 2):
            pair_id = len(same_records)
            same_records.append(
                {
                    "pair_id": pair_id,
                    "drug": str(drug),
                    "cell_line_1": meta.loc[i, "cell_line"],
                    "cell_line_2": meta.loc[j, "cell_line"],
                    "index_1": int(i),
                    "index_2": int(j),
                }
            )
            candidates = [idx for idx in by_cell[meta.loc[j, "cell_line"]] if meta.loc[idx, "drug"] != drug]
            ctrl_j = rng.choice(candidates)
            control_records.append(
                {
                    "pair_id": pair_id,
                    "drug_1": str(drug),
                    "drug_2": meta.loc[ctrl_j, "drug"],
                    "cell_line_1": meta.loc[i, "cell_line"],
                    "cell_line_2": meta.loc[j, "cell_line"],
                    "index_1": int(i),
                    "control_index_2": int(ctrl_j),
                }
            )
    return pd.DataFrame(same_records), pd.DataFrame(control_records)

def cosine_tables(
    meta: pd.DataFrame,
    score_mat: np.ndarray,
    expression_mat: np.ndarray,
    same_pairs: pd.DataFrame,
    control_pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_norm = l2_normalize(score_mat)
    expression_norm = l2_normalize(expression_mat)
    same_records: list[dict[str, object]] = []
    control_records: list[dict[str, object]] = []
    for same, control in zip(same_pairs.itertuples(index=False), control_pairs.itertuples(index=False)):
        i = same.index_1
        j = same.index_2
        ctrl_j = control.control_index_2
        same_records.append(
            {
                "pair_id": int(same.pair_id),
                "drug": same.drug,
                "cell_line_1": same.cell_line_1,
                "cell_line_2": same.cell_line_2,
                "score_cosine": float(np.dot(score_norm[i], score_norm[j])),
                "standardized_expression_cosine": float(np.dot(expression_norm[i], expression_norm[j])),
            }
        )
        control_records.append(
            {
                "pair_id": int(control.pair_id),
                "drug_1": control.drug_1,
                "drug_2": control.drug_2,
                "cell_line_1": control.cell_line_1,
                "cell_line_2": control.cell_line_2,
                "score_cosine": float(np.dot(score_norm[i], score_norm[ctrl_j])),
                "standardized_expression_cosine": float(np.dot(expression_norm[i], expression_norm[ctrl_j])),
            }
        )

    same_df = pd.DataFrame(same_records)
    control_df = pd.DataFrame(control_records)
    summary_rows: list[dict[str, object]] = []
    for vector_type, col in [
        ("Direct MeTar target-score", "score_cosine"),
        ("Standardized L1000 input", "standardized_expression_cosine"),
    ]:
        for comparison, df in [
            ("same_drug_cross_cell", same_df),
            ("different_drug_matched_control", control_df),
        ]:
            values = df[col].astype(float).to_numpy()
            summary_rows.append(
                {
                    "vector_type": vector_type,
                    "comparison": comparison,
                    "n_pairs": int(len(values)),
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "sem": sem(values),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                }
            )
        merged = same_df[["pair_id", col]].merge(
            control_df[["pair_id", col]],
            on="pair_id",
            suffixes=("_same", "_control"),
            validate="one_to_one",
        )
        wins = (merged[f"{col}_same"] > merged[f"{col}_control"]).astype(float).to_numpy()
        summary_rows.append(
            {
                "vector_type": vector_type,
                "comparison": "paired_same_greater_than_control",
                "n_pairs": int(len(wins)),
                "mean": float(wins.mean()),
                "median": float(np.median(wins)),
                "sem": sem(wins),
                "q25": float(np.quantile(wins, 0.25)),
                "q75": float(np.quantile(wins, 0.75)),
            }
        )
    return same_df, control_df, pd.DataFrame(summary_rows)

def top_target_overlap(score_mat: np.ndarray, same_pairs: pd.DataFrame, control_pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    for same, control in zip(same_pairs.itertuples(index=False), control_pairs.itertuples(index=False)):
        i = same.index_1
        j = same.index_2
        ctrl_j = control.control_index_2
        for k in [10, 30, 50]:
            top_i = set(np.argpartition(-score_mat[i], k - 1)[:k].tolist())
            top_j = set(np.argpartition(-score_mat[j], k - 1)[:k].tolist())
            top_control = set(np.argpartition(-score_mat[ctrl_j], k - 1)[:k].tolist())
            for comparison, other in [
                ("same_drug_cross_cell", top_j),
                ("different_drug_matched_control", top_control),
            ]:
                inter = len(top_i & other)
                union = len(top_i | other)
                records.append(
                    {
                        "comparison": comparison,
                        "pair_id": int(same.pair_id),
                        "k": int(k),
                        "overlap_count": int(inter),
                        "jaccard": inter / union if union else 0.0,
                    }
                )
    pair_df = pd.DataFrame(records)
    summary = (
        pair_df.groupby(["comparison", "k"])["jaccard"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .rename(columns={"count": "n_pairs", "mean": "mean_jaccard", "median": "median_jaccard"})
    )
    summary["sem_jaccard"] = [
        sem(pair_df[(pair_df["comparison"] == row.comparison) & (pair_df["k"] == row.k)]["jaccard"])
        for row in summary.itertuples(index=False)
    ]
    return pair_df, summary

def recovery_states(meta: pd.DataFrame, same_pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    for row in same_pairs.itertuples(index=False):
        a = meta.loc[row.index_1]
        b = meta.loc[row.index_2]
        for threshold, col in [("Hit@10", "Hit@10"), ("Hit@30", "Hit@30")]:
            hit_a = bool(int(a[col]))
            hit_b = bool(int(b[col]))
            if hit_a and hit_b:
                state = "both recovered"
            elif hit_a or hit_b:
                state = "one context only"
            else:
                state = "neither recovered"
            records.append(
                {
                    "pair_id": int(row.pair_id),
                    "drug": row.drug,
                    "threshold": threshold,
                    "state": state,
                }
            )
    states = pd.DataFrame(records)
    summary = states.groupby(["threshold", "state"]).size().reset_index(name="n_pairs")
    total = states.groupby("threshold").size().reset_index(name="total_pairs")
    summary = summary.merge(total, on="threshold", validate="many_to_one")
    summary["fraction"] = summary["n_pairs"] / summary["total_pairs"]
    order = ["both recovered", "one context only", "neither recovered"]
    full_rows: list[dict[str, object]] = []
    for threshold in ["Hit@10", "Hit@30"]:
        total_pairs = int(total.loc[total["threshold"] == threshold, "total_pairs"].iloc[0])
        for state in order:
            sub = summary[(summary["threshold"] == threshold) & (summary["state"] == state)]
            if sub.empty:
                full_rows.append(
                    {
                        "threshold": threshold,
                        "state": state,
                        "n_pairs": 0,
                        "total_pairs": total_pairs,
                        "fraction": 0.0,
                    }
                )
            else:
                full_rows.append(sub.iloc[0].to_dict())
    return states, pd.DataFrame(full_rows)

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--rows",type=Path,required=True)
    parser.add_argument("--scores",type=Path,required=True)
    parser.add_argument("--expression",type=Path,required=True)
    parser.add_argument("--metrics",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=20260608)
    args=parser.parse_args()
    global ROWS_PATH,SCORE_NPZ,EXPRESSION_NPZ,METRICS_PATH,SOURCE,SEED
    ROWS_PATH=args.rows
    SCORE_NPZ=args.scores
    EXPRESSION_NPZ=args.expression
    METRICS_PATH=args.metrics
    SOURCE=args.output_dir
    SEED=args.seed
    args.output_dir.mkdir(parents=True,exist_ok=False)
    rows, score_profile, expression_profile = load_inputs()
    meta, score_drugcell, expression_drugcell = collapse_to_drugcell(rows, score_profile, expression_profile)
    meta = attach_noadapt_metrics(meta)
    same_pairs, control_pairs = make_pairs(meta)
    same_cos, control_cos, cos_summary = cosine_tables(meta, score_drugcell, expression_drugcell, same_pairs, control_pairs)
    top_pair, top_summary = top_target_overlap(score_drugcell, same_pairs, control_pairs)
    state_pair, state_summary = recovery_states(meta, same_pairs)

    meta.to_csv(SOURCE / "Supplementary_Figure_S2_drugcell_vector_metadata_direct_score.tsv", sep="\t", index=False)
    same_cos.to_csv(SOURCE / "Supplementary_Figure_S2a_same_drug_crosscell_pair_cosine_direct_score.tsv", sep="\t", index=False)
    control_cos.to_csv(SOURCE / "Supplementary_Figure_S2a_matched_different_drug_control_cosine_direct_score.tsv", sep="\t", index=False)
    cos_summary.to_csv(SOURCE / "Supplementary_Figure_S2a_expression_direct_score_cosine_summary.tsv", sep="\t", index=False)
    top_pair.to_csv(SOURCE / "Supplementary_Figure_S2b_top_target_overlap_pairlevel_direct_score.tsv", sep="\t", index=False)
    top_summary.to_csv(SOURCE / "Supplementary_Figure_S2b_top_target_overlap_summary_direct_score.tsv", sep="\t", index=False)
    state_pair.to_csv(SOURCE / "Supplementary_Figure_S2c_recovery_pair_states_direct_score.tsv", sep="\t", index=False)
    state_summary.to_csv(SOURCE / "Supplementary_Figure_S2c_recovery_pair_state_summary_direct_score.tsv", sep="\t", index=False)



if __name__ == "__main__":
    main()
