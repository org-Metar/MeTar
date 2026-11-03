"""Calculate shared-target enrichment among response-score nearest neighbors."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

def parse_targets(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    text = str(value).strip()
    if not text:
        return set()
    return {part.strip() for part in text.replace(";", "|").replace(",", "|").split("|") if part.strip()}

def binomial_se(p: float, n: int) -> float:
    if n <= 0:
        return float("nan")
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)

def build_bulk() -> dict:
    burden = pd.read_csv(FIG2_BURDEN, sep="\t")
    sim_df = pd.read_csv(BULK_SIM, sep="\t")
    drugs = sim_df["drug"].tolist()
    sim = sim_df.drop(columns=["drug"]).to_numpy(dtype=float)
    target_sets = {
        row.drug: parse_targets(row.targets_in_668)
        for row in burden[["drug", "targets_in_668"]].itertuples(index=False)
    }
    eligible = [d for d in drugs if target_sets.get(d)]
    eligible_set = set(eligible)
    eligible_idx = np.array([drugs.index(d) for d in eligible], dtype=int)

    # Precompute shared-target adjacency among eligible drugs.
    shared = np.zeros((len(eligible), len(eligible)), dtype=bool)
    for i, d1 in enumerate(eligible):
        t1 = target_sets[d1]
        for j, d2 in enumerate(eligible):
            if i != j and t1 & target_sets[d2]:
                shared[i, j] = True

    k_values = [1, 3, 5, 10, 20, 50]
    rng = np.random.default_rng(20260514)
    n_perm = 500
    rows = []
    detail_rows = []
    for k in k_values:
        obs_hits = []
        rand_hits_per_query = np.zeros(len(eligible), dtype=float)
        for qi, d in enumerate(eligible):
            global_i = drugs.index(d)
            order = np.argsort(-sim[global_i, :])
            nn = [drugs[j] for j in order if drugs[j] != d and drugs[j] in eligible_set][:k]
            nn_local = [eligible.index(x) for x in nn]
            obs = int(shared[qi, nn_local].any()) if nn_local else 0
            obs_hits.append(obs)
            choices = np.array([j for j in range(len(eligible)) if j != qi], dtype=int)
            perm_hits = []
            for _ in range(n_perm):
                sampled = rng.choice(choices, size=min(k, len(choices)), replace=False)
                perm_hits.append(int(shared[qi, sampled].any()))
            rand_hits_per_query[qi] = float(np.mean(perm_hits))
            detail_rows.append(
                {
                    "drug": d,
                    "k": k,
                    "observed_any_shared_target_neighbor": obs,
                    "permutation_random_rate": rand_hits_per_query[qi],
                    "nearest_neighbors": "|".join(nn),
                    "targets_in_668": "|".join(sorted(target_sets[d])),
                    "n_targets_in_668": len(target_sets[d]),
                }
            )
        obs_rate = float(np.mean(obs_hits))
        rand_rate = float(np.mean(rand_hits_per_query))
        rows.append(
            {
                "k": k,
                "n_query_drugs": len(eligible),
                "observed_any_shared_target_neighbor_rate": obs_rate,
                "observed_se": binomial_se(obs_rate, len(eligible)),
                "permutation_random_rate_mean": rand_rate,
                "permutation_random_rate_sd_across_queries": float(np.std(rand_hits_per_query, ddof=1)),
                "fold_over_random": obs_rate / rand_rate if rand_rate > 0 else np.nan,
                "n_permutations_per_query": n_perm,
            }
        )

    enrich = pd.DataFrame(rows)
    details = pd.DataFrame(detail_rows)
    enrich.to_csv(SOURCE / "bulk_nearest_neighbor_shared_target_enrichment.tsv", sep="\t", index=False)
    details.to_csv(SOURCE / "bulk_nearest_neighbor_details.tsv", sep="\t", index=False)

    # Top-40 compact heatmap subset prioritizes drugs whose top-1 neighbor shares a target.
    top1 = details[details["k"] == 1].copy()
    top1["top1_similarity"] = top1.apply(
        lambda r: sim[drugs.index(r["drug"]), drugs.index(str(r["nearest_neighbors"]).split("|")[0])]
        if str(r["nearest_neighbors"])
        else np.nan,
        axis=1,
    )
    selected = top1.sort_values(
        ["observed_any_shared_target_neighbor", "top1_similarity", "n_targets_in_668"],
        ascending=[False, False, True],
    ).head(40)["drug"].tolist()
    heat = sim_df[["drug"] + selected]
    heat = heat[heat["drug"].isin(selected)].copy()
    heat.to_csv(SOURCE / "bulk_similarity_heatmap_top40.tsv", sep="\t", index=False)

    return {
        "n_drugs": len(drugs),
        "n_eligible_annotated_drugs": len(eligible),
        "candidate_space": "668-label Fig.2 common-intersection space",
        "source_matrix": str(BULK_SIM),
    }


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--similarity",type=Path,required=True)
    parser.add_argument("--target-annotations",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    global FIG2_BURDEN,BULK_SIM,SOURCE
    FIG2_BURDEN=args.target_annotations
    BULK_SIM=args.similarity
    SOURCE=args.output_dir
    SOURCE.mkdir(parents=True,exist_ok=False)
    result=build_bulk()
    print(f"Evaluated {result['n_eligible_annotated_drugs']} annotated query drugs")


if __name__ == "__main__":
    main()
