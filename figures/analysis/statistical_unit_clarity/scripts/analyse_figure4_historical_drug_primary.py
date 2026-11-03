from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "data/plot_sources/figure4/Figure4_drug_level_percentile_distribution_source.tsv"

BOOTSTRAP_RESAMPLES = 20_000
RANDOM_SEED = 20_260_713
PRIMARY_CUTOFF_PERCENT = 10.0


def bootstrap_interval(values: np.ndarray, indices: np.ndarray) -> tuple[float, float]:
    draws = values[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate paired drug-level recovery statistics.")
    parser.add_argument("--input", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=Path("results/figure4_statistics"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input, sep="\t")
    if len(data) != 177 or data["drug"].nunique() != 177:
        raise RuntimeError("The historical drug-mean source must contain 177 unique paired drugs.")

    metar_percentile = data["metar_best_percentile_percent"].to_numpy(dtype=float)
    pertkge_percentile = data["pertkge_best_percentile_percent"].to_numpy(dtype=float)
    metar_hit = metar_percentile <= PRIMARY_CUTOFF_PERCENT
    pertkge_hit = pertkge_percentile <= PRIMARY_CUTOFF_PERCENT

    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.integers(0, len(data), size=(BOOTSTRAP_RESAMPLES, len(data)))
    metar_rate = metar_hit.astype(float) * 100.0
    pertkge_rate = pertkge_hit.astype(float) * 100.0
    paired_difference = metar_rate - pertkge_rate

    metar_ci = bootstrap_interval(metar_rate, indices)
    pertkge_ci = bootstrap_interval(pertkge_rate, indices)
    difference_ci = bootstrap_interval(paired_difference, indices)

    metar_only = int(np.sum(metar_hit & ~pertkge_hit))
    pertkge_only = int(np.sum(~metar_hit & pertkge_hit))
    both = int(np.sum(metar_hit & pertkge_hit))
    neither = int(np.sum(~metar_hit & ~pertkge_hit))
    discordant = metar_only + pertkge_only
    mcnemar_p = float(binomtest(min(metar_only, pertkge_only), discordant, 0.5).pvalue)

    continuous_gain = pertkge_percentile - metar_percentile
    wilcoxon_result = wilcoxon(continuous_gain, alternative="two-sided", zero_method="wilcox")

    rows = [
        {
            "endpoint": "Historical drug-mean top-10% recovery",
            "unit": "drug",
            "n": 177,
            "method": "MeTar",
            "estimate_percent": float(metar_rate.mean()),
            "ci95_low_percent": metar_ci[0],
            "ci95_high_percent": metar_ci[1],
            "comparison": "",
            "effect_percent_points": np.nan,
            "effect_ci95_low_percent_points": np.nan,
            "effect_ci95_high_percent_points": np.nan,
            "test": "",
            "p_value": np.nan,
        },
        {
            "endpoint": "Historical drug-mean top-10% recovery",
            "unit": "drug",
            "n": 177,
            "method": "PertKGE",
            "estimate_percent": float(pertkge_rate.mean()),
            "ci95_low_percent": pertkge_ci[0],
            "ci95_high_percent": pertkge_ci[1],
            "comparison": "",
            "effect_percent_points": np.nan,
            "effect_ci95_low_percent_points": np.nan,
            "effect_ci95_high_percent_points": np.nan,
            "test": "",
            "p_value": np.nan,
        },
        {
            "endpoint": "Historical drug-mean top-10% recovery",
            "unit": "drug",
            "n": 177,
            "method": "",
            "estimate_percent": np.nan,
            "ci95_low_percent": np.nan,
            "ci95_high_percent": np.nan,
            "comparison": "MeTar minus PertKGE",
            "effect_percent_points": float(paired_difference.mean()),
            "effect_ci95_low_percent_points": difference_ci[0],
            "effect_ci95_high_percent_points": difference_ci[1],
            "test": "two-sided exact McNemar",
            "p_value": mcnemar_p,
        },
    ]
    pd.DataFrame(rows).to_csv(args.output_dir / "drug_recovery_statistics.tsv", sep="\t", index=False)

    payload = {
        "source": args.input.name,
        "endpoint_definition": "Historical per-drug mean-embedding best-test-target percentile at the top-10% cutoff",
        "independent_unit": "drug",
        "n_drugs": 177,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": RANDOM_SEED,
        "metar_hits": int(metar_hit.sum()),
        "pertkge_hits": int(pertkge_hit.sum()),
        "both_hit": both,
        "metar_only": metar_only,
        "pertkge_only": pertkge_only,
        "neither_hit": neither,
        "metar_recovery_percent": float(metar_rate.mean()),
        "metar_ci95_percent": list(metar_ci),
        "pertkge_recovery_percent": float(pertkge_rate.mean()),
        "pertkge_ci95_percent": list(pertkge_ci),
        "paired_difference_percent_points": float(paired_difference.mean()),
        "paired_difference_ci95_percent_points": list(difference_ci),
        "exact_mcnemar_p": mcnemar_p,
        "continuous_secondary": {
            "metar_median_percentile_percent": float(np.median(metar_percentile)),
            "pertkge_median_percentile_percent": float(np.median(pertkge_percentile)),
            "mean_pertkge_minus_metar_percentile_points": float(continuous_gain.mean()),
            "metar_better_drugs": int(np.sum(continuous_gain > 0)),
            "pertkge_better_drugs": int(np.sum(continuous_gain < 0)),
            "wilcoxon_statistic": float(wilcoxon_result.statistic),
            "wilcoxon_p": float(wilcoxon_result.pvalue),
        },
    }
    (args.output_dir / "drug_recovery_statistics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
