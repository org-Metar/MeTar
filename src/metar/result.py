"""Prediction result container and deterministic ranking serialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PredictionResult:
    """Scores and metadata for one or more MeTar queries."""

    scores: np.ndarray
    targets: tuple[str, ...]
    query_ids: tuple[str, ...]
    scenario: str
    metadata: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        scores = np.asarray(self.scores)
        if scores.ndim != 2:
            raise ValueError(f"scores must be two-dimensional, observed {scores.shape}")
        if not np.isfinite(scores).all():
            raise ValueError("scores must be finite before ranking")
        if scores.shape != (len(self.query_ids), len(self.targets)):
            raise ValueError(
                "score shape does not match query/target labels: "
                f"{scores.shape} != ({len(self.query_ids)}, {len(self.targets)})"
            )
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("target labels must be unique")
        if len(set(self.query_ids)) != len(self.query_ids):
            raise ValueError("query identifiers must be unique")
        if not self.metadata.empty and len(self.metadata) != len(self.query_ids):
            raise ValueError("metadata row count must match query count")

    def to_long(self, top_k: int = 0) -> pd.DataFrame:
        """Return stable descending rankings; ``top_k=0`` retains all targets."""

        n_targets = len(self.targets)
        keep = n_targets if top_k <= 0 else min(int(top_k), n_targets)
        rows: list[dict[str, object]] = []
        metadata = self.metadata.reset_index(drop=True)
        for query_index, query_id in enumerate(self.query_ids):
            values = np.asarray(self.scores[query_index], dtype=np.float64)
            order = np.argsort(-values, kind="mergesort")[:keep]
            base: dict[str, object] = {
                "query_id": query_id,
                "scenario": self.scenario,
                "candidate_count": n_targets,
            }
            if not metadata.empty:
                base.update(metadata.iloc[query_index].to_dict())
            for rank, target_index in enumerate(order, start=1):
                rows.append(
                    {
                        **base,
                        "target": self.targets[int(target_index)],
                        "score": float(values[int(target_index)]),
                        "rank": rank,
                        "rank_fraction": rank / float(n_targets),
                        "rank_percent": 100.0 * rank / float(n_targets),
                    }
                )
        return pd.DataFrame(rows)

    def to_wide(self) -> pd.DataFrame:
        metadata = self.metadata.reset_index(drop=True).copy()
        if metadata.empty:
            metadata = pd.DataFrame(index=range(len(self.query_ids)))
        metadata.insert(0, "scenario", self.scenario)
        metadata.insert(0, "query_id", self.query_ids)
        score_frame = pd.DataFrame(self.scores, columns=self.targets)
        return pd.concat([metadata.reset_index(drop=True), score_frame], axis=1)

    def write(
        self, path: str | Path, *, output_format: str = "long", top_k: int = 0
    ) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame = self.to_long(top_k=top_k) if output_format == "long" else self.to_wide()
        suffixes = "".join(output.suffixes).lower()
        if suffixes.endswith((".tsv", ".tsv.gz")):
            frame.to_csv(output, sep="\t", index=False)
        elif suffixes.endswith((".csv", ".csv.gz")):
            frame.to_csv(output, index=False)
        elif suffixes.endswith(".parquet"):
            frame.to_parquet(output, index=False)
        else:
            raise ValueError(f"unsupported output suffix: {output}")
        return output
