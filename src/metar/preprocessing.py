"""Deterministic preprocessing metadata for the Round261 proteomics interface."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .io import split_targets, validate_identifiers


@dataclass(frozen=True)
class ProteomicsPreprocessor:
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    log1p: bool
    training_seed: int = 2026
    validation_fraction: float = 0.2
    top_k_features: int = 2000

    def __post_init__(self) -> None:
        size = len(self.feature_names)
        if size == 0 or len(self.mean) != size or len(self.std) != size:
            raise ValueError(
                "proteomics preprocessing fields have inconsistent lengths"
            )
        validate_identifiers(self.feature_names, label="proteomics feature names")
        if not np.isfinite(self.mean).all():
            raise ValueError("proteomics preprocessing means must be finite")
        if any(value <= 0 or not np.isfinite(value) for value in self.std):
            raise ValueError(
                "proteomics standard deviations must be finite and positive"
            )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [name for name in self.feature_names if name not in frame.columns]
        if missing:
            raise ValueError(
                f"proteomics input is missing {len(missing)} selected features, e.g. {missing[:5]}"
            )
        values = (
            frame[list(self.feature_names)]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        if self.log1p:
            if np.any(values < 0):
                raise ValueError(
                    "the locked Round261 preprocessor requires nonnegative input before log1p"
                )
            values = np.log1p(values)
        transformed = (values - np.asarray(self.mean, dtype=np.float32)) / np.asarray(
            self.std, dtype=np.float32
        )
        if not np.isfinite(transformed).all():
            raise ValueError("proteomics preprocessing produced nonfinite features")
        return transformed

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        return output

    @classmethod
    def read(cls, path: str | Path) -> ProteomicsPreprocessor:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["feature_names"] = tuple(data["feature_names"])
        data["mean"] = tuple(map(float, data["mean"]))
        data["std"] = tuple(map(float, data["std"]))
        return cls(**data)


def _read_round261_reference(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    separator = (
        "\t" if "".join(source.suffixes).lower().endswith((".tsv", ".tsv.gz")) else ","
    )
    return pd.read_csv(source, sep=separator, index_col=0)


def fit_round261_preprocessor(
    reference_path: str | Path,
    target_genes: tuple[str, ...],
    *,
    seed: int = 2026,
    validation_fraction: float = 0.2,
    top_k_features: int = 2000,
) -> ProteomicsPreprocessor:
    """Rebuild the train-only preprocessing state used by the selected Round261 run."""

    frame = _read_round261_reference(reference_path)
    if "drug" not in frame.columns or "target" not in frame.columns:
        raise ValueError("Round261 reference table requires drug and target columns")
    target_set = set(target_genes)
    frame = frame.copy()
    frame["target_filt"] = frame["target"].map(
        lambda value: [
            target for target in split_targets(value) if target in target_set
        ]
    )
    frame = frame[frame["target_filt"].map(bool)].copy().reset_index(drop=True)
    frame["drug"] = frame["drug"].astype(str).str.strip().str.lower()
    drug_targets = {
        drug: set().union(*group["target_filt"].map(set))
        for drug, group in frame.groupby("drug")
    }
    all_drugs = np.asarray(sorted(drug_targets))
    outer_rng = np.random.RandomState(0)
    outer_rng.shuffle(all_drugs)
    n_test = max(1, int(len(all_drugs) * 0.2))
    train_drugs_full = all_drugs[n_test:]
    split_rng = np.random.RandomState(seed)
    split_drugs = np.asarray(train_drugs_full).copy()
    split_rng.shuffle(split_drugs)
    n_validation = max(1, int(len(split_drugs) * validation_fraction))
    raw_validation = split_drugs[:n_validation]
    raw_training = split_drugs[n_validation:]
    training_targets = set().union(*(drug_targets[drug] for drug in raw_training))
    validation_drugs = np.asarray(
        [drug for drug in raw_validation if drug_targets[drug] & training_targets]
    )
    if len(validation_drugs) == 0:
        validation_drugs = np.asarray(raw_validation[:1])
    training_drugs = np.asarray(
        [drug for drug in split_drugs if drug not in set(validation_drugs)]
    )
    train_rows = frame[frame["drug"].isin(training_drugs)]

    feature_names = [
        name for name in frame.columns if name not in {"drug", "target", "target_filt"}
    ]
    raw = (
        frame[feature_names]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    train_values = raw[train_rows.index.to_numpy()]
    keep_count = min(int(top_k_features), train_values.shape[1])
    variance = train_values.var(axis=0)
    kept_indices = np.sort(np.argsort(variance)[::-1][:keep_count])
    selected_names = tuple(feature_names[index] for index in kept_indices)
    selected_train = train_values[:, kept_indices]
    use_log1p = bool(selected_train.min() >= 0)
    if use_log1p:
        selected_train = np.log1p(selected_train)
    mean = selected_train.mean(axis=0)
    std = selected_train.std(axis=0) + 1.0e-6
    return ProteomicsPreprocessor(
        feature_names=selected_names,
        mean=tuple(map(float, mean)),
        std=tuple(map(float, std)),
        log1p=use_log1p,
        training_seed=seed,
        validation_fraction=validation_fraction,
        top_k_features=top_k_features,
    )
