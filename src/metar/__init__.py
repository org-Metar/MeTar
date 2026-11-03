"""Stable prediction and training interfaces for the article-used MeTar chain."""

from importlib import import_module

# Small maintenance tools can import metar.paths without importing PyTorch.
_MODULES = {
    "consensus_from_long": "consensus",
    "PredictionResult": "result",
    "ProteomicsPreprocessor": "preprocessing",
    "fit_round261_preprocessor": "preprocessing",
    "run_training_stage": "training",
    **{name: "predict" for name in (
        "predict_bulk_668", "predict_cross_cell", "predict_full_space",
        "predict_proteomics", "predict_single_cell",
    )},
}


def __getattr__(name: str):
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module(f".{_MODULES[name]}", __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "PredictionResult",
    "ProteomicsPreprocessor",
    "consensus_from_long",
    "fit_round261_preprocessor",
    "predict_bulk_668",
    "predict_cross_cell",
    "predict_full_space",
    "predict_proteomics",
    "predict_single_cell",
    "run_training_stage",
]

__version__ = "0.2.0"
