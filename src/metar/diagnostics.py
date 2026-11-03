"""Check prediction prerequisites and input schemas without loading a model."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .consensus import consensus_from_long
from .io import (load_feature_matrix, load_grouped_cells,
                 load_target_table, read_feature_order, read_table, split_targets)
from .preprocessing import ProteomicsPreprocessor


def check_prediction(args, defaults: dict, single_defaults: dict) -> dict[str, Any]:
    """Inspect required files and optionally validate a user's input matrix."""
    checks: list[dict[str, Any]] = []
    scenario = args.scenario

    def require(label: str, path: str | Path | None) -> Path | None:
        value = Path(path).expanduser().resolve() if path else None
        ok = value is not None and value.is_file() and value.stat().st_size > 0
        checks.append({"name": label, "path": str(value) if value else None,
                       "status": "ok" if ok else "missing"})
        return value if ok else None

    checkpoint_key = {"bulk-668": "closed_checkpoint", "cross-cell": "closed_checkpoint",
                      "full-space": "full_checkpoint", "proteomics": "proteomics_checkpoint"}
    if scenario in checkpoint_key:
        require("checkpoint", args.checkpoint or defaults[checkpoint_key[scenario]])
    elif scenario == "single-cell":
        if args.cell_line not in single_defaults:
            checks.append({"name": "cell_line", "status": "missing",
                           "detail": "Specify A549, SW480, or NCI-H460."})
        else:
            require("checkpoint", args.checkpoint or single_defaults[args.cell_line])

    if scenario == "cross-cell" and args.mode == "adapted":
        require("adapter_checkpoint", args.adapter_checkpoint or defaults["adapter_checkpoint"])
    target_path = None
    if scenario in {"full-space", "single-cell", "proteomics"}:
        key = "proteomics_target_table" if scenario == "proteomics" else "target_table"
        target_path = require("target_table", args.target_table or defaults[key])

    preprocessor_path = None
    if scenario == "proteomics":
        preprocessor_path = require("preprocessor", args.preprocessor or defaults["proteomics_preprocessor"])

    targets = None
    if target_path:
        try:
            targets, values = load_target_table(target_path)
            checks.append({"name": "target_schema", "status": "ok",
                           "shape": list(values.shape)})
        except (ValueError, OSError) as exc:
            checks.append({"name": "target_schema", "status": "invalid", "detail": str(exc)})

    inputs = args.input or []
    if not inputs:
        return {"status": "ok" if all(x["status"] == "ok" for x in checks) else "error",
                "scenario": scenario, "checks": checks,
                "input_checked": False, "model_loaded": False,
                "note": "No --input supplied: prerequisite files only. Normalization is not verified."}
    try:
        for value in inputs:
            if require("input", value) is None:
                raise ValueError(f"Input file is missing: {value}")
        if scenario == "consensus":
            result = consensus_from_long(pd.concat([read_table(p) for p in inputs], ignore_index=True),
                                         drug_column=args.drug_column,
                                         cell_line_column=args.cell_line_column,
                                         min_cell_lines=args.min_cell_lines)
            shape = [len(result.query_ids), len(result.targets)]
        else:
            if len(inputs) != 1:
                raise ValueError("Only consensus accepts multiple input files")
            if scenario == "single-cell":
                values, groups = load_grouped_cells(inputs[0], group_column=args.id_column)
                shape = list(values.shape)
                checks.append({"name": "query_groups", "status": "ok", "count": len(set(groups))})
            elif scenario == "proteomics":
                if not preprocessor_path:
                    raise ValueError("A valid --preprocessor is needed to check protein features")
                frame = read_table(inputs[0])
                preprocessor = ProteomicsPreprocessor.read(preprocessor_path)
                values = preprocessor.transform(frame)
                shape = list(values.shape)
            else:
                order_path = args.feature_order
                if not order_path and Path(defaults["feature_order"]).is_file():
                    order_path = defaults["feature_order"]
                order = read_feature_order(order_path, column=args.feature_column) if order_path else None
                values, ids = load_feature_matrix(inputs[0], expected_dim=978,
                                                  id_column=args.id_column, feature_order=order)
                shape = list(values.shape)
                if scenario == "cross-cell" and args.mode == "adapted":
                    support_path = require("support_input", args.support_input)
                    label_path = require("support_labels", args.support_labels)
                    if support_path is None or label_path is None:
                        raise ValueError("Adapted prediction needs --support-input and --support-labels")
                    _, support_ids = load_feature_matrix(support_path, expected_dim=978,
                                                          id_column=args.id_column, feature_order=order)
                    if set(ids) & set(support_ids):
                        raise ValueError("Query and reference compound identifiers must be disjoint")
                    # Read the selected label vocabulary without loading a checkpoint.
                    label_frame = read_table(label_path)
                    if args.id_column not in label_frame or args.target_column not in label_frame:
                        raise ValueError("Support labels lack the configured ID/target columns")
                    if label_frame[args.id_column].astype(str).duplicated().any():
                        raise ValueError("Support label identifiers must be unique")
                    if not set(support_ids).issubset(set(label_frame[args.id_column].astype(str))):
                        raise ValueError("Support features do not all have target annotations")
                    label_map = dict(zip(label_frame[args.id_column].astype(str),
                                         label_frame[args.target_column]))
                    if any(not split_targets(label_map[key]) for key in support_ids):
                        raise ValueError("Each support compound needs a nonempty target annotation")
            if not np.isfinite(values).all():
                raise ValueError("Input contains nonfinite values")
        checks.append({"name": "input_schema", "status": "ok", "shape": shape})
    except (ValueError, OSError, KeyError, RuntimeError, ImportError, SyntaxError, TypeError) as exc:
        checks.append({"name": "input_schema", "status": "invalid", "detail": str(exc)})
    return {"status": "ok" if all(x["status"] == "ok" for x in checks) else "error",
            "scenario": scenario, "checks": checks, "input_checked": True,
            "model_loaded": False,
            "note": "Checks file presence, dimensions and finite values; cannot infer whether normalization matches training."}
