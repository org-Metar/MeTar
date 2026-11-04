#!/usr/bin/env python3
"""Inspect, check, or render explicitly selected MeTar manuscript plots.

Listing and checking use the standard library only. Rendering is opt-in and
uses archived plotting functions and publication tables, without inference.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "figures/plot_catalog.json"
DATA = ROOT / "data/plot_sources"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load plot module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_entry(entry: dict, catalog: dict, assets: Path | None) -> dict:
    failures = []
    checked = []
    for relative in sorted(set(entry["scripts"] + entry["inputs"])):
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing repository input: {relative}")
            continue
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=relative)
            except SyntaxError as error:
                failures.append(f"invalid Python syntax: {relative}: {error}")
        checked.append(relative)
    for item in entry["external_assets"]:
        path = assets / item["name"] if assets is not None else None
        if path is None or not path.is_file():
            failures.append(f"external asset required: {item['name']} ({item['role']})")
        elif sha256(path) != item["sha256"]:
            failures.append(f"external asset hash mismatch: {item['name']}")
        else:
            checked.append(str(path))
    return {"id": entry["id"], "ok": not failures, "checked": checked,
            "failures": failures, "rendering_executed": False,
            "scope": entry["scope"]}


def configure_fonts(font_dir: Path | None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager

    if font_dir is not None:
        if not font_dir.is_dir():
            raise FileNotFoundError(f"Font directory not found: {font_dir}")
        for path in sorted(font_dir.iterdir()):
            if path.suffix.lower() in {".ttf", ".otf"}:
                font_manager.fontManager.addfont(str(path))
    try:
        font_path = font_manager.findfont("Arial", fallback_to_default=False)
    except ValueError as error:
        raise RuntimeError("The archived layout requires Arial. Install Arial or pass --font-dir.") from error
    return Path(font_path)


def render_fig2g(output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    source = ROOT / "figures/manuscript/figure2"
    load_module(source / "statistical_panel_style.py", "statistical_panel_style")
    spec = load_module(source / "panel_layout.py", "panel_layout")
    drawing = load_module(source / "family_composition.py", "family_composition")
    spec.HEIGHT_MM = 74.0
    # Preserve the existing native panel coordinates, including its 1.1 mm offset.
    spec.ROW_BOTTOMS["g"] = 1.1
    spec.configure()
    heat = pd.read_csv(DATA / "figure2g/moa_family_heatmap.tsv", sep="\t")
    permutations = pd.read_csv(DATA / "figure2g/family_cluster_permutation.tsv", sep="\t")
    keys = pd.MultiIndex.from_product(
        [drawing.MOAS, drawing.FAMILIES], names=["moa_class", "family"])
    selected = heat.loc[heat.condition.eq("targets_removed")].set_index(
        ["moa_class", "family"]).reindex(keys)
    perm = permutations.loc[permutations.condition.eq("targets_removed")].set_index(
        ["moa_class", "family"]).reindex(keys)
    ratios = selected.pure_fold_over_random_candidates.to_numpy(float).reshape(6, 8)
    q_values = perm.q_bh_all_family_cells.to_numpy(float).reshape(6, 8)
    if not np.isfinite(ratios).all() or not (ratios > 0).all():
        raise ValueError("The 48 archived family fold ratios must be finite and positive")
    if not np.isfinite(q_values).all() or int((q_values < 0.05).sum()) != 4:
        raise ValueError("The 48 archived q values must retain four significant cells")
    fig = plt.figure(figsize=(178 / 25.4, 74 / 25.4), dpi=300, facecolor="white")
    drawing.draw_g(fig, ratios, q_values)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output / f"Figure2g.{suffix}", dpi=300)
    plt.close(fig)


def render_fig3f(output: Path) -> None:
    import pandas as pd

    recovery = load_module(ROOT / "figures/manuscript/figure3/recovery.py", "recovery")
    recovery.OUT = output
    recovery.setup_style()
    # These are the final six-panel builder's exact letter and geometry changes.
    code = inspect.getsource(recovery.draw).replace('"d", fontsize=10.5', '"f", fontsize=10.5')
    code = code.replace("21 / HEIGHT_MM", "18 / HEIGHT_MM")
    code = code.replace("42 / HEIGHT_MM", "27 / HEIGHT_MM")
    code = code.replace("69 / HEIGHT_MM", "50.5 / HEIGHT_MM")
    recovery.HEIGHT_MM = 55
    exec(compile(code, str(ROOT / "figures/manuscript/figure3/recovery.py"), "exec"), recovery.__dict__)
    recovery.draw(pd.read_csv(DATA / "figure3/grouped_recovery_summary.tsv", sep="\t"))


def render_fig3(output: Path, assets: Path) -> None:
    source = ROOT / "figures/manuscript/figure3"
    shutil.copytree(DATA / "figure3", output / "source_data")
    (output / "panels").mkdir()
    # Run in a fresh process, using only the selected sibling module closure.
    sys.path.insert(0, str(source))
    os.environ["METAR_PLOT_OUTPUT"] = str(output)
    os.environ["METAR_PLOT_BASELINE"] = str(assets / "Figure3_20261019.pdf")
    runpy.run_path(str(source / "build_figure.py"), run_name="__main__")


def render_main(entry: dict, output: Path, assets: Path | None) -> None:
    number = int(entry["id"][3:])
    source = ROOT / entry["scripts"][0]
    os.environ["METAR_PLOT_OUTPUT"] = str(output)
    if assets is not None:
        os.environ["METAR_PLOT_ASSETS"] = str(assets)
    if number == 2:
        for name in ("baseline", "panel", "full"):
            (output / name).mkdir()
        shutil.copy2(assets / "Figure2_baseline.pdf", output / "baseline/Figure2.pdf")
        for name in ("Figure2a_editable.pdf", "Figure2a_editable.pptx"):
            shutil.copy2(assets / name, output / "panel" / name)
        runpy.run_path(str(source), run_name="__main__")
    elif number == 6:
        name = "Figure6a_review_20261019_editable.pptx"
        shutil.copy2(assets / name, output / name)
        runpy.run_path(str(source), run_name="__main__")
    else:
        module = load_module(source, "metar_current_main_figure")
        module.ROOT = output
        module.WORK = output
        if number == 4:
            module.BASE = assets / "Figure4_baseline.pdf"
            module.PANEL = assets / "Figure4a_editable.pdf"
            module.OUT = output / "Figure4.pdf"
        elif number == 5:
            module.BASE = assets / "Figure5_baseline.pdf"
            module.PPTX = assets / "Figure5a_editable.pptx"
            module.OUT = output / "Figure5.pdf"
        elif number == 7:
            module.DATA = DATA / "figure7"
            module.CONTRACT_PATH = ROOT / "figures/manuscript/figure3/rendering_style.json"
            module.PREVIEWS = output / "previews"
        else:
            raise ValueError(f"Unsupported main-figure source: {number}")
        module.main()


def render_supplement(entry: dict, output: Path, assets: Path | None) -> None:
    renderer, number_text = entry["renderer"].split(":")
    number = int(number_text)
    script = ROOT / entry["scripts"][0]
    module = load_module(script, "metar_selected_supplement")
    module.OUT = output
    module.FIG = output
    module.HERE = output
    module.DATA = DATA / "supplementary/biology"
    module.SOURCE = DATA / "supplementary/legacy"
    if renderer == "legacy":
        recovered = module.SOURCE / "recovered_originals"
        module.FIG3_BOOTSTRAP = recovered / "Figure3_cluster_bootstrap_curve.tsv"
        module.FIG3_BOOTSTRAP_TOP5 = recovered / "Figure3_cluster_bootstrap_top5.tsv"
        module.FIG7_GENE = recovered / "fig2_single_target_gene_scores.tsv"
        module.FIG7_TERM = recovered / "fig2_pathway_continuous_rank.tsv"
        module.style()
        if number == 3:
            # Use the manuscript terminology for the full candidate target space.
            code = inspect.getsource(module.figure3)
            code = code.replace("Full 16,583-gene universe", "Full 16,583-gene target space")
            code = code.replace('Full\\nuniverse', 'Full\\ntarget space')
            exec(compile(code, str(script), "exec"), module.__dict__)
            module.figure3()
        else:
            getattr(module, f"figure{number}")()
    elif renderer == "moa":
        # INPUTS was constructed at import time; relocate each selected basename.
        module.INPUTS = {
            key: module.DATA / path.parent.name / path.name
            for key, path in module.INPUTS.items()
        }
        module.configure_style()
        getattr(module, f"figure{number}")()
    elif number == 4:
        module.style()
        module.figure4()
    else:
        module.configure_style()
        getattr(module, f"render_figure{number}")()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    listing = subparsers.add_parser("list", help="show the selected routes and coverage boundaries")
    listing.add_argument("--json", action="store_true")
    for name in ("check", "render"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("figure", help="catalog id, or 'all' for check")
        subparser.add_argument("--asset-root", type=Path,
                               help="directory containing explicitly locked external artwork/raster assets")
        if name == "render":
            subparser.add_argument("--output", required=True, type=Path)
            subparser.add_argument("--font-dir", type=Path)
    args = parser.parse_args()
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    if args.command == "list":
        if args.json:
            print(json.dumps(catalog, indent=2, ensure_ascii=False))
        else:
            for entry in catalog["render_entries"]:
                assets = ", ".join(item["name"] for item in entry["external_assets"]) or "none"
                status = entry.get("status", "available")
                print(f"{entry['id']}\t{entry['title']}\t{status}\texternal assets: {assets}")
            print("See docs/plotting.md for supported figures and required assets.")
        return 0
    lookup = {entry["id"]: entry for entry in catalog["render_entries"]}
    if args.figure == "all" and args.command == "check":
        selected = list(lookup.values())
    elif args.figure in lookup:
        selected = [lookup[args.figure]]
    else:
        parser.error(f"Unknown render route: {args.figure}; run 'list' for supported ids")
    assets = args.asset_root.resolve() if args.asset_root is not None else None
    checks = [check_entry(entry, catalog, assets) for entry in selected]
    if args.command == "check" or not all(check["ok"] for check in checks):
        print(json.dumps(checks, indent=2, ensure_ascii=False))
        return 0 if all(check["ok"] for check in checks) else 2
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    # Keep generated output outside the selected source/table trees.
    protected_roots = (
        "figures", "data", "scripts", "src", "training", "analysis", "vendor", "manuscript_current",
    )
    for directory in protected_roots:
        protected = ROOT / directory
        if output == protected or protected in output.parents:
            raise ValueError(f"Output must be outside repository source/input trees: {output}")
    entry = selected[0]
    if entry["id"] not in {"fig2", "fig5"}:
        os.environ["METAR_PLOT_ARIAL"] = str(configure_fonts(args.font_dir))
    output.mkdir(parents=True)
    if entry["renderer"] == "fig2g":
        render_fig2g(output)
    elif entry["renderer"] == "fig3f":
        render_fig3f(output)
    elif entry["renderer"] == "fig3":
        render_fig3(output, assets)
    elif entry["renderer"].startswith(("assembly:", "current:")):
        render_main(entry, output, assets)
    else:
        render_supplement(entry, output, assets)
    print(f"PLOT_RENDERED\t{entry['id']}\t{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
