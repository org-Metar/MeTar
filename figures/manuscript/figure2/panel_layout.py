"""Shared millimetre coordinates and typography for Figure2g."""
from pathlib import Path
from typing import Any
import matplotlib
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text

WIDTH_MM = 178.0
HEIGHT_MM = 74.0
ROW_BOTTOMS = {"g": 1.1}
TEXT = "#263238"

def configure() -> None:
    font_path = font_manager.findfont("Arial", fallback_to_default=False)
    if Path(font_path).name.lower() != "arial.ttf":
        raise ValueError(f"Arial did not resolve to arial.ttf: {font_path}")
    matplotlib.rcParams.update({
        "font.family": "Arial", "font.sans-serif": ["Arial"],
        "font.size": 8.0, "font.weight": "normal", "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": 0.65,
        "axes.labelcolor": TEXT, "text.color": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT, "legend.frameon": False,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def axis_mm(fig: Figure, x: float, y: float, w: float, h: float) -> Axes:
    return fig.add_axes([x / WIDTH_MM, y / HEIGHT_MM, w / WIDTH_MM, h / HEIGHT_MM])


def fig_text(fig: Figure, x: float, y: float, value: str, **kwargs: Any) -> Text:
    return fig.text(x / WIDTH_MM, y / HEIGHT_MM, value,
                    fontfamily="Arial", color=TEXT, **kwargs)


def panel_letter(fig: Figure, value: str, x: float, y: float) -> Text:
    return fig_text(fig, x, y, value, ha="left", va="center",
                    fontsize=10.5, fontweight="bold")
