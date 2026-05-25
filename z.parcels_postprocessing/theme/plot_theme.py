from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt


# ============================================================
# EXISTING MATPLOTLIB STYLE
# ============================================================

STYLE = "bmh"


# ============================================================
# GENERAL PLOT STYLE
# ============================================================

DPI = 120
SAVE_DPI = 300

FIGSIZE_MAP = (7, 6)
FIGSIZE_WIDE = (10, 5)
FIGSIZE_PANEL = (15, 5)
FIGSIZE_SMALL = (6, 4)

FONT_SIZE = 14
TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_SIZE = 11
LEGEND_SIZE = 12

LINE_WIDTH = 1.2
THIN_LINE_WIDTH = 0.8

TRAJECTORY_LINE_WIDTH = 0.7
TRAJECTORY_ALPHA = 0.45

MARKER_SIZE = 4
SCATTER_ALPHA = 0.7

TRAJECTORY_SAMPLE_N = 300
RANDOM_SEED = 42

# Distinct class styling for multi-class particle plots
CLASS_COLORMAP = "tab10"
MARKER_CYCLE = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "*")

POSITION_MARKER_SIZE = 14
POSITION_ALPHA = 0.85
POSITION_EDGEWIDTH = 0.0


def apply_theme() -> None:
    plt.style.use(STYLE)

    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": SAVE_DPI,
        "savefig.bbox": "tight",

        "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,

        "lines.linewidth": LINE_WIDTH,
    })


def save_figure(fig, path: str | Path, save: bool = True) -> None:
    if not save:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")


def available_styles() -> list[str]:
    return plt.style.available