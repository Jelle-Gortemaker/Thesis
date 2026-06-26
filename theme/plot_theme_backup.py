from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from cycler import cycler


# ============================================================
# BASE MATPLOTLIB STYLE
# ============================================================

STYLE = "default"
USE_TEX = False


def apply_theme() -> None:
    """
    Apply shared thesis plot settings.
    White background is enforced here.
    """
    _register_custom_cmaps()

    plt.style.use(STYLE)

    font = {
        "family": FONT_FAMILY,
        "sans-serif": [SANS_SERIF_FONT, "DejaVu Sans", "Arial"],
        "weight": "normal",
        "size": FONT_SIZE,
    }

    rc("font", **font)
    rc("text", usetex=USE_TEX)

    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": SAVE_DPI,
        "savefig.bbox": "tight",

        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",

        "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,

        "lines.linewidth": LINE_WIDTH,
        "axes.prop_cycle": cycler(color=DEFAULT_COLOR_CYCLE),

        "axes.grid": True,
        "grid.alpha": GRID_ALPHA,

        "legend.frameon": True,
        "legend.framealpha": 0.90,

        "image.cmap": CMAPS["default"],
        "mathtext.fontset": "dejavusans",
    })


# ============================================================
# TOC COLOR THEME
# ============================================================

TOC_DARK_BLUE = "#003755"
TOC_CYAN = "#01CBE1"
TOC_LIGHT = "#F2F5F6"
TOC_RED = "#D61418"

TOC_COLORS = [TOC_DARK_BLUE, TOC_CYAN, TOC_LIGHT]

toc_listed_cmap = ListedColormap(TOC_COLORS, name="toc")
toc_cmap = LinearSegmentedColormap.from_list("toc_cmap", TOC_COLORS, N=64)
toc_cmap_r = toc_cmap.reversed()

# Clear categorical cycle based on the TOC theme, with fallback Matplotlib colors.
DEFAULT_COLOR_CYCLE = [
    TOC_DARK_BLUE,
    TOC_CYAN,
    TOC_RED,
    "#6A7D89",
    "#7A3E3E",
    "#2E8B57",
    "#8A6FB0",
    "#E08E45",
]


def _register_custom_cmaps() -> None:
    """
    Register custom colormaps safely during notebook reloads.
    """
    custom_maps = {
        "toc": toc_listed_cmap,
        "toc_cmap": toc_cmap,
        "toc_cmap_r": toc_cmap_r,
    }

    for name, cmap in custom_maps.items():
        try:
            mpl.colormaps.register(cmap, name=name)
        except TypeError:
            # Older Matplotlib versions may not support force=True.
            try:
                mpl.colormaps.register(cmap, name=name)
            except ValueError:
                pass


# ============================================================
# FIGURE SIZES
# ============================================================

DPI = 120
SAVE_DPI = 300

FIGSIZES = {
    "map": (7, 6),
    "wide": (10, 5),
    "panel": (15, 5),
    "small": (6, 4),
    "tall": (8.5, 12.5),
    "square": (6.5, 6.5),
    "profile_panel": (10, 5),
}

# Backward-compatible aliases
FIGSIZE_MAP = FIGSIZES["map"]
FIGSIZE_WIDE = FIGSIZES["wide"]
FIGSIZE_PANEL = FIGSIZES["panel"]
FIGSIZE_SMALL = FIGSIZES["small"]
FIGSIZE_TALL = FIGSIZES["tall"]


# ============================================================
# FONT SIZES
# Keep your existing thesis sizes, not the provided size=24.
# ============================================================

FONT_FAMILY = "sans-serif"
SANS_SERIF_FONT = "Proxima Nova"

FONT_SIZE = 14
TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_SIZE = 11
LEGEND_SIZE = 12
ANNOTATION_SIZE = 10

# ============================================================
# LINE / MARKER SETTINGS
# ============================================================

LINE_WIDTH = 1.2
THIN_LINE_WIDTH = 0.8
THICK_LINE_WIDTH = 2.0

MARKER_SIZE = 4
LARGE_MARKER_SIZE = 16

SCATTER_ALPHA = 0.70
BAND_ALPHA = 0.15
GRID_ALPHA = 0.35
REFERENCE_ALPHA = 0.80

TRAJECTORY_LINE_WIDTH = 0.7
TRAJECTORY_ALPHA = 0.45
TRAJECTORY_SAMPLE_N = 300

POSITION_MARKER_SIZE = 16
POSITION_ALPHA = 0.85
POSITION_EDGEWIDTH = 0.25

RANDOM_SEED = 42


# ============================================================
# CARTOPY / MAP STYLE
# ============================================================

MAP_GRIDLINE_ALPHA = 0.35
MAP_GRIDLINE_LINESTYLE = "--"
MAP_GRIDLINE_LINEWIDTH = 0.6

MAP_LABEL_SIZE = TICK_SIZE

COLORBAR_PAD = 0.075
COLORBAR_FRACTION = 0.046
COLORBAR_ASPECT = 25

QUIVER_COLOR = "0.35"
QUIVER_ALPHA = 0.45
QUIVER_WIDTH = 0.0022

TARGET_BOX_LINEWIDTH = THICK_LINE_WIDTH
TARGET_BOX_COLOR = TOC_RED


# ============================================================
# COLORMAP CATEGORIES
# ============================================================

CMAPS = {
    # Generic categories
    "default": "toc_cmap",
    "default_r": "toc_cmap_r",
    "sequential": "toc_cmap",
    "sequential_r": "toc_cmap_r",
    "categorical": "tab10",

    # Scientific categories
    "diverging": "RdBu_r",
    "rossby": "RdBu_r",
    "vorticity": "RdBu_r",
    "divergence": "RdBu_r",
    "ow": "RdBu_r",

    "density": "turbo",
    "jpdf": "turbo",

    "strain": "toc_cmap",
    "speed": "toc_cmap",
    "temperature": "inferno",
    "salinity": "toc_cmap",
    "voronoi": "toc_cmap",
    "mask": "YlOrRd",

    # Seasonal / grouped plots
    "winter": "Blues",
    "summer": "Reds",
    "spring": "Greens",
    "autumn": "Oranges",
}

CLASS_COLORMAP = "tab10"
MARKER_CYCLE = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "*")


# ============================================================
# NAMED COLORS
# ============================================================

COLORS = {
    "default": TOC_DARK_BLUE,
    "secondary": TOC_CYAN,
    "tertiary": TOC_LIGHT,
    "highlight": TOC_RED,

    "reference": "0.25",
    "zero_line": "0.15",
    "grid": "0.65",
    "box": TOC_RED,

    "spectrum_shell": TOC_DARK_BLUE,
    "spectrum_density": TOC_CYAN,
    "std_band_shell": TOC_DARK_BLUE,
    "std_band_density": TOC_CYAN,

    "passive": TOC_DARK_BLUE,
    "inertial": TOC_RED,
}


# ============================================================
# THEME APPLICATION
# ============================================================

def apply_theme() -> None:
    """
    Apply one existing Matplotlib style plus shared thesis plot settings.
    Call once at the top of every notebook.
    """
    _register_custom_cmaps()

    plt.style.use(STYLE)

    font = {
        "family": FONT_FAMILY,
        "sans-serif": [SANS_SERIF_FONT, "DejaVu Sans", "Arial"],
        "weight": "normal",
        "size": FONT_SIZE,
    }

    rc("font", **font)
    rc("text", usetex=USE_TEX)

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
        "axes.prop_cycle": cycler(color=DEFAULT_COLOR_CYCLE),

        "axes.grid": True,
        "grid.alpha": GRID_ALPHA,

        "legend.frameon": True,
        "legend.framealpha": 0.90,

        "image.cmap": CMAPS["default"],
    })


# ============================================================
# ACCESSORS
# ============================================================

def get_figsize(kind: str = "map") -> tuple[float, float]:
    return FIGSIZES.get(kind, FIGSIZES["map"])


def get_cmap(category: Optional[str] = None):
    """
    Return a Matplotlib colormap from a semantic category.

    Examples
    --------
    get_cmap("default")    -> toc_cmap
    get_cmap("rossby")     -> RdBu_r
    get_cmap("divergence") -> RdBu_r
    get_cmap("jpdf")       -> turbo
    """
    _register_custom_cmaps()

    if category is None:
        category = "default"

    cmap_name = CMAPS.get(category, category)
    return plt.get_cmap(cmap_name)


def get_color(name: str = "default"):
    """
    Return a named color or fallback to any valid Matplotlib color string.
    """
    return COLORS.get(name, name)


def get_class_style(index: int, label: Optional[str] = None) -> dict:
    """
    Return a consistent color/marker combination for particle or category classes.
    """
    cmap = plt.get_cmap(CLASS_COLORMAP)

    return {
        "label": label,
        "color": cmap(index % cmap.N),
        "marker": MARKER_CYCLE[index % len(MARKER_CYCLE)],
    }


# ============================================================
# FIGURE / AXES HELPERS
# ============================================================

def new_figure(kind: str = "map"):
    """
    Create a standard figure and axes.
    """
    fig, ax = plt.subplots(figsize=get_figsize(kind))
    return fig, ax


def format_axes(
    ax,
    *,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    equal: bool = False,
    grid: bool = True,
) -> None:
    """
    Apply common axes formatting.
    """
    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    if equal:
        ax.set_aspect("equal", adjustable="box")

    ax.grid(grid, alpha=GRID_ALPHA)


def add_colorbar(fig, ax, mappable, label: str = ""):
    cbar = fig.colorbar(mappable, ax=ax)
    cbar.set_label(label)
    return cbar


def save_figure(fig, path: str | Path, save: bool = True, close: bool = False) -> None:
    """
    Save figure only when save=True.
    """
    if not save:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.canvas.draw()
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")

    if close:
        plt.close(fig)


def available_styles() -> list[str]:
    return plt.style.available