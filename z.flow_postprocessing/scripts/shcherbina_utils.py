from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# SHARED THESIS PLOTTING STYLE
# ============================================================

# The module is normally located in:
#   THESIS/z.flow_postprocessing/scripts/shcherbina_utils.py
# while the shared theme is located in:
#   THESIS/theme/plot_theme.py
# This small block keeps the utility module usable from notebooks in
# different subfolders.
for _parent in Path(__file__).resolve().parents:
    if (_parent / "theme" / "plot_theme.py").exists():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

try:
    import theme.plot_theme as ptheme
except Exception:  # fallback when the shared theme is unavailable
    ptheme = None


def apply_plot_style():
    """Apply the shared thesis plotting style with a white background."""
    if ptheme is not None and hasattr(ptheme, "apply_theme"):
        ptheme.apply_theme()
    else:
        plt.style.use("default")
        plt.rcParams.update({
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 12,
            "axes.grid": True,
            "grid.alpha": 0.35,
        })

    # Enforce this even if the selected style changes later.
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
    })


def get_figsize(kind="map"):
    """Return a shared figure size, with safe fallbacks."""
    if ptheme is not None and hasattr(ptheme, "get_figsize"):
        try:
            return ptheme.get_figsize(kind)
        except Exception:
            pass

    fallback = {
        "map": (7, 5),
        "wide": (10, 5),
        "panel": (15, 5),
        "square": (7, 6),
        "small": (6, 4),
    }
    return fallback.get(kind, fallback["map"])


def get_color(name="default"):
    """Return a shared semantic color, with safe fallbacks."""
    if ptheme is not None and hasattr(ptheme, "get_color"):
        try:
            return ptheme.get_color(name)
        except Exception:
            pass

    fallback = {
        "default": "#003755",
        "secondary": "#01CBE1",
        "highlight": "#D61418",
        "reference": "0.25",
        "grid": "0.65",
    }
    return fallback.get(name, name)


def get_cmap(category=None):
    """Return a shared colormap, accepting both semantic names and Matplotlib names."""
    if category is None:
        category = "default"

    if ptheme is not None and hasattr(ptheme, "get_cmap"):
        try:
            return ptheme.get_cmap(category)
        except Exception:
            pass

    semantic = {
        "rossby": "RdBu_r",
        "divergence": "RdBu_r",
        "strain": "turbo",
        "temperature": "turbo",
        "jpdf": "turbo",
        "conditional": "RdBu_r",
        "categorical": "tab20",
        "default": "viridis",
    }
    return plt.get_cmap(semantic.get(category, category))


def _theme_attr(name, fallback):
    if ptheme is not None and hasattr(ptheme, name):
        return getattr(ptheme, name)
    return fallback


def format_axis(ax, title=None, xlabel=None, ylabel=None, grid=True, equal=False):
    """Consistent formatting for regular Cartesian axes."""
    ax.set_facecolor("white")

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.grid(grid, alpha=_theme_attr("GRID_ALPHA", 0.35))

    if equal:
        ax.set_aspect("equal", adjustable="box")

    return ax


def add_colorbar(fig, ax, mappable, label="", extend="neither"):
    """Add a consistently styled colorbar."""
    cbar = fig.colorbar(
        mappable,
        ax=ax,
        extend=extend,
        pad=_theme_attr("COLORBAR_PAD", 0.035),
        fraction=_theme_attr("COLORBAR_FRACTION", 0.046),
        aspect=_theme_attr("COLORBAR_ASPECT", 25),
    )
    cbar.set_label(label)
    return cbar


# Apply once at import. 
apply_plot_style()


# ============================================================
# BASIC DATA HELPERS
# ============================================================

def find_dim(dims, candidates):
    """
    Find a dimension name from a list of possible candidates.

    Useful for MITgcm MNC output where dimensions can be named
    e.g. T, iter, Z, Zmd000059, Xp1, Yp1, etc.
    """
    dims = list(dims)

    for c in candidates:
        if c in dims:
            return c

    for d in dims:
        if any(d.startswith(prefix) for prefix in candidates if isinstance(prefix, str)):
            return d

    for d in dims:
        dl = d.lower()
        if dl.startswith("zmd") or dl == "z":
            return d

    raise KeyError(f"Could not find any of {candidates} in dims={dims}")


def clean_fill(a, big=1e10):
    """
    Replace very large MITgcm fill values by NaN.
    """
    a = np.asarray(a, dtype=np.float32)
    a[np.abs(a) > big] = np.nan
    return a


def get_time_days(it, day_per_index=0.5):
    """
    Convert model output index to days.
    """
    return float(it) * float(day_per_index)


def format_threshold(val):
    """
    Clean threshold formatting for plot titles.
    """
    return f"{val:g}"


# ============================================================
# KINEMATIC DIAGNOSTICS
# ============================================================

def uv_to_center(u, v):
    """
    Convert staggered MITgcm U/V to common tracer-cell centers.

    Parameters
    ----------
    u : ndarray
        Shape (..., y, x_u)
    v : ndarray
        Shape (..., y_v, x)

    Returns
    -------
    u_c, v_c : ndarray
        Both cropped to common horizontal shape (..., y, x).
    """
    u = np.asarray(u)
    v = np.asarray(v)

    u_c = 0.5 * (u[..., :, :-1] + u[..., :, 1:])
    v_c = 0.5 * (v[..., :-1, :] + v[..., 1:, :])

    ny = min(u_c.shape[-2], v_c.shape[-2])
    nx = min(u_c.shape[-1], v_c.shape[-1])

    return u_c[..., :ny, :nx], v_c[..., :ny, :nx]


def gradients_uv(u_c, v_c, dx, dy):
    """
    Compute speed, relative vorticity, horizontal divergence,
    and strain magnitude from centered velocity components.

    Definitions:
        zeta   = dv/dx - du/dy
        div    = du/dx + dv/dy
        strain = sqrt((du/dx - dv/dy)^2 + (dv/dx + du/dy)^2)
    """
    u_c = np.asarray(u_c, dtype=np.float64)
    v_c = np.asarray(v_c, dtype=np.float64)

    du_dy, du_dx = np.gradient(u_c, dy, dx, axis=(-2, -1))
    dv_dy, dv_dx = np.gradient(v_c, dy, dx, axis=(-2, -1))

    zeta = dv_dx - du_dy
    div = du_dx + dv_dy

    sn = du_dx - dv_dy
    ss = dv_dx + du_dy
    strain = np.sqrt(sn**2 + ss**2)

    speed = np.hypot(u_c, v_c)

    return speed, zeta, div, strain


def compute_surface_kinematics(ds, time_dim, z_dim, it, levels, dx, dy, f0):
    """
    Compute Ro, div/f, and strain/f from UVEL/VVEL.

    If multiple vertical levels are provided, diagnostics are computed
    per layer first and then averaged. This avoids vertically averaging
    velocities before taking horizontal derivatives.
    """
    layer_results = []

    for k in levels:
        u = clean_fill(ds["UVEL"].isel({time_dim: it, z_dim: k}).values)
        v = clean_fill(ds["VVEL"].isel({time_dim: it, z_dim: k}).values)

        u_c, v_c = uv_to_center(u[None, ...], v[None, ...])
        u_c = u_c[0]
        v_c = v_c[0]

        _, zeta_k, div_k, strain_k = gradients_uv(u_c, v_c, dx, dy)

        layer_results.append({
            "ro": zeta_k / f0,
            "div_f": div_k / f0,
            "strain_f": strain_k / f0,
        })

    ro = np.nanmean(np.stack([d["ro"] for d in layer_results], axis=0), axis=0)
    div_f = np.nanmean(np.stack([d["div_f"] for d in layer_results], axis=0), axis=0)
    strain_f = np.nanmean(np.stack([d["strain_f"] for d in layer_results], axis=0), axis=0)

    return ro, div_f, strain_f


# ============================================================
# STATISTICS
# ============================================================

def normalize_hist(y):
    """
    Normalize histogram values by their maximum.
    """
    y = np.asarray(y, dtype=float)
    m = np.nanmax(y)
    if np.isfinite(m) and m > 0:
        return y / m
    return y


def nan_skew(a):
    """
    NaN-safe Pearson moment skewness.
    """
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]

    if a.size == 0:
        return np.nan

    mu = np.mean(a)
    sig = np.std(a)

    if sig == 0 or not np.isfinite(sig):
        return np.nan

    return np.mean((a - mu) ** 3) / sig ** 3


def stats_dict(a):
    """
    Summary statistics for flattened finite values.
    """
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]

    if a.size == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "skew": np.nan,
        }

    return {
        "mean": np.mean(a),
        "median": np.median(a),
        "std": np.std(a),
        "skew": nan_skew(a),
    }


# ============================================================
# DOMAIN SUBDIVISION / LOCAL PRo
# ============================================================

def build_square_index_ranges(ny, nx, nsy, nsx):
    """
    Divide a 2D domain into nsy x nsx rectangular/square-ish regions.

    Numbering starts bottom-left and proceeds row-wise:
        1, 2, 3, ...
    """
    y_edges = np.linspace(0, ny, nsy + 1, dtype=int)
    x_edges = np.linspace(0, nx, nsx + 1, dtype=int)

    squares = []
    sq_id = 1

    for jy in range(nsy):
        for ix in range(nsx):
            y0, y1 = y_edges[jy], y_edges[jy + 1]
            x0, x1 = x_edges[ix], x_edges[ix + 1]

            squares.append({
                "square_id": sq_id,
                "ix": ix,
                "jy": jy,
                "x0": x0,
                "x1": x1,
                "y0": y0,
                "y1": y1,
            })

            sq_id += 1

    return squares, x_edges, y_edges


def compute_pro_per_square(ro, squares, threshold):
    """
    Compute PRo = area fraction where |Ro| > threshold for each square.
    """
    vals = []

    for sq in squares:
        sub = ro[sq["y0"]:sq["y1"], sq["x0"]:sq["x1"]]
        finite = np.isfinite(sub)

        if np.any(finite):
            pro = np.mean(np.abs(sub[finite]) > threshold)
        else:
            pro = np.nan

        vals.append({
            "square_id": sq["square_id"],
            "PRo": float(pro),
        })

    return vals


# ============================================================
# PLOTTING HELPERS
# ============================================================

def savefig_if_needed(fig, filename_base, save=False, out_dir=None, dpi=None):
    """
    Save figure if save=True, using the shared white-background style.
    """
    if not save:
        return None

    if out_dir is None:
        raise ValueError("out_dir must be provided when save=True.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if dpi is None:
        dpi = _theme_attr("SAVE_DPI", 300)

    png_path = out_dir / f"{filename_base}.png"

    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        ax.set_facecolor("white")

    fig.canvas.draw()
    fig.savefig(
        png_path,
        bbox_inches="tight",
        dpi=dpi,
        facecolor="white",
        transparent=False,
    )
    print(f"Saved: {png_path}")

    return png_path


def plot_map(
    field,
    title="",
    cbar_label="",
    cmap=None,
    vmin=None,
    vmax=None,
    dx=500.0,
    dy=500.0,
    figsize=None,
    xlabel="x [km]",
    ylabel="y [km]",
    grid=True,
    extend="neither",
):
    """
    Consistent map plot on a uniform Cartesian grid.

    Returns
    -------
    fig, ax
        Returning these explicitly avoids blank saved figures.
    """
    apply_plot_style()

    if figsize is None:
        figsize = get_figsize("map")

    if cmap is None:
        cmap = get_cmap("default")
    elif isinstance(cmap, str):
        cmap = get_cmap(cmap)

    field = np.asarray(field)
    ny, nx = field.shape[-2], field.shape[-1]

    x_edges_km = np.arange(nx + 1) * dx / 1000.0
    y_edges_km = np.arange(ny + 1) * dy / 1000.0

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    pcm = ax.pcolormesh(
        x_edges_km,
        y_edges_km,
        field,
        shading="flat",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    add_colorbar(fig, ax, pcm, label=cbar_label, extend=extend)

    ax.set_xlim(x_edges_km[0], x_edges_km[-1])
    ax.set_ylim(y_edges_km[0], y_edges_km[-1])

    format_axis(
        ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        grid=grid,
        equal=True,
    )

    fig.tight_layout()

    return fig, ax


def plot_pdf_panel(
    ax,
    data,
    bins,
    title,
    xlabel,
    color_line=None,
    show_zero_line=True,
):
    """
    Histogram bins + normalized PDF line + summary stats,
    using the shared thesis style.
    """
    apply_plot_style()

    if color_line is None:
        color_line = get_color("highlight")

    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]

    hist, edges = np.histogram(data, bins=bins, density=True)
    hist_n = normalize_hist(hist)

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    ax.bar(
        centers,
        hist_n,
        width=widths,
        align="center",
        facecolor="0.88",
        edgecolor="0.35",
        linewidth=_theme_attr("THIN_LINE_WIDTH", 0.8),
    )

    ax.plot(
        centers,
        hist_n,
        color=color_line,
        lw=_theme_attr("THICK_LINE_WIDTH", 2.0),
    )

    if show_zero_line:
        ax.axvline(
            0.0,
            color=get_color("reference"),
            ls="--",
            lw=_theme_attr("LINE_WIDTH", 1.2),
            alpha=_theme_attr("REFERENCE_ALPHA", 0.8),
        )

    s = stats_dict(data)

    txt = (
        f"Mean    {s['mean']:6.2f}\n"
        f"Median  {s['median']:6.2f}\n"
        f"St. dev.{s['std']:6.2f}\n"
        f"Skew.   {s['skew']:6.2f}"
    )

    ax.text(
        0.98,
        0.98,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=_theme_attr("ANNOTATION_SIZE", 10),
        bbox=dict(facecolor="white", edgecolor="0.80", alpha=0.85, pad=3),
    )

    format_axis(ax, title=title, xlabel=xlabel, ylabel="PDF", grid=False)
    ax.set_ylim(bottom=0)

    return ax


def add_regime_labels(ax, x_edges, y_edges):
    """
    Add AVD / SD / CVD labels in the vorticity-strain plane.
    """
    x0, x1 = x_edges[0], x_edges[-1]
    y0, y1 = y_edges[0], y_edges[-1]

    xrng = x1 - x0
    yrng = y1 - y0

    label_kwargs = dict(
        fontsize=_theme_attr("LABEL_SIZE", 14),
        color=get_color("reference"),
        fontweight="bold",
    )

    ax.text(x0 + 0.10 * xrng, y0 + 0.10 * yrng, "AVD", **label_kwargs)
    ax.text(x0 + 0.38 * xrng, y0 + 0.72 * yrng, "SD", **label_kwargs)
    ax.text(x0 + 0.84 * xrng, y0 + 0.10 * yrng, "CVD", ha="right", **label_kwargs)


def plot_square_overlay(
    ro,
    squares,
    dx,
    dy,
    title="",
    cmap="RdBu_r",
    vmin=-2,
    vmax=2,
    figsize=None,
):
    """
    Plot Rossby number with discretized square overlay and square IDs.
    """
    apply_plot_style()

    if figsize is None:
        figsize = get_figsize("map")

    if isinstance(cmap, str):
        cmap = get_cmap(cmap)

    ro = np.asarray(ro)
    ny, nx = ro.shape

    x_edges_km = np.arange(nx + 1) * dx / 1000.0
    y_edges_km = np.arange(ny + 1) * dy / 1000.0

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    pcm = ax.pcolormesh(
        x_edges_km,
        y_edges_km,
        ro,
        shading="flat",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    add_colorbar(fig, ax, pcm, label=r"$\zeta/f$", extend="both")

    overlay_color = get_color("reference")

    for sq in squares:
        x0_km = x_edges_km[sq["x0"]]
        x1_km = x_edges_km[sq["x1"]]
        y0_km = y_edges_km[sq["y0"]]
        y1_km = y_edges_km[sq["y1"]]

        rect = Rectangle(
            (x0_km, y0_km),
            x1_km - x0_km,
            y1_km - y0_km,
            fill=False,
            edgecolor=overlay_color,
            linewidth=_theme_attr("LINE_WIDTH", 1.2),
        )
        ax.add_patch(rect)

        xc = 0.5 * (x0_km + x1_km)
        yc = 0.5 * (y0_km + y1_km)

        ax.text(
            xc,
            yc,
            str(sq["square_id"]),
            ha="center",
            va="center",
            fontsize=_theme_attr("ANNOTATION_SIZE", 10),
            color=overlay_color,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0),
        )

    ax.set_xlim(x_edges_km[0], x_edges_km[-1])
    ax.set_ylim(y_edges_km[0], y_edges_km[-1])

    format_axis(
        ax,
        title=title,
        xlabel="x [km]",
        ylabel="y [km]",
        grid=True,
        equal=True,
    )

    fig.tight_layout()

    return fig, ax


def plot_jpdf(
    x_edges,
    y_edges,
    H_log,
    title="",
    xlabel="",
    ylabel="",
    cbar_label=r"$\log_{10}(P/P_{\max})$",
    cmap="jpdf",
    vmin=-2,
    vmax=0,
    figsize=None,
    diagonal_abs=False,
    zero_lines=False,
    regime_labels=False,
):
    """
    Consistent JPDF plot for log-normalized 2D histograms.
    """
    apply_plot_style()

    if figsize is None:
        figsize = get_figsize("square")

    if isinstance(cmap, str):
        cmap = get_cmap(cmap)

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    pcm = ax.pcolormesh(
        x_edges,
        y_edges,
        H_log.T,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    add_colorbar(fig, ax, pcm, label=cbar_label)

    if diagonal_abs:
        xx = np.linspace(x_edges[0], x_edges[-1], 500)
        ax.plot(
            xx,
            np.abs(xx),
            "--",
            color=get_color("reference"),
            lw=_theme_attr("LINE_WIDTH", 1.2),
            alpha=_theme_attr("REFERENCE_ALPHA", 0.8),
        )

    if zero_lines:
        ax.axvline(0.0, color=get_color("reference"), ls="--", lw=_theme_attr("THIN_LINE_WIDTH", 0.8))
        ax.axhline(0.0, color=get_color("reference"), ls="--", lw=_theme_attr("THIN_LINE_WIDTH", 0.8))

    if regime_labels:
        add_regime_labels(ax, x_edges, y_edges)

    format_axis(ax, title=title, xlabel=xlabel, ylabel=ylabel, grid=True)

    fig.tight_layout()

    return fig, ax


def plot_conditional_mean(
    x_edges,
    y_edges,
    mean_field,
    title="",
    xlabel="",
    ylabel="",
    cbar_label="",
    cmap="conditional",
    vlim=None,
    percentile_limit=98,
    figsize=None,
    diagonal_abs=True,
    regime_labels=True,
):
    """
    Consistent conditional-mean plot, using symmetric color limits by default.
    """
    apply_plot_style()

    if figsize is None:
        figsize = get_figsize("square")

    if isinstance(cmap, str):
        cmap = get_cmap(cmap)

    field = np.asarray(mean_field, dtype=float)

    if vlim is None:
        valid = np.isfinite(field)
        if np.any(valid):
            vlim = np.nanpercentile(np.abs(field[valid]), percentile_limit)
        else:
            vlim = 1.0

    if not np.isfinite(vlim) or vlim == 0:
        vlim = 1.0

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    pcm = ax.pcolormesh(
        x_edges,
        y_edges,
        field.T,
        shading="auto",
        cmap=cmap,
        vmin=-float(vlim),
        vmax=float(vlim),
    )

    add_colorbar(fig, ax, pcm, label=cbar_label, extend="both")

    if diagonal_abs:
        xx = np.linspace(x_edges[0], x_edges[-1], 500)
        ax.plot(
            xx,
            np.abs(xx),
            "--",
            color=get_color("reference"),
            lw=_theme_attr("LINE_WIDTH", 1.2),
            alpha=_theme_attr("REFERENCE_ALPHA", 0.8),
        )

    if regime_labels:
        add_regime_labels(ax, x_edges, y_edges)

    format_axis(ax, title=title, xlabel=xlabel, ylabel=ylabel, grid=True)

    fig.tight_layout()

    return fig, ax, float(vlim)


def plot_pro_timeseries(
    square_pro_wide,
    plot_time_days=None,
    threshold=0.5,
    title="",
    figsize=None,
):
    """
    Consistent local PRo time-series plot with square IDs annotated at the end
    """
    apply_plot_style()

    if figsize is None:
        figsize = get_figsize("wide")

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    cmap = get_cmap("categorical")

    xmin = np.nanmin(square_pro_wide.index.values)
    
    if plot_time_days is not None:
        xmax = plot_time_days
        line_cutoff = plot_time_days - 0.8
        text_offset = plot_time_days - 0.72 
    else:
        xmax = np.nanmax(square_pro_wide.index.values) + 0.8
        line_cutoff = np.nanmax(square_pro_wide.index.values)
        text_offset = line_cutoff + 0.08

    for i, sq_id in enumerate(square_pro_wide.columns):
        color = cmap((i % 20) / 19) if callable(cmap) else None
        
        mask = square_pro_wide.index.values <= line_cutoff
        y = square_pro_wide[sq_id].values[mask]
        x = square_pro_wide.index.values[mask]

        ax.plot(
            x,
            y,
            lw=_theme_attr("LINE_WIDTH", 1.2),
            alpha=0.9,
            color=color,
        )

        finite = np.isfinite(y)
        if np.any(finite):
            ax.text(
                text_offset,
                y[-1],  
                str(int(sq_id)),
                color=color,
                fontsize=_theme_attr("ANNOTATION_SIZE", 10),
                va="center",
                ha="left",
            )
        
    ax.set_xlim(xmin, xmax)

    if title == "":
        title = f"Local PRo > {format_threshold(threshold)}"

    format_axis(ax, title=title, xlabel="time [days]", ylabel="PRo", grid=True)

    ax.set_yscale("symlog", linthresh=1.0e-4, linscale=1.0, base=10)
    ax.set_ylim(0.0, 0.5)
    ax.set_yticks([0.0, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 5.0e-1,])
    ax.set_yticklabels([
        "0",
        r"$10^{-4}$",
        r"$10^{-3}$",
        r"$10^{-2}$",
        r"$10^{-1}$",
        r"$5\times10^{-1}$",
    ])
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.15)


    fig.tight_layout()

    return fig, ax


def crop_to_common_shape(*arrays):
    """
    Crop all arrays to the smallest common horizontal (..., y, x) shape.
    Useful when WVEL/THETA and velocity-gradient fields differ by one
    point because of MITgcm staggering.
    """
    if len(arrays) == 0:
        return []

    ny = min(np.asarray(a).shape[-2] for a in arrays)
    nx = min(np.asarray(a).shape[-1] for a in arrays)

    return [np.asarray(a)[..., :ny, :nx] for a in arrays]


def conditioned_mean_2d(x, y, z, x_edges, y_edges):
    """
    Compute conditional mean E[z | x-bin, y-bin].

    This is used for Balwada-style conditional means, e.g.
    mean vertical velocity conditioned on surface vorticity and strain.

    Returns
    -------
    mean_z : ndarray
        Shape (len(x_edges)-1, len(y_edges)-1), consistent with:
        pcolormesh(x_edges, y_edges, mean_z.T)
    count : ndarray
        Number of samples per bin.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

    x = x[mask]
    y = y[mask]
    z = z[mask]

    sum_z, _, _ = np.histogram2d(
        x,
        y,
        bins=[x_edges, y_edges],
        weights=z
    )

    count, _, _ = np.histogram2d(
        x,
        y,
        bins=[x_edges, y_edges]
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_z = sum_z / count

    mean_z[count == 0] = np.nan

    return mean_z, count



# ============================================================
# REYNOLDS NUMBER DIAGNOSTICS
# ============================================================

def velocity_anomaly(u, v, method="domain_mean", filter_size=None):
    """
    Compute velocity anomalies.

    Parameters
    ----------
    method : str
        "domain_mean":
            u' = u - mean(u)
            v' = v - mean(v)

        "box_filter":
            subtract a simple running mean using scipy.ndimage.uniform_filter.
    filter_size : int or None
        Filter size in grid cells for method="box_filter".

    Returns
    -------
    u_prime, v_prime : 2D arrays
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    if method == "domain_mean":
        return u - np.nanmean(u), v - np.nanmean(v)

    if method == "box_filter":
        if filter_size is None:
            raise ValueError("filter_size must be provided for method='box_filter'.")

        try:
            from scipy.ndimage import uniform_filter
        except ImportError as exc:
            raise ImportError("scipy is required for method='box_filter'.") from exc

        u_fill = np.where(np.isfinite(u), u, np.nanmean(u))
        v_fill = np.where(np.isfinite(v), v, np.nanmean(v))

        u_smooth = uniform_filter(u_fill, size=filter_size, mode="nearest")
        v_smooth = uniform_filter(v_fill, size=filter_size, mode="nearest")

        return u - u_smooth, v - v_smooth

    raise ValueError("method must be 'domain_mean' or 'box_filter'.")


def rms_velocity(u, v):
    """
    RMS horizontal velocity magnitude.

    U_rms = sqrt(<u^2 + v^2>)
    """
    u, v = crop_to_common_shape(u, v)
    mask = np.isfinite(u) & np.isfinite(v)

    if not np.any(mask):
        return np.nan

    return float(np.sqrt(np.nanmean(u[mask] ** 2 + v[mask] ** 2)))
