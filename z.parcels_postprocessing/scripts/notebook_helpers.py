from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

import theme.plot_theme as ptheme


# ============================================================
# SMALL GENERAL HELPERS
# ============================================================

def apply_notebook_style() -> None:
    """Apply the shared plotting style used by the notebook."""
    ptheme.apply_theme()


def get_figsize(kind: str = "map"):
    """Return a themed figure size, with a fallback for older plot_theme files."""
    if hasattr(ptheme, "get_figsize"):
        return ptheme.get_figsize(kind)
    if kind == "wide":
        return getattr(ptheme, "FIGSIZE_WIDE", (10, 5))
    return getattr(ptheme, "FIGSIZE_MAP", (7, 6))


def savefig_if_enabled(fig, path: str | Path, save: bool = True):
    """Save a figure only when save=True."""
    if not save:
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        ptheme.save_figure(fig, path, save=True)
    except TypeError:
        ptheme.save_figure(fig, path)

    return path


def to_km(a):
    return np.asarray(a, dtype=float) / 1000.0


def resolve_obs_index(selection, n_obs: int) -> int:
    """Resolve snapshot selection strings such as 'first' and 'last'."""
    if isinstance(selection, str):
        selection = selection.lower()

        if selection == "first":
            return 0
        if selection == "last":
            return n_obs - 1

        raise ValueError(f"Unknown obs selection string: {selection}")

    selection = int(selection)

    if selection < 0:
        return n_obs + selection

    return selection

def load_ett_flow_timescale(
    ett_json: str | Path,
    *,
    multiplier: float = 0.5,
) -> dict[str, Any]:
    """
    Read an ETT metadata JSON and return the selected flow timescale.

    The expected JSON entry is:
        eddy_turnover_time_days -> median_T_eddy_days

    The selected flow timescale is:
        multiplier * median_T_eddy_days
    """
    ett_json = Path(ett_json)

    if not ett_json.exists():
        raise FileNotFoundError(f"Could not find ETT metadata file:\n{ett_json}")

    if multiplier <= 0.0:
        raise ValueError("multiplier must be positive.")

    with open(ett_json, "r") as f:
        metadata = json.load(f)

    try:
        median_days = float(
            metadata["eddy_turnover_time_days"]["median_T_eddy_days"]
        )
    except KeyError as exc:
        raise KeyError(
            "Expected ETT JSON entry: "
            "eddy_turnover_time_days -> median_T_eddy_days"
        ) from exc

    flow_days = float(multiplier) * median_days

    return {
        "ett_json": ett_json,
        "metadata": metadata,
        "median_eddy_turnover_days": median_days,
        "multiplier": float(multiplier),
        "flow_timescale_days": flow_days,
        "flow_timescale_seconds": flow_days * 86400.0,
    }


# ============================================================
# PARTICLE / TRAJECTORY DATA HELPERS
# ============================================================


def _tau_value_and_unit(tau_seconds: float) -> tuple[float, str]:
    """Return a compact numerical value and unit for tau_p."""
    tau_seconds = float(tau_seconds)

    if tau_seconds <= 0.0:
        return 0.0, "s"

    days = tau_seconds / 86400.0
    if np.isclose(days, round(days)):
        return float(round(days)), "d"

    hours = tau_seconds / 3600.0
    if np.isclose(hours, round(hours)):
        return float(round(hours)), "h"

    minutes = tau_seconds / 60.0
    if np.isclose(minutes, round(minutes)):
        return float(round(minutes)), "min"

    return tau_seconds, "s"


def _tau_tag(tau_seconds: float) -> str:
    """
    Compact file-safe tag containing only tau_p.

    Examples:
        600 s   -> tau10min
        3600 s  -> tau1h
        43200 s -> tau12h
    """
    value, unit = _tau_value_and_unit(tau_seconds)
    value_txt = f"{value:g}".replace("-", "m").replace(".", "p")
    return f"tau{value_txt}{unit}"


def _tau_display_text(tau_seconds: float) -> str:
    value, unit = _tau_value_and_unit(tau_seconds)
    return f"{value:g} {unit}"

def build_particle_specs(
    tau_p_seconds_list: Iterable[float] | None = None,
    *,
    include_passive: bool = True,
    flow_timescale_seconds: float | None = None,
) -> list[dict[str, Any]]:
    """
    Build particle specifications using prescribed effective tau_p values.

    Particle tags contain only tau_p. Display labels contain tau_p and the
    automatically calculated effective Stokes number:

        St = tau_p / flow_timescale_seconds
    """
    specs: list[dict[str, Any]] = []

    if flow_timescale_seconds is not None:
        flow_timescale_seconds = float(flow_timescale_seconds)

        if flow_timescale_seconds <= 0.0:
            raise ValueError("flow_timescale_seconds must be positive.")

    if include_passive:
        specs.append(
            {
                "particle_class": "passive",
                "tag": "passive",
                "label": "Passive particles",
                "tau_p_seconds": 0.0,
                "stokes_number": 0.0,
                "flow_timescale_seconds": flow_timescale_seconds,
            }
        )

    if tau_p_seconds_list is None:
        tau_p_seconds_list = []

    for tau_p_seconds in tau_p_seconds_list:
        tau_p_seconds = float(tau_p_seconds)

        if tau_p_seconds <= 0.0:
            raise ValueError(
                "Every MR-SM tau_p_seconds value must be positive."
            )

        if flow_timescale_seconds is None:
            stokes_number = np.nan
        else:
            stokes_number = tau_p_seconds / flow_timescale_seconds

        tau_txt = _tau_display_text(tau_p_seconds)
        tag = _tau_tag(tau_p_seconds)

        if np.isfinite(stokes_number):
            label = f"τₚ={tau_txt}, St={stokes_number:.3g}"
        else:
            label = f"τₚ={tau_txt}"

        specs.append(
            {
                "particle_class": "mr_sm",
                "tag": tag,
                "label": label,
                "tau_p_seconds": tau_p_seconds,
                "stokes_number": stokes_number,
                "flow_timescale_seconds": flow_timescale_seconds,
            }
        )

    return specs


def print_particle_specs(particle_specs: list[dict[str, Any]]) -> None:
    print("\nParticle classes")

    for spec in particle_specs:
        print(
            f"  {spec['tag']} | "
            f"{spec['label']} | "
            f"tau_p={spec['tau_p_seconds']:.6g} s"
        )

def get_xy_names(ds: xr.Dataset) -> tuple[str, str]:
    xname = "lon" if "lon" in ds.variables else "x"
    yname = "lat" if "lat" in ds.variables else "y"
    return xname, yname


def get_static_particle_var(ds: xr.Dataset, varname: str, obs_index: int = 0):
    if varname not in ds:
        return None

    da = ds[varname]

    if "obs" in da.dims:
        return da.isel(obs=obs_index).values

    return da.values


def get_release_ids(ds: xr.Dataset) -> np.ndarray:
    release_ids = get_static_particle_var(ds, "release_id", obs_index=0)

    if release_ids is None:
        release_ids = np.arange(ds.sizes["trajectory"], dtype=int)

    return np.asarray(release_ids, dtype=int)


def get_common_release_ids(trajectories: dict) -> np.ndarray:
    """Return release IDs shared by all selected particle classes."""
    if not trajectories:
        return np.array([], dtype=int)

    common_ids: set[int] | None = None

    for item in trajectories.values():
        release_ids = get_release_ids(item["ds"])
        release_id_set = set(release_ids.tolist())

        if common_ids is None:
            common_ids = release_id_set
        else:
            common_ids &= release_id_set

    return np.asarray(sorted(common_ids or []), dtype=int)


def get_periodic_domain_lengths(ds_parcels: xr.Dataset) -> tuple[float, float]:
    xg = np.asarray(ds_parcels["x"].values, dtype=float)
    yg = np.asarray(ds_parcels["y"].values, dtype=float)

    dxg = float(np.nanmedian(np.diff(xg)))
    dyg = float(np.nanmedian(np.diff(yg)))

    x_edge_min = float(xg[0] - 0.5 * dxg)
    x_edge_max = float(xg[-1] + 0.5 * dxg)
    y_edge_min = float(yg[0] - 0.5 * dyg)
    y_edge_max = float(yg[-1] + 0.5 * dyg)

    return x_edge_max - x_edge_min, y_edge_max - y_edge_min


def insert_nan_at_periodic_jumps(
    x,
    y,
    Lx: float,
    Ly: float,
    jump_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Break plotted trajectories when a periodic wrap would otherwise create a long line.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    xb = []
    yb = []

    for i in range(len(x)):
        if i > 0:
            jump_x = (
                np.isfinite(x[i])
                and np.isfinite(x[i - 1])
                and abs(x[i] - x[i - 1]) > jump_fraction * Lx
            )
            jump_y = (
                np.isfinite(y[i])
                and np.isfinite(y[i - 1])
                and abs(y[i] - y[i - 1]) > jump_fraction * Ly
            )

            if jump_x or jump_y:
                xb.append(np.nan)
                yb.append(np.nan)

        xb.append(x[i])
        yb.append(y[i])

    return np.asarray(xb), np.asarray(yb)


def get_elapsed_time_days(ds: xr.Dataset, obs_index: int) -> float:
    if "time" not in ds:
        return np.nan

    t = ds["time"].isel(trajectory=0, obs=obs_index).values
    t0 = ds["time"].isel(trajectory=0, obs=0).values

    if np.issubdtype(np.asarray(t).dtype, np.datetime64):
        return float((t - t0) / np.timedelta64(1, "s")) / 86400.0

    return float(t - t0) / 86400.0


def build_class_style_map(trajectories: dict) -> dict:
    """Build a consistent color/marker map for each particle class."""
    particle_tags = list(trajectories.keys())

    if hasattr(ptheme, "get_class_style"):
        return {
            particle_tag: {
                **ptheme.get_class_style(
                    i,
                    label=trajectories[particle_tag].get(
                        "display_label",
                        trajectories[particle_tag].get("label", particle_tag),
                    ),
                ),
                "label": trajectories[particle_tag].get(
                    "display_label",
                    trajectories[particle_tag].get("label", particle_tag),
                ),
            }
            for i, particle_tag in enumerate(particle_tags)
        }

    cmap = plt.get_cmap(getattr(ptheme, "CLASS_COLORMAP", "tab10"))
    marker_cycle = getattr(
        ptheme,
        "MARKER_CYCLE",
        ("o", "s", "^", "D", "P", "X", "v", "<", ">", "*"),
    )

    style_map = {}

    for i, particle_tag in enumerate(particle_tags):
        item = trajectories[particle_tag]
        style_map[particle_tag] = {
            "label": item.get(
                "display_label",
                item.get("label", particle_tag),
            ),
            "color": cmap(i % cmap.N),
            "marker": marker_cycle[i % len(marker_cycle)],
        }

    return style_map


def load_trajectory_collection(
    run_outputs: dict,
    load_if_temporary: bool,
    loader,
) -> dict:
    """Load trajectories from a run_outputs dictionary."""
    trajectories = {}

    for particle_tag, item in run_outputs.items():
        ds_traj = loader(item["path"])

        if load_if_temporary:
            ds_traj = ds_traj.load()

        info = item.get("info", {}) or {}
        spec = item.get("spec", {}) or {}
        label = (
            item.get("display_label")
            or item.get("label")
            or info.get("particle_label")
            or spec.get("label")
            or particle_tag
        )

        trajectories[particle_tag] = {
            "ds": ds_traj,
            "label": label,
            "display_label": label,
            "path": Path(item["path"]),
            "info": info,
            "spec": spec,
        }

        print(f"\nTrajectory QC: {particle_tag}")
        print(f"  label     : {label}")
        print(f"  path      : {item['path']}")
        print(f"  dims      : {dict(ds_traj.sizes)}")
        print(f"  variables : {list(ds_traj.data_vars)}")

    return trajectories


def load_collection_config(config_json: str | Path) -> dict:
    """Load the run-collection JSON written by the run notebook."""
    config_json = Path(config_json)
    with open(config_json, "r") as f:
        return json.load(f)


def load_trajectory_collection_from_config(
    config_json: str | Path,
    *,
    loader,
    load: bool = False,
) -> tuple[dict, dict]:
    """Load saved trajectories and their collection metadata."""
    config = load_collection_config(config_json)
    run_outputs = config.get("run_outputs", {})

    trajectories = load_trajectory_collection(
        run_outputs,
        load_if_temporary=load,
        loader=loader,
    )

    return trajectories, config


def filter_trajectories(
    trajectories: dict,
    *,
    include_tags: Iterable[str] | None = None,
    include_particle_classes: Iterable[str] | None = None,
    stokes_range: tuple[float | None, float | None] | None = None,
    tau_p_range_seconds: tuple[float | None, float | None] | None = None,
) -> dict:
    """
    Select trajectory classes using tau-only particle metadata.

    Parameters
    ----------
    include_tags
        Exact particle tags to retain.
    include_particle_classes
        Particle classes to retain, e.g. ``passive`` and ``mr_sm``.
    stokes_range
        Inclusive range for St = tau_p / flow_timescale.
    tau_p_range_seconds
        Inclusive range for prescribed tau_p in seconds.
    """
    include_tags = None if include_tags is None else set(include_tags)
    include_particle_classes = (
        None
        if include_particle_classes is None
        else set(include_particle_classes)
    )

    selected = {}

    for particle_tag, item in trajectories.items():
        info = item.get("info", {}) or {}
        spec = item.get("spec", {}) or {}

        particle_class = info.get(
            "particle_class",
            spec.get("particle_class"),
        )
        tau_p = info.get(
            "tau_p_seconds",
            spec.get("tau_p_seconds"),
        )
        stokes_number = _effective_stokes_number(info, spec)

        if include_tags is not None and particle_tag not in include_tags:
            continue

        if (
            include_particle_classes is not None
            and particle_class not in include_particle_classes
        ):
            continue

        if stokes_range is not None and np.isfinite(stokes_number):
            st_min, st_max = stokes_range
            if st_min is not None and stokes_number < st_min:
                continue
            if st_max is not None and stokes_number > st_max:
                continue

        if tau_p_range_seconds is not None and tau_p is not None:
            tau_p = float(tau_p)
            tau_min, tau_max = tau_p_range_seconds
            if tau_min is not None and tau_p < tau_min:
                continue
            if tau_max is not None and tau_p > tau_max:
                continue

        selected[particle_tag] = item

    print("Selected particle classes:")
    for particle_tag, item in selected.items():
        print(
            f"  {particle_tag} | "
            f"{item.get('display_label', item.get('label', particle_tag))}"
        )

    return selected


# ============================================================
# OUTPUT ORGANIZATION HELPERS
# ============================================================

@dataclass(frozen=True)
class OutputPaths:
    case_results_dir: Path
    traj_dir: Path
    tmp_traj_dir: Path
    metadata_dir: Path
    metrics_dir: Path
    fig_dir: Path
    run_out_dir: Path
    config_json: Path


def prepare_output_paths(
    *,
    notebook_dir: Path,
    case_name: str,
    run_collection_id: str,
    save_trajectories: bool,
    save_metadata: bool,
    save_figures: bool,
    compute_statistics: bool,
) -> OutputPaths:
    case_results_dir = (notebook_dir / f"../results/{case_name}").resolve()
    case_results_dir.mkdir(parents=True, exist_ok=True)

    traj_dir = case_results_dir / "trajectories"
    tmp_traj_dir = case_results_dir / "_tmp" / "trajectories"
    metadata_dir = case_results_dir / "metadata"
    metrics_dir = case_results_dir / "metrics"
    fig_dir = case_results_dir / "figures" / "trajectory_snapshots"

    run_out_dir = traj_dir if save_trajectories else tmp_traj_dir
    run_out_dir.mkdir(parents=True, exist_ok=True)

    if save_metadata:
        metadata_dir.mkdir(parents=True, exist_ok=True)

    if save_figures:
        fig_dir.mkdir(parents=True, exist_ok=True)

    if compute_statistics:
        metrics_dir.mkdir(parents=True, exist_ok=True)

    return OutputPaths(
        case_results_dir=case_results_dir,
        traj_dir=traj_dir,
        tmp_traj_dir=tmp_traj_dir,
        metadata_dir=metadata_dir,
        metrics_dir=metrics_dir,
        fig_dir=fig_dir,
        run_out_dir=run_out_dir,
        config_json=metadata_dir / f"{run_collection_id}_config.json",
    )


def print_output_paths(data_file, input_dir, paths: OutputPaths, save_trajectories: bool, save_figures: bool) -> None:
    print("\nInput / output")
    print(f"  input file        : {Path(data_file).name}")
    print(f"  input dir         : {input_dir}")
    print(f"  case results dir  : {paths.case_results_dir}")
    print(f"  trajectory dir    : {paths.run_out_dir}")
    print(f"  save trajectories : {save_trajectories}")
    print(f"  figure dir        : {paths.fig_dir}")
    print(f"  save figures      : {save_figures}")


# ============================================================
# PLOTTING HELPERS
# ============================================================

def _format_xy_axis(ax, *, equal: bool = True) -> None:
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    if equal:
        ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=getattr(ptheme, "GRID_ALPHA", 0.35))


def plot_trajectory_subset_per_class(
    trajectories: dict,
    *,
    Lx: float,
    Ly: float,
    run_title_info: str,
    case_name: str,
    fig_dir: Path,
    save_figures: bool,
    n_sample: int | None = None,
    class_style_map: dict | None = None,
    show: bool = True,
):
    """
    Plot one subset-of-trajectories figure per particle class.
    """
    apply_notebook_style()

    if class_style_map is None:
        class_style_map = build_class_style_map(trajectories)

    common_release_ids = get_common_release_ids(trajectories)

    if n_sample is None:
        n_sample = getattr(ptheme, "TRAJECTORY_SAMPLE_N", 300)

    n_plot = min(n_sample, len(common_release_ids))

    rng = np.random.default_rng(getattr(ptheme, "RANDOM_SEED", 42))
    selected_release_ids = rng.choice(common_release_ids, size=n_plot, replace=False)

    figures = {}

    for particle_tag, item in trajectories.items():
        ds = item["ds"]

        xname, yname = get_xy_names(ds)
        release_ids = get_release_ids(ds)

        selected_mask = np.isin(release_ids, selected_release_ids)
        trajectory_indices = np.where(selected_mask)[0]

        style = class_style_map[particle_tag]

        fig, ax = plt.subplots(figsize=get_figsize("map"), facecolor="white")
        ax.set_facecolor("white")

        for j, traj_idx in enumerate(trajectory_indices):
            xi = ds[xname].isel(trajectory=traj_idx).values
            yi = ds[yname].isel(trajectory=traj_idx).values

            xi_plot, yi_plot = insert_nan_at_periodic_jumps(xi, yi, Lx, Ly)

            ax.plot(
                to_km(xi_plot),
                to_km(yi_plot),
                color=style["color"],
                lw=getattr(ptheme, "TRAJECTORY_LINE_WIDTH", 0.7),
                alpha=getattr(ptheme, "TRAJECTORY_ALPHA", 0.45),
                label=style["label"] if j == 0 else "_nolegend_",
            )

        _format_xy_axis(ax)
        ax.set_title(f"{run_title_info} | n={n_plot}/{len(common_release_ids)}")
        ax.legend(loc="upper right")
        fig.tight_layout()

        savefig_if_enabled(
            fig,
            fig_dir / f"{case_name}_{particle_tag}_trajectory_subset.png",
            save=save_figures,
        )

        figures[particle_tag] = fig

        if show:
            plt.show()

    return figures


def plot_position_snapshots_combined(
    trajectories: dict,
    *,
    obs_selections: Iterable[int | str],
    run_title_info: str,
    case_name: str,
    fig_dir: Path,
    save_figures: bool,
    class_style_map: dict | None = None,
    show: bool = True,
):
    """
    Plot combined particle-position snapshots for selected observation indices.
    """
    apply_notebook_style()

    if class_style_map is None:
        class_style_map = build_class_style_map(trajectories)

    reference_ds = next(iter(trajectories.values()))["ds"]
    n_obs = reference_ds.sizes["obs"]

    figures = {}

    for obs_selection in obs_selections:
        obs_idx = resolve_obs_index(obs_selection, n_obs)

        if obs_idx < 0 or obs_idx >= n_obs:
            print(f"Skipping invalid obs selection: {obs_selection}")
            continue

        elapsed_days = get_elapsed_time_days(reference_ds, obs_idx)

        fig, ax = plt.subplots(figsize=get_figsize("map"), facecolor="white")
        ax.set_facecolor("white")

        for particle_tag, item in trajectories.items():
            ds = item["ds"]
            style = class_style_map[particle_tag]

            xname, yname = get_xy_names(ds)

            xt = ds[xname].isel(obs=obs_idx).values
            yt = ds[yname].isel(obs=obs_idx).values

            ax.scatter(
                to_km(xt),
                to_km(yt),
                s=getattr(ptheme, "POSITION_MARKER_SIZE", 16),
                alpha=getattr(ptheme, "POSITION_ALPHA", 0.85),
                linewidths=getattr(ptheme, "POSITION_EDGEWIDTH", 0.25),
                color=style["color"],
                marker=style["marker"],
                label=style["label"],
            )

        _format_xy_axis(ax)
        ax.set_title(f"T={elapsed_days:.2f} d | {run_title_info}")
        ax.legend(loc="upper right")
        fig.tight_layout()

        savefig_if_enabled(
            fig,
            fig_dir / f"{case_name}_positions_obs{obs_idx:04d}_combined.png",
            save=save_figures,
        )

        figures[obs_idx] = fig

        if show:
            plt.show()

    return figures


def plot_position_snapshots_per_class(
    trajectories: dict,
    *,
    obs_selections: Iterable[int | str],
    run_title_info: str,
    case_name: str,
    level_tag: str,
    release_time_index: int,
    fig_dir: Path,
    save_figures: bool,
    class_style_map: dict | None = None,
    show: bool = True,
):
    """
    Plot one particle-position snapshot per class and selected observation.
    """
    apply_notebook_style()

    if class_style_map is None:
        class_style_map = build_class_style_map(trajectories)

    reference_ds = next(iter(trajectories.values()))["ds"]
    n_obs = reference_ds.sizes["obs"]

    figures = {}

    for obs_selection in obs_selections:
        obs_idx = resolve_obs_index(obs_selection, n_obs)

        if obs_idx < 0 or obs_idx >= n_obs:
            print(f"Skipping invalid obs selection: {obs_selection}")
            continue

        for particle_tag, item in trajectories.items():
            ds = item["ds"]
            style = class_style_map[particle_tag]

            elapsed_days = get_elapsed_time_days(ds, obs_idx)

            xname, yname = get_xy_names(ds)

            xt = ds[xname].isel(obs=obs_idx).values
            yt = ds[yname].isel(obs=obs_idx).values

            fig, ax = plt.subplots(figsize=get_figsize("map"), facecolor="white")
            ax.set_facecolor("white")

            ax.scatter(
                to_km(xt),
                to_km(yt),
                s=getattr(ptheme, "POSITION_MARKER_SIZE", 16),
                alpha=getattr(ptheme, "POSITION_ALPHA", 0.85),
                linewidths=getattr(ptheme, "POSITION_EDGEWIDTH", 0.25),
                color=style["color"],
                marker=style["marker"],
                label=style["label"],
            )

            _format_xy_axis(ax)
            ax.set_title(f"T={elapsed_days:.2f} d | {run_title_info}")
            ax.legend(loc="upper right")
            fig.tight_layout()

            savefig_if_enabled(
                fig,
                fig_dir / (
                    f"{particle_tag}_{level_tag}_"
                    f"release_t{release_time_index:04d}_"
                    f"positions_obs{obs_idx:04d}.png"
                ),
                save=save_figures,
            )

            figures[(particle_tag, obs_idx)] = fig

            if show:
                plt.show()

    return figures


# ============================================================
# METADATA / LABEL HELPERS
# ============================================================

def _effective_stokes_number(info: dict, spec: dict) -> float:
    """Calculate St from tau_p and flow timescale, with metadata fallback."""
    tau_p = info.get("tau_p_seconds", spec.get("tau_p_seconds"))
    flow_timescale = info.get(
        "flow_timescale_seconds",
        spec.get("flow_timescale_seconds"),
    )

    try:
        tau_p = float(tau_p)
        flow_timescale = float(flow_timescale)
        if tau_p >= 0.0 and flow_timescale > 0.0:
            return tau_p / flow_timescale
    except (TypeError, ValueError):
        pass

    stored = info.get("stokes_number", spec.get("stokes_number", np.nan))

    try:
        return float(stored)
    except (TypeError, ValueError):
        return np.nan

def compact_particle_label(
    info: dict | None = None,
    spec: dict | None = None,
) -> str:
    """Build a compact label from tau_p and effective Stokes number."""
    info = info or {}
    spec = spec or {}

    particle_class = info.get(
        "particle_class",
        spec.get("particle_class", ""),
    )

    if particle_class == "passive":
        return "Passive particles"

    if particle_class == "mr_sm":
        tau_p_seconds = info.get(
            "tau_p_seconds",
            spec.get("tau_p_seconds"),
        )

        try:
            tau_txt = _tau_display_text(float(tau_p_seconds))
        except (TypeError, ValueError):
            tau_txt = "?"

        stokes_number = _effective_stokes_number(info, spec)

        if np.isfinite(stokes_number):
            return f"τₚ={tau_txt}, St={stokes_number:.3g}"

        return f"τₚ={tau_txt}"

    return str(
        info.get(
            "particle_label",
            spec.get("label", particle_class or "particle"),
        )
    )


def apply_label_overrides(trajectories: dict, label_overrides: dict[str, str] | None = None) -> dict:
    """
    Add/replace display labels without changing the stored metadata.

    Parameters
    ----------
    label_overrides
        Keys can be particle tags. Values are the exact labels used in legends.
    """
    label_overrides = label_overrides or {}

    out = {}
    for tag, item in trajectories.items():
        item_new = dict(item)
        if tag in label_overrides:
            item_new["display_label"] = label_overrides[tag]
        else:
            item_new["display_label"] = compact_particle_label(
                item_new.get("info", {}),
                item_new.get("spec", {}),
            )
        out[tag] = item_new

    return out


def particle_metadata_table(trajectories: dict) -> pd.DataFrame:
    """Return a compact tau-only particle metadata table."""
    rows = []

    for particle_tag, item in trajectories.items():
        info = item.get("info", {}) or {}
        spec = item.get("spec", {}) or {}

        tau_p = info.get("tau_p_seconds", spec.get("tau_p_seconds"))
        flow_timescale = info.get(
            "flow_timescale_seconds",
            spec.get("flow_timescale_seconds"),
        )

        rows.append(
            {
                "particle_tag": particle_tag,
                "display_label": item.get(
                    "display_label",
                    item.get("label", particle_tag),
                ),
                "particle_class": info.get(
                    "particle_class",
                    spec.get("particle_class"),
                ),
                "tau_p_seconds": tau_p,
                "flow_timescale_seconds": flow_timescale,
                "stokes_number": _effective_stokes_number(info, spec),
                "output_path": item.get("path", info.get("output_path")),
            }
        )

    return pd.DataFrame(rows)


def short_time_title(run_case: str, elapsed_days: float | None = None) -> str:
    """Compact plot title: only time and run case."""
    if elapsed_days is None or not np.isfinite(float(elapsed_days)):
        return str(run_case)
    return f"T={float(elapsed_days):.2f} d | {run_case}"


# ============================================================
# COUPLED FLOW-PARTICLE ANALYSIS HELPERS
# ============================================================

def resolve_obs_indices(selections, n_obs: int) -> list[int]:
    """Resolve and deduplicate observation selections."""
    resolved = []
    for selection in selections:
        obs_idx = resolve_obs_index(selection, n_obs)
        if 0 <= obs_idx < n_obs:
            resolved.append(int(obs_idx))
        else:
            print(f"Skipping invalid particle obs selection: {selection}")
    return sorted(set(resolved))


def _numeric_time_seconds(value) -> float:
    arr = np.asarray(value)

    if np.issubdtype(arr.dtype, np.timedelta64):
        return float(arr / np.timedelta64(1, "s"))
    if np.issubdtype(arr.dtype, np.datetime64):
        return np.nan
    return float(arr)


def particle_absolute_time_seconds(
    ds_traj: xr.Dataset,
    obs_idx: int,
    *,
    ds_parcels: xr.Dataset,
    release_time_index: int,
) -> float:
    """Return the absolute particle time in seconds for one saved observation."""
    t = ds_traj["time"].isel(trajectory=0, obs=obs_idx).values
    t_seconds = _numeric_time_seconds(t)

    if np.isfinite(t_seconds):
        return t_seconds

    t0 = ds_traj["time"].isel(trajectory=0, obs=0).values
    elapsed_seconds = float((t - t0) / np.timedelta64(1, "s"))
    release_seconds = float(ds_parcels["time"].isel(time=release_time_index))
    return release_seconds + elapsed_seconds


def build_particle_flow_time_alignment(
    reference_ds: xr.Dataset,
    obs_indices: Iterable[int],
    *,
    ds_parcels: xr.Dataset,
    release_time_index: int,
) -> pd.DataFrame:
    """Match each particle output time to the nearest available flow time."""
    flow_times_seconds = np.asarray(ds_parcels["time"].values, dtype=float)
    records = []

    for obs_idx in obs_indices:
        particle_time_seconds = particle_absolute_time_seconds(
            reference_ds,
            int(obs_idx),
            ds_parcels=ds_parcels,
            release_time_index=release_time_index,
        )
        flow_idx = int(np.argmin(np.abs(flow_times_seconds - particle_time_seconds)))
        flow_time_seconds = float(flow_times_seconds[flow_idx])
        mismatch_seconds = abs(flow_time_seconds - particle_time_seconds)

        records.append(
            {
                "obs": int(obs_idx),
                "particle_time_seconds": particle_time_seconds,
                "elapsed_days": get_elapsed_time_days(reference_ds, int(obs_idx)),
                "flow_time_index": flow_idx,
                "flow_time_seconds": flow_time_seconds,
                "mismatch_seconds": mismatch_seconds,
            }
        )

    return pd.DataFrame(records).set_index("obs")


def build_selected_flow_diagnostics(
    ds_flow_raw: xr.Dataset,
    time_alignment: pd.DataFrame,
    *,
    analysis_obs_indices: Iterable[int],
    flow_level_indices: Iterable[int],
    dx: float,
    dy: float,
    f0: float,
    ds_parcels: xr.Dataset,
    time_dim: str,
    z_dim: str,
    compute_surface_kinematics,
    case_name: str,
    run_collection_id: str,
    data_file: str | Path,
) -> xr.Dataset:
    """Compute the selected Eulerian flow diagnostics with the shared formula."""
    analysis_obs_indices = list(analysis_obs_indices)
    flow_level_indices = tuple(flow_level_indices)

    ro_stack = []
    div_stack = []
    strain_stack = []

    for _, row in time_alignment.loc[analysis_obs_indices].iterrows():
        flow_idx = int(row["flow_time_index"])

        ro, div_f, strain_f = compute_surface_kinematics(
            ds_flow_raw,
            time_dim=time_dim,
            z_dim=z_dim,
            it=flow_idx,
            levels=list(flow_level_indices),
            dx=dx,
            dy=dy,
            f0=f0,
        )

        ro_stack.append(np.asarray(ro, dtype=np.float32))
        div_stack.append(np.asarray(div_f, dtype=np.float32))
        strain_stack.append(np.asarray(strain_f, dtype=np.float32))

    return xr.Dataset(
        data_vars={
            "ro": (
                ("obs", "y", "x"),
                np.stack(ro_stack, axis=0),
                {"long_name": "Rossby number", "definition": "zeta / f0", "units": "1"},
            ),
            "div_f": (
                ("obs", "y", "x"),
                np.stack(div_stack, axis=0),
                {
                    "long_name": "normalized horizontal divergence",
                    "definition": "div_h(u) / f0",
                    "units": "1",
                },
            ),
            "strain_f": (
                ("obs", "y", "x"),
                np.stack(strain_stack, axis=0),
                {
                    "long_name": "normalized horizontal strain",
                    "definition": "sigma / abs(f0)",
                    "units": "1",
                },
            ),
        },
        coords={
            "obs": np.asarray(analysis_obs_indices, dtype=int),
            "x": np.asarray(ds_parcels["x"].values, dtype=float),
            "y": np.asarray(ds_parcels["y"].values, dtype=float),
            "flow_time_index": (
                "obs",
                time_alignment.loc[analysis_obs_indices, "flow_time_index"].values.astype(int),
            ),
            "time_seconds": (
                "obs",
                time_alignment.loc[analysis_obs_indices, "flow_time_seconds"].values.astype(float),
            ),
            "elapsed_days": (
                "obs",
                time_alignment.loc[analysis_obs_indices, "elapsed_days"].values.astype(float),
            ),
            "time_mismatch_seconds": (
                "obs",
                time_alignment.loc[analysis_obs_indices, "mismatch_seconds"].values.astype(float),
            ),
        },
        attrs={
            "case_name": case_name,
            "run_collection_id": run_collection_id,
            "source_file": str(data_file),
            "flow_formula_source": "shcherbina_utils.compute_surface_kinematics",
            "flow_level_indices": ",".join(map(str, flow_level_indices)),
            "dx_m": float(dx),
            "dy_m": float(dy),
            "f0_s-1": float(f0),
            "analysis_particle_obs": ",".join(map(str, analysis_obs_indices)),
        },
    )


def load_or_recompute_cluster_data(
    particle_tag: str,
    ds_traj: xr.Dataset,
    *,
    particle_level_tag: str,
    release_time_index: int,
    metrics_dir: str | Path,
    analysis_obs_indices: Iterable[int],
    domain,
    metrics_clustering,
    fallback_threshold: float = 0.5,
) -> xr.Dataset:
    """Load saved particle clustering fields, or recompute them if needed."""
    analysis_obs_indices = list(analysis_obs_indices)
    metrics_dir = Path(metrics_dir)
    stem = f"{particle_tag}_{particle_level_tag}_release_t{release_time_index:04d}"

    cluster_path = metrics_dir / "voronoi" / f"{stem}_clustered_state_timeseries.nc"
    if cluster_path.exists():
        with xr.open_dataset(cluster_path) as ds_open:
            cluster_full = ds_open.load()

        if np.all(
            np.isin(
                analysis_obs_indices,
                np.asarray(cluster_full["obs"].values),
            )
        ):
            print(f"  using saved Voronoï metrics: {cluster_path.name}")
            return cluster_full.sel(obs=analysis_obs_indices)

        print(
            "  saved Voronoï metrics do not contain every requested obs; "
            "recomputing."
        )

    pdf_path = metrics_dir / "voronoi" / f"{stem}_aggregate_pdf.nc"
    if pdf_path.exists():
        with xr.open_dataset(pdf_path) as ds_open:
            class_threshold = float(ds_open["cluster_threshold_nu_c"].load().values)
    else:
        class_threshold = np.nan

    if not np.isfinite(class_threshold) or class_threshold <= 0.0:
        class_threshold = float(fallback_threshold)
        print(
            "  warning: no valid saved aggregate threshold; "
            f"using fallback nu_c={class_threshold:.3f}"
        )

    return metrics_clustering.compute_voronoi_clustering_timeseries(
        ds_traj=ds_traj,
        obs_indices=analysis_obs_indices,
        domain=domain,
        threshold_method="fixed",
        fixed_threshold_area_norm=class_threshold,
        log_area=False,
    )


def extend_periodic_snapshot(flow_snapshot: xr.Dataset, domain) -> xr.Dataset:
    """Add one periodic halo cell around an xarray flow snapshot."""
    left = flow_snapshot.isel(x=[-1]).assign_coords(
        x=[float(flow_snapshot["x"].values[-1] - domain.Lx)]
    )
    right = flow_snapshot.isel(x=[0]).assign_coords(
        x=[float(flow_snapshot["x"].values[0] + domain.Lx)]
    )
    extended_x = xr.concat([left, flow_snapshot, right], dim="x")

    bottom = extended_x.isel(y=[-1]).assign_coords(
        y=[float(extended_x["y"].values[-1] - domain.Ly)]
    )
    top = extended_x.isel(y=[0]).assign_coords(
        y=[float(extended_x["y"].values[0] + domain.Ly)]
    )
    return xr.concat([bottom, extended_x, top], dim="y")


def sample_flow_snapshot_at_particles(
    flow_snapshot: xr.Dataset,
    xp,
    yp,
    *,
    domain,
    wrap_to_domain,
) -> dict[str, np.ndarray]:
    """Sample Ro, divergence and strain at periodic particle positions."""
    trajectory_coord = np.arange(len(xp), dtype=int)
    xp_wrapped, yp_wrapped = wrap_to_domain(
        np.asarray(xp, dtype=float),
        np.asarray(yp, dtype=float),
        domain,
    )

    xp_da = xr.DataArray(
        xp_wrapped,
        dims="trajectory",
        coords={"trajectory": trajectory_coord},
    )
    yp_da = xr.DataArray(
        yp_wrapped,
        dims="trajectory",
        coords={"trajectory": trajectory_coord},
    )

    flow_periodic = extend_periodic_snapshot(flow_snapshot, domain)
    sampled = {}
    for name in ["ro", "div_f", "strain_f"]:
        sampled[name] = np.asarray(
            flow_periodic[name].interp(x=xp_da, y=yp_da).values,
            dtype=np.float32,
        )
    return sampled


def build_coupled_class_dataset(
    particle_tag: str,
    item: dict,
    cluster_selected: xr.Dataset,
    *,
    ds_flow: xr.Dataset,
    analysis_obs_indices: Iterable[int],
    time_alignment: pd.DataFrame,
    domain,
    wrap_to_domain,
    case_name: str,
    run_title_info: str,
    run_collection_id: str,
) -> xr.Dataset:
    """Build the compact per-class particle-flow coupling dataset."""
    analysis_obs_indices = list(analysis_obs_indices)
    ds_traj = item["ds"]
    xname, yname = get_xy_names(ds_traj)

    n_particles = ds_traj.sizes["trajectory"]
    n_selected = len(analysis_obs_indices)

    x_particle = np.full((n_particles, n_selected), np.nan, dtype=np.float32)
    y_particle = np.full_like(x_particle, np.nan)
    ro_particle = np.full_like(x_particle, np.nan)
    div_particle = np.full_like(x_particle, np.nan)
    strain_particle = np.full_like(x_particle, np.nan)

    for j, obs_idx in enumerate(analysis_obs_indices):
        xp = np.asarray(ds_traj[xname].isel(obs=obs_idx).values, dtype=float)
        yp = np.asarray(ds_traj[yname].isel(obs=obs_idx).values, dtype=float)

        x_particle[:, j] = xp.astype(np.float32)
        y_particle[:, j] = yp.astype(np.float32)

        sampled = sample_flow_snapshot_at_particles(
            ds_flow.sel(obs=obs_idx),
            xp,
            yp,
            domain=domain,
            wrap_to_domain=wrap_to_domain,
        )
        ro_particle[:, j] = sampled["ro"]
        div_particle[:, j] = sampled["div_f"]
        strain_particle[:, j] = sampled["strain_f"]

    area = np.asarray(cluster_selected["voronoi_area"].values, dtype=np.float32)
    area_norm = np.asarray(cluster_selected["voronoi_area_norm"].values, dtype=np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        concentration = 1.0 / area
        concentration_norm = 1.0 / area_norm

    concentration[~np.isfinite(concentration)] = np.nan
    concentration_norm[~np.isfinite(concentration_norm)] = np.nan

    data_vars = {
        "x_particle": (("trajectory", "obs"), x_particle, {"units": "m"}),
        "y_particle": (("trajectory", "obs"), y_particle, {"units": "m"}),
        "ro_particle": (
            ("trajectory", "obs"),
            ro_particle,
            {"definition": "zeta/f0 sampled at particle position"},
        ),
        "div_f_particle": (
            ("trajectory", "obs"),
            div_particle,
            {"definition": "div_h(u)/f0 sampled at particle position"},
        ),
        "strain_f_particle": (
            ("trajectory", "obs"),
            strain_particle,
            {"definition": "sigma/abs(f0) sampled at particle position"},
        ),
        "voronoi_area": (("trajectory", "obs"), area, {"units": "m2"}),
        "voronoi_area_norm": (
            ("trajectory", "obs"),
            area_norm,
            {"definition": "A / mean(A) at each snapshot"},
        ),
        "voronoi_concentration": (
            ("trajectory", "obs"),
            concentration.astype(np.float32),
            {"units": "m-2", "definition": "1/A"},
        ),
        "particle_concentration_norm": (
            ("trajectory", "obs"),
            concentration_norm.astype(np.float32),
            {"units": "1", "definition": "mean(A)/A = 1/(A/mean(A))"},
        ),
    }

    for name in ["cluster_flag", "cluster_flag_tolerant", "gap_filled_flag"]:
        if name in cluster_selected:
            data_vars[name] = (
                ("trajectory", "obs"),
                np.asarray(cluster_selected[name].values),
            )

    coords = {
        "trajectory": np.arange(n_particles, dtype=int),
        "obs": np.asarray(analysis_obs_indices, dtype=int),
        "time_seconds": (
            "obs",
            time_alignment.loc[analysis_obs_indices, "particle_time_seconds"].values.astype(float),
        ),
        "elapsed_days": (
            "obs",
            time_alignment.loc[analysis_obs_indices, "elapsed_days"].values.astype(float),
        ),
        "flow_time_index": (
            "obs",
            time_alignment.loc[analysis_obs_indices, "flow_time_index"].values.astype(int),
        ),
    }

    coords["release_id"] = ("trajectory", get_release_ids(ds_traj).astype(int))

    return xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "case_name": case_name,
            "run_title_info": run_title_info,
            "run_collection_id": run_collection_id,
            "particle_tag": particle_tag,
            "particle_label": item.get("display_label", item.get("label", particle_tag)),
            "source_trajectory": str(item["path"]),
            "flow_formula_source": "shcherbina_utils.compute_surface_kinematics",
            "voronoi_formula_source": (
                "saved clustered_state_timeseries.nc generated by "
                "metrics_clustering.compute_voronoi_clustering_timeseries"
            ),
            "analysis_particle_obs": ",".join(map(str, analysis_obs_indices)),
        },
    )


def cluster_var_name(ds_c: xr.Dataset, *, use_tolerant: bool = True) -> str:
    """Choose the raw or one-frame-tolerant clustering flag."""
    if use_tolerant and "cluster_flag_tolerant" in ds_c:
        return "cluster_flag_tolerant"
    return "cluster_flag"


def snapshot_elapsed_days(obs_idx: int, ds_like: xr.Dataset) -> float:
    return float(ds_like["elapsed_days"].sel(obs=obs_idx).values)


def conditional_edges(
    flow_snapshot: xr.Dataset,
    x_flow_var: str,
    y_flow_var: str,
    *,
    nbins: int,
    percentiles=(0.1, 99.9),
    y_nonnegative: bool = True,
):
    x = np.asarray(flow_snapshot[x_flow_var].values, dtype=float)
    y = np.asarray(flow_snapshot[y_flow_var].values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x1 = x[mask]
    y1 = y[mask]

    p0, p1 = percentiles
    x_min = np.nanpercentile(x1, p0)
    x_max = np.nanpercentile(x1, p1)
    y_min = np.nanpercentile(y1, p0)
    y_max = np.nanpercentile(y1, p1)

    if y_nonnegative:
        y_min = max(0.0, y_min)

    x_pad = 0.03 * (x_max - x_min) if x_max > x_min else 0.1
    y_pad = 0.03 * (y_max - y_min) if y_max > y_min else 0.1

    x_edges = np.linspace(x_min - x_pad, x_max + x_pad, nbins + 1)
    y_edges = np.linspace(
        max(0.0, y_min - y_pad) if y_nonnegative else y_min - y_pad,
        y_max + y_pad,
        nbins + 1,
    )
    return x_edges, y_edges


def global_conditional_edges(
    flow_dataset: xr.Dataset,
    obs_indices: Iterable[int],
    x_flow_var: str,
    y_flow_var: str,
    *,
    nbins: int,
    percentiles=(0.1, 99.9),
    y_nonnegative: bool = True,
):
    """Determine common conditional-bin edges over selected snapshots."""
    obs_indices = list(obs_indices)
    x_all = np.asarray(
        flow_dataset[x_flow_var].sel(obs=obs_indices).values,
        dtype=float,
    ).ravel()
    y_all = np.asarray(
        flow_dataset[y_flow_var].sel(obs=obs_indices).values,
        dtype=float,
    ).ravel()

    valid = np.isfinite(x_all) & np.isfinite(y_all)
    x_all = x_all[valid]
    y_all = y_all[valid]

    p0, p1 = percentiles
    x_min, x_max = np.nanpercentile(x_all, [p0, p1])
    y_min, y_max = np.nanpercentile(y_all, [p0, p1])

    if y_nonnegative:
        y_min = max(0.0, y_min)

    x_padding = 0.03 * (x_max - x_min) if x_max > x_min else 0.1
    y_padding = 0.03 * (y_max - y_min) if y_max > y_min else 0.1

    x_edges = np.linspace(x_min - x_padding, x_max + x_padding, nbins + 1)
    y_edges = np.linspace(
        max(0.0, y_min - y_padding) if y_nonnegative else y_min - y_padding,
        y_max + y_padding,
        nbins + 1,
    )
    return x_edges, y_edges


def conditioned_mean_2d(
    x,
    y,
    z,
    x_edges,
    y_edges,
    *,
    conditioned_mean_func,
    min_count: int,
):
    mean, count = conditioned_mean_func(x, y, z, x_edges, y_edges)
    mean = np.asarray(mean, dtype=float)
    count = np.asarray(count, dtype=float)
    mean[count < min_count] = np.nan
    return mean, count


def conditional_cluster_probability_2d(
    x,
    y,
    cluster_flag,
    x_edges,
    y_edges,
    *,
    min_count: int,
):
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(cluster_flag)
    x = np.asarray(x[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)
    c = np.asarray(cluster_flag[mask], dtype=float)

    count, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    count_cluster, _, _ = np.histogram2d(
        x,
        y,
        bins=[x_edges, y_edges],
        weights=c,
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        prob = count_cluster / count
    prob[count < min_count] = np.nan
    return prob, count


def particle_regime_masks(ro, strain) -> dict[str, np.ndarray]:
    ro = np.asarray(ro, dtype=float)
    strain = np.asarray(strain, dtype=float)
    finite = np.isfinite(ro) & np.isfinite(strain)
    avd = finite & (strain < np.abs(ro)) & (ro < 0.0)
    cvd = finite & (strain < np.abs(ro)) & (ro > 0.0)
    sd = finite & (strain >= np.abs(ro))
    return {"AVD": avd, "SD": sd, "CVD": cvd}


def plot_conditional_panel(
    field2d,
    x_edges,
    y_edges,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    cbar_label: str,
    plot_utils,
    cmap="viridis",
    vmin=None,
    vmax=None,
    log_color: bool = False,
    add_ro_regimes: bool = False,
):
    """Plot a conditional field using the exact coupled-notebook styling."""
    from matplotlib.colors import LogNorm

    fig, ax = plt.subplots(figsize=plot_utils.get_figsize("square"), facecolor="white")

    plot_kwargs = {"shading": "auto", "cmap": cmap}
    if log_color and (vmin is not None) and (vmin > 0):
        plot_kwargs["norm"] = LogNorm(vmin=vmin, vmax=vmax)
    else:
        if vmin is not None:
            plot_kwargs["vmin"] = vmin
        if vmax is not None:
            plot_kwargs["vmax"] = vmax

    pcm = ax.pcolormesh(x_edges, y_edges, np.asarray(field2d).T, **plot_kwargs)
    cbar = plot_utils.add_colorbar(fig, ax, pcm)
    cbar.set_label(cbar_label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(float(x_edges[0]), float(x_edges[-1]))
    ax.set_ylim(float(y_edges[0]), float(y_edges[-1]))

    if add_ro_regimes:
        xx = np.linspace(x_edges[0], x_edges[-1], 500)
        ax.plot(
            xx,
            np.abs(xx),
            "--",
            color=plot_utils.get_color("reference"),
            lw=plot_utils._theme_attr("LINE_WIDTH", 1.2),
            alpha=plot_utils._theme_attr("REFERENCE_ALPHA", 0.8),
        )
        plot_utils.add_regime_labels(ax, x_edges, y_edges)

    fig.tight_layout()
    return fig, ax


def conditioned_geometric_mean_2d(
    x,
    y,
    concentration,
    x_edges,
    y_edges,
    *,
    smoothing_sigma: float = 1.0,
    alpha_count_scale: float = 8.0,
):
    """Smoothed conditional geometric mean used in the original notebook."""
    from scipy.ndimage import gaussian_filter

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    concentration = np.asarray(concentration, dtype=float)

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(concentration)
        & (concentration > 0.0)
    )

    x = x[valid]
    y = y[valid]
    log_c = np.log(concentration[valid])

    count, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    log_sum, _, _ = np.histogram2d(
        x,
        y,
        bins=[x_edges, y_edges],
        weights=log_c,
    )

    count_smooth = gaussian_filter(count, sigma=smoothing_sigma, mode="nearest")
    log_sum_smooth = gaussian_filter(log_sum, sigma=smoothing_sigma, mode="nearest")

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_log_c = log_sum_smooth / count_smooth
        concentration_geo = np.exp(mean_log_c)

    concentration_geo[count_smooth <= 0.0] = np.nan
    alpha = 1.0 - np.exp(-count_smooth / float(alpha_count_scale))
    alpha = np.clip(alpha, 0.0, 1.0)
    return concentration_geo, count_smooth, alpha


def plot_flow_particle_overlay(
    *,
    flow_var: str,
    obs_idx: int,
    particle_tag: str,
    coupled_outputs: dict,
    class_style_map: dict,
    ds_flow: xr.Dataset,
    plot_utils,
    plot_theme,
    dx: float,
    dy: float,
    run_title_info: str,
    rossby_limits,
    divergence_limits,
    plot_only_clustered_particles: bool,
    show_all_particles_faint: bool,
    use_tolerant_cluster_flag: bool,
    particle_point_size: float,
    particle_point_alpha: float,
    faint_point_alpha: float,
    save_figures: bool,
    coupled_fig_dir: str | Path,
    flow_level_tag: str,
):
    """Plot the original Rossby/divergence overlay without changing its style."""
    ds_c = coupled_outputs[particle_tag]["ds"]
    style = class_style_map[particle_tag]
    label = coupled_outputs[particle_tag]["label"]
    cvar = cluster_var_name(ds_c, use_tolerant=use_tolerant_cluster_flag)

    flow_snapshot = ds_flow.sel(obs=obs_idx)
    field = np.asarray(flow_snapshot[flow_var].values, dtype=float)
    elapsed_days = float(ds_c["elapsed_days"].sel(obs=obs_idx).values)

    if flow_var == "ro":
        field_title = "Rossby number"
        cbar_label = r"$\zeta/f$"
        vmin, vmax = rossby_limits
        output_group = "rossby_overlays"
    elif flow_var == "div_f":
        field_title = "Horizontal divergence"
        cbar_label = r"$\delta/f$"
        vmin, vmax = divergence_limits
        output_group = "divergence_overlays"
    else:
        raise ValueError(f"Unsupported flow overlay variable: {flow_var}")

    fig, ax = plot_utils.plot_map(
        field,
        title=f"{field_title} (t={elapsed_days:.2f} days)\n{run_title_info}",
        cbar_label=cbar_label,
        cmap=plot_utils.get_cmap("rossby"),
        vmin=vmin,
        vmax=vmax,
        dx=dx,
        dy=dy,
    )

    xp = np.asarray(ds_c["x_particle"].sel(obs=obs_idx).values, dtype=float)
    yp = np.asarray(ds_c["y_particle"].sel(obs=obs_idx).values, dtype=float)
    valid = np.isfinite(xp) & np.isfinite(yp)

    clustered = (
        np.asarray(ds_c[cvar].sel(obs=obs_idx).values, dtype=bool)
        if cvar in ds_c
        else np.zeros_like(valid, dtype=bool)
    )
    clustered &= valid

    if show_all_particles_faint and not plot_only_clustered_particles:
        ax.scatter(
            xp[valid] / 1000.0,
            yp[valid] / 1000.0,
            s=0.7 * particle_point_size,
            color="0.2",
            alpha=faint_point_alpha,
            linewidths=0.0,
            label="all particles",
            zorder=3,
        )

    scatter_mask = clustered if plot_only_clustered_particles else valid
    scatter_label = (
        f"{label} (clustered state)"
        if plot_only_clustered_particles
        else label
    )

    ax.scatter(
        xp[scatter_mask] / 1000.0,
        yp[scatter_mask] / 1000.0,
        s=particle_point_size,
        color=style["color"],
        alpha=particle_point_alpha,
        linewidths=0.0,
        label=scatter_label,
        zorder=4,
    )

    ax.legend(
        loc="upper right",
        fontsize=getattr(plot_theme, "LEGEND_FONTSIZE", 12),
    )
    fig.tight_layout()

    if save_figures:
        out_dir = Path(coupled_fig_dir) / output_group
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_utils.savefig_if_needed(
            fig,
            f"{flow_var}_{particle_tag}_{flow_level_tag}_obs{obs_idx:04d}",
            save=True,
            out_dir=out_dir,
        )

    plt.show()
    return fig, ax

# ============================================================
# FLOW-PARTICLE COUPLING: CONDITIONAL PDF HELPERS
# ============================================================

from matplotlib.colors import LogNorm

def compute_2d_particle_pdf(x, y, x_bins, y_bins):
    """
    Compute a 2D particle PDF p(x, y) from particle samples.

    Returns
    -------
    pdf_T : 2D array
        Transposed PDF array, ready for pcolormesh(x_bins, y_bins, pdf_T).
    counts_T : 2D array
        Transposed raw counts per bin.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    counts, x_edges, y_edges = np.histogram2d(x, y, bins=[x_bins, y_bins])

    total = counts.sum()

    if total <= 0:
        pdf = np.full_like(
            counts,
            np.nan,
            dtype=float,
        )
    else:
        # Fraction of all valid particles contained in each bin.
        pdf = counts / total
        pdf[counts == 0] = np.nan

    return pdf.T, counts.T


def draw_ro_strain_regime_guides(
    ax,
    x_min,
    x_max,
    y_max,
    *,
    line_color="0.4",
    line_style="--",
    line_width=1.2,
    text_color="0.25",
    text_size=18,
):
    """
    Draw the regime boundaries sigma = |zeta| and add AVD / SD / CVD labels.
    """
    # left branch: y = -x for x <= 0
    x_left = np.linspace(x_min, 0.0, 200)
    y_left = -x_left

    # right branch: y = x for x >= 0
    x_right = np.linspace(0.0, x_max, 200)
    y_right = x_right

    ax.plot(x_left, y_left, line_style, color=line_color, lw=line_width)
    ax.plot(x_right, y_right, line_style, color=line_color, lw=line_width)

    ax.text(x_min + 0.10 * (x_max - x_min), 0.10 * y_max, "AVD",
            color=text_color, fontsize=text_size, fontweight="bold")
    ax.text(0.50 * (x_min + x_max), 0.72 * y_max, "SD",
            color=text_color, fontsize=text_size, fontweight="bold", ha="center")
    ax.text(x_max - 0.22 * (x_max - x_min), 0.10 * y_max, "CVD",
            color=text_color, fontsize=text_size, fontweight="bold")