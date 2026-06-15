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
