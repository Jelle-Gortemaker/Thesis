from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
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


# ============================================================
# PARTICLE / TRAJECTORY DATA HELPERS
# ============================================================

def build_particle_specs(
    inertial_relaxation_times_seconds: Iterable[float],
    flow_timescale_seconds: float,
) -> list[dict[str, Any]]:
    """
    Build the list of particle specifications used by the notebook.

    Always includes passive particles. Adds one inertial particle class per tau_p.
    """
    def tau_to_tag(tau):
        return f"tau{tau:g}s".replace(".", "p").replace("+", "")

    specs = [
        {
            "particle_class": "passive",
            "tau_p_seconds": 0.0,
            "label": "passive",
            "tag": "passive",
        }
    ]

    for tau in inertial_relaxation_times_seconds:
        st = float(tau) / float(flow_timescale_seconds)
        specs.append(
            {
                "particle_class": "inertial",
                "tau_p_seconds": float(tau),
                "label": f"St = {st:g}",
                "tag": f"inertial_{tau_to_tag(tau)}_St{st:g}".replace(".", "p"),
            }
        )

    return specs


def print_particle_specs(particle_specs: list[dict[str, Any]]) -> None:
    print("Particle classes:")
    for spec in particle_specs:
        print(f"  {spec['tag']} | {spec['label']} | tau_p = {spec['tau_p_seconds']} s")


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
    common_ids = None

    for item in trajectories.values():
        release_ids = get_release_ids(item["ds"])

        if common_ids is None:
            common_ids = set(release_ids.tolist())
        else:
            common_ids = common_ids.intersection(set(release_ids.tolist()))

    return np.array(sorted(common_ids), dtype=int)


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
    """
    Build a consistent color/marker map for each particle class.
    """
    particle_tags = list(trajectories.keys())

    if hasattr(ptheme, "get_class_style"):
        return {
            tag: {
                **ptheme.get_class_style(i, label=trajectories[tag]["label"]),
                "label": trajectories[tag]["label"],
            }
            for i, tag in enumerate(particle_tags)
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
            "label": item["label"],
            "color": cmap(i % cmap.N),
            "marker": marker_cycle[i % len(marker_cycle)],
        }

    return style_map


def load_trajectory_collection(
    run_outputs: dict,
    load_if_temporary: bool,
    loader,
) -> dict:
    """
    Load trajectories from run_outputs into the notebook's trajectory dictionary format.
    """
    trajectories = {}

    for particle_tag, item in run_outputs.items():
        ds_traj = loader(item["path"])

        if load_if_temporary:
            ds_traj = ds_traj.load()

        trajectories[particle_tag] = {
            "ds": ds_traj,
            "label": item["label"],
            "path": item["path"],
            "info": item["info"],
            "spec": item["spec"],
        }

        print(f"\nTrajectory QC: {particle_tag}")
        print(f"  label     : {item['label']}")
        print(f"  path      : {item['path']}")
        print(f"  dims      : {dict(ds_traj.sizes)}")
        print(f"  variables : {list(ds_traj.data_vars)}")

    return trajectories


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
        ax.set_title(
            f"Particle trajectories, subset of {n_plot} particles\n"
            f"{run_title_info}; {style['label']}"
        )
        ax.legend(loc="best")
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
        ax.set_title(
            f"Particle positions at t = {elapsed_days:.2f} days\n"
            f"{run_title_info}; obs = {obs_idx}"
        )
        ax.legend(loc="best")
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
            ax.set_title(
                f"Particle positions at t = {elapsed_days:.2f} days\n"
                f"{run_title_info}; {style['label']}; obs = {obs_idx}"
            )
            ax.legend(loc="best")
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
