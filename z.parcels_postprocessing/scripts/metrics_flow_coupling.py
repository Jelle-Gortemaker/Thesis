from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


# ============================================================
# I/O HELPERS
# ============================================================

def open_dataset_auto(path: str | Path) -> xr.Dataset:
    """Open a NetCDF or Zarr xarray dataset."""
    path = Path(path)
    if path.suffix == ".zarr":
        return xr.open_zarr(path)
    return xr.open_dataset(path)


def save_dataset_auto(ds: xr.Dataset, path: str | Path, *, overwrite: bool = True) -> Path:
    """Save a dataset as NetCDF or Zarr, inferred from the suffix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".zarr":
        mode = "w" if overwrite else "w-"
        ds.to_zarr(path, mode=mode)
    else:
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        ds.to_netcdf(path)

    return path


def open_flow_diagnostics(path: str | Path) -> xr.Dataset:
    """
    Open a standard flow-diagnostics product.

    Expected dimensions/coordinates are normally:
        time [seconds], y [m], x [m]
    Expected variables include any of:
        ro, div_f, strain_f, speed, ow, ow_f2, zeta, div, strain
    """
    return open_dataset_auto(path)


def save_flow_diagnostics_dataset(
    ds_flow: xr.Dataset,
    path: str | Path,
    *,
    overwrite: bool = True,
) -> Path:
    """Save flow diagnostics in the standard coupling format."""
    return save_dataset_auto(ds_flow, path, overwrite=overwrite)


# ============================================================
# BASIC TRAJECTORY HELPERS
# ============================================================

def get_xy_names(ds: xr.Dataset) -> tuple[str, str]:
    """Return trajectory x/y variable names used by Parcels outputs."""
    xname = "lon" if "lon" in ds.variables else "x"
    yname = "lat" if "lat" in ds.variables else "y"
    return xname, yname


def _trajectory_dim(ds: xr.Dataset) -> str:
    if "trajectory" in ds.dims:
        return "trajectory"
    if "particle" in ds.dims:
        return "particle"
    raise KeyError("Could not find trajectory/particle dimension.")


def trajectory_time_values_seconds(ds_traj: xr.Dataset, obs_indices: Sequence[int]) -> np.ndarray:
    """Return trajectory time values in seconds relative to the first observation."""
    if "time" not in ds_traj:
        return np.asarray(obs_indices, dtype=float)

    traj_dim = _trajectory_dim(ds_traj)
    t = ds_traj["time"].isel({traj_dim: 0, "obs": list(obs_indices)}).values
    t0 = ds_traj["time"].isel({traj_dim: 0, "obs": 0}).values

    if np.issubdtype(np.asarray(t).dtype, np.datetime64):
        return ((t - t0) / np.timedelta64(1, "s")).astype(float)

    return np.asarray(t, dtype=float) - float(t0)


def elapsed_days_for_obs(ds_traj: xr.Dataset, obs_idx: int) -> float:
    """Elapsed days for one trajectory observation index."""
    return float(trajectory_time_values_seconds(ds_traj, [int(obs_idx)])[0] / 86400.0)


def default_flow_variables(ds_flow: xr.Dataset) -> tuple[str, ...]:
    """Variables sampled by default when they are available."""
    candidates = ("ro", "div_f", "strain_f", "speed", "ow", "ow_f2", "zeta", "div", "strain")
    return tuple(v for v in candidates if v in ds_flow.data_vars)


# ============================================================
# FLOW SAMPLING AT PARTICLE POSITIONS
# ============================================================

def sample_flow_at_particles(
    ds_traj: xr.Dataset,
    ds_flow: xr.Dataset,
    *,
    variables: Iterable[str] | None = None,
    obs_indices: Iterable[int] | None = None,
    flow_time_name: str = "time",
    flow_x_name: str = "x",
    flow_y_name: str = "y",
    method_time: str = "nearest",
) -> xr.Dataset:
    """
    Interpolate gridded flow diagnostics to particle positions.

    Output dimensions match the particle trajectories: (trajectory, obs).
    This dataset is the main bridge between flow diagnostics and particle
    clustering/residence-time products.
    """
    xname, yname = get_xy_names(ds_traj)
    traj_dim = _trajectory_dim(ds_traj)

    if variables is None:
        variables = default_flow_variables(ds_flow)
    variables = tuple(variables)

    if len(variables) == 0:
        raise ValueError("No flow variables selected/found for sampling.")

    missing = [v for v in variables if v not in ds_flow]
    if missing:
        raise KeyError(f"Flow variables not found in ds_flow: {missing}")

    if obs_indices is None:
        obs_indices = list(range(ds_traj.sizes["obs"]))
    else:
        obs_indices = list(obs_indices)

    n_particles = ds_traj.sizes[traj_dim]
    obs_coord = np.asarray(obs_indices, dtype=int)
    traj_coord = np.arange(n_particles, dtype=int)
    time_values = trajectory_time_values_seconds(ds_traj, obs_indices)

    out_vars = {
        f"{var}_particle": np.full((n_particles, len(obs_indices)), np.nan, dtype=np.float32)
        for var in variables
    }

    x_particle = np.full((n_particles, len(obs_indices)), np.nan, dtype=np.float32)
    y_particle = np.full((n_particles, len(obs_indices)), np.nan, dtype=np.float32)

    for j, obs_idx in enumerate(obs_indices):
        xp = np.asarray(ds_traj[xname].isel(obs=obs_idx).values, dtype=float)
        yp = np.asarray(ds_traj[yname].isel(obs=obs_idx).values, dtype=float)

        x_particle[:, j] = xp.astype(np.float32)
        y_particle[:, j] = yp.astype(np.float32)

        xp_da = xr.DataArray(xp, dims="trajectory", coords={"trajectory": traj_coord})
        yp_da = xr.DataArray(yp, dims="trajectory", coords={"trajectory": traj_coord})

        if flow_time_name in ds_flow.coords or flow_time_name in ds_flow.dims:
            flow_t = ds_flow.sel({flow_time_name: float(time_values[j])}, method=method_time)
        else:
            flow_t = ds_flow.isel({flow_time_name: int(obs_idx)})

        for var in variables:
            sampled = flow_t[var].interp({flow_x_name: xp_da, flow_y_name: yp_da})
            out_vars[f"{var}_particle"][:, j] = np.asarray(sampled.values, dtype=np.float32)

    ds_out = xr.Dataset(
        data_vars={
            "x_particle": (("trajectory", "obs"), x_particle),
            "y_particle": (("trajectory", "obs"), y_particle),
            **{
                name: (("trajectory", "obs"), data)
                for name, data in out_vars.items()
            },
        },
        coords={
            "trajectory": traj_coord,
            "obs": obs_coord,
            "time_seconds": ("obs", time_values.astype(float)),
            "time_days": ("obs", (time_values / 86400.0).astype(float)),
        },
        attrs={
            "description": "Flow diagnostics sampled at particle positions.",
            "source_flow_variables": ",".join(variables),
            "flow_time_name": flow_time_name,
            "flow_x_name": flow_x_name,
            "flow_y_name": flow_y_name,
        },
    )

    return ds_out


def build_coupled_particle_flow_dataset(
    ds_traj: xr.Dataset,
    ds_flow: xr.Dataset,
    *,
    cluster_ds: xr.Dataset | None = None,
    residence_labels_ds: xr.Dataset | None = None,
    variables: Iterable[str] | None = None,
    obs_indices: Iterable[int] | None = None,
) -> xr.Dataset:
    """
    Build the reusable coupled particle-flow dataset.

    Merges:
        sampled flow diagnostics at particles
        particle-level Voronoï area / cluster flags
        particle-level residence labels
    """
    sampled = sample_flow_at_particles(
        ds_traj,
        ds_flow,
        variables=variables,
        obs_indices=obs_indices,
    )

    datasets = [sampled]
    if cluster_ds is not None:
        datasets.append(cluster_ds)
    if residence_labels_ds is not None:
        datasets.append(residence_labels_ds)

    ds = xr.merge(datasets, compat="override", join="inner")

    if "voronoi_area_norm" in ds:
        a = ds["voronoi_area_norm"]
        ds["log_voronoi_area_norm"] = xr.where(a > 0.0, np.log(a), np.nan)
        ds["log10_voronoi_area_norm"] = xr.where(a > 0.0, np.log10(a), np.nan)
        ds["inverse_voronoi_area_norm"] = xr.where(a > 0.0, 1.0 / a, np.nan)
        ds["log_voronoi_area_norm"].attrs["description"] = "ln(A/<A>), where small values indicate dense particle clusters"
        ds["inverse_voronoi_area_norm"].attrs["description"] = "<A>/A, proxy for local particle concentration"

    ds.attrs.update(
        {
            "description": "Coupled particle-flow dataset: flow diagnostics sampled at particle positions and merged with clustering/residence labels.",
            "recommended_use": "Use this product for JPDFs, conditional clustering, residence-time conditioning, and particle-flow overlays.",
        }
    )

    return ds


# ============================================================
# JPDF AND CONDITIONAL STATISTICS
# ============================================================

def _finite_flat(*arrays):
    flat = [np.asarray(a, dtype=float).ravel() for a in arrays]
    mask = np.ones_like(flat[0], dtype=bool)
    for a in flat:
        mask &= np.isfinite(a)
    return [a[mask] for a in flat]


def _auto_edges(values: np.ndarray, nbins: int, percentile_limits: tuple[float, float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot build bin edges from empty/non-finite data.")
    lo, hi = np.nanpercentile(values, percentile_limits)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = np.nanmin(values), np.nanmax(values)
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    return np.linspace(float(lo), float(hi), int(nbins) + 1)


def compute_jpdf(
    ds: xr.Dataset,
    *,
    x_var: str,
    y_var: str,
    x_edges: np.ndarray | None = None,
    y_edges: np.ndarray | None = None,
    nbins: int = 70,
    percentile_limits: tuple[float, float] = (1.0, 99.0),
    density: bool = True,
) -> xr.Dataset:
    """Compute a 2D joint PDF of two particle-sampled quantities."""
    x, y = _finite_flat(ds[x_var].values, ds[y_var].values)

    if x_edges is None:
        x_edges = _auto_edges(x, nbins, percentile_limits)
    if y_edges is None:
        y_edges = _auto_edges(y, nbins, percentile_limits)

    H, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], density=density)
    count, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], density=False)

    Hmax = np.nanmax(H) if np.any(np.isfinite(H)) else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        H_log = np.log10(H / Hmax)
    H_log[~np.isfinite(H_log)] = np.nan

    return xr.Dataset(
        data_vars={
            "pdf": (("xbin", "ybin"), H),
            "count": (("xbin", "ybin"), count),
            "log10_pdf_norm": (("xbin", "ybin"), H_log),
        },
        coords={
            "x_left": ("xbin", x_edges[:-1]),
            "x_right": ("xbin", x_edges[1:]),
            "y_left": ("ybin", y_edges[:-1]),
            "y_right": ("ybin", y_edges[1:]),
        },
        attrs={
            "description": "Joint PDF of particle-sampled variables.",
            "x_var": x_var,
            "y_var": y_var,
            "density": bool(density),
        },
    )


def compute_conditional_mean(
    ds: xr.Dataset,
    *,
    x_var: str,
    y_var: str,
    value_var: str,
    x_edges: np.ndarray | None = None,
    y_edges: np.ndarray | None = None,
    nbins: int = 70,
    percentile_limits: tuple[float, float] = (1.0, 99.0),
    min_count: int = 3,
) -> xr.Dataset:
    """
    Compute E[value_var | x_var, y_var].

    Useful examples:
        mean inverse_voronoi_area_norm | Ro, strain/f
        mean final_residence_time_days | div/f, strain/f
    """
    x, y, z = _finite_flat(ds[x_var].values, ds[y_var].values, ds[value_var].values)

    if x_edges is None:
        x_edges = _auto_edges(x, nbins, percentile_limits)
    if y_edges is None:
        y_edges = _auto_edges(y, nbins, percentile_limits)

    count, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    weighted, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=z)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = weighted / count
    mean[count < min_count] = np.nan

    return xr.Dataset(
        data_vars={
            "conditional_mean": (("xbin", "ybin"), mean),
            "count": (("xbin", "ybin"), count),
        },
        coords={
            "x_left": ("xbin", x_edges[:-1]),
            "x_right": ("xbin", x_edges[1:]),
            "y_left": ("ybin", y_edges[:-1]),
            "y_right": ("ybin", y_edges[1:]),
        },
        attrs={
            "description": "Conditional mean in particle-sampled flow-diagnostic space.",
            "x_var": x_var,
            "y_var": y_var,
            "value_var": value_var,
            "min_count": int(min_count),
        },
    )


def compute_conditional_probability(
    ds: xr.Dataset,
    *,
    x_var: str,
    y_var: str,
    flag_var: str = "cluster_flag",
    x_edges: np.ndarray | None = None,
    y_edges: np.ndarray | None = None,
    nbins: int = 70,
    percentile_limits: tuple[float, float] = (1.0, 99.0),
    min_count: int = 3,
) -> xr.Dataset:
    """Compute P(flag_var=True | x_var, y_var)."""
    x, y, f = _finite_flat(ds[x_var].values, ds[y_var].values, ds[flag_var].values)
    f = (f > 0).astype(float)

    if x_edges is None:
        x_edges = _auto_edges(x, nbins, percentile_limits)
    if y_edges is None:
        y_edges = _auto_edges(y, nbins, percentile_limits)

    count, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    flagged, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=f)

    with np.errstate(invalid="ignore", divide="ignore"):
        probability = flagged / count
    probability[count < min_count] = np.nan

    return xr.Dataset(
        data_vars={
            "conditional_probability": (("xbin", "ybin"), probability),
            "count": (("xbin", "ybin"), count),
        },
        coords={
            "x_left": ("xbin", x_edges[:-1]),
            "x_right": ("xbin", x_edges[1:]),
            "y_left": ("ybin", y_edges[:-1]),
            "y_right": ("ybin", y_edges[1:]),
        },
        attrs={
            "description": "Conditional probability in particle-sampled flow-diagnostic space.",
            "x_var": x_var,
            "y_var": y_var,
            "flag_var": flag_var,
            "min_count": int(min_count),
        },
    )


def compute_standard_coupled_products(
    ds_coupled: xr.Dataset,
    *,
    nbins: int = 70,
    min_count: int = 3,
) -> dict[str, xr.Dataset]:
    """
    Compute a useful default set of coupled diagnostics.

    Returned keys include only products for which required variables exist.
    """
    products: dict[str, xr.Dataset] = {}

    pairs = [
        ("ro_particle", "strain_f_particle", "ro_strain"),
        ("div_f_particle", "strain_f_particle", "div_strain"),
        ("ro_particle", "div_f_particle", "ro_div"),
    ]

    for x_var, y_var, tag in pairs:
        if x_var in ds_coupled and y_var in ds_coupled:
            products[f"jpdf_{tag}"] = compute_jpdf(ds_coupled, x_var=x_var, y_var=y_var, nbins=nbins)

            if "cluster_flag" in ds_coupled:
                products[f"p_cluster_{tag}"] = compute_conditional_probability(
                    ds_coupled,
                    x_var=x_var,
                    y_var=y_var,
                    flag_var="cluster_flag",
                    nbins=nbins,
                    min_count=min_count,
                )

            if "inverse_voronoi_area_norm" in ds_coupled:
                products[f"mean_inv_area_{tag}"] = compute_conditional_mean(
                    ds_coupled,
                    x_var=x_var,
                    y_var=y_var,
                    value_var="inverse_voronoi_area_norm",
                    nbins=nbins,
                    min_count=min_count,
                )

            if "final_residence_time_days" in ds_coupled:
                products[f"mean_residence_{tag}"] = compute_conditional_mean(
                    ds_coupled,
                    x_var=x_var,
                    y_var=y_var,
                    value_var="final_residence_time_days",
                    nbins=nbins,
                    min_count=min_count,
                )

    if "ro_particle" in ds_coupled and "log_voronoi_area_norm" in ds_coupled:
        products["jpdf_ro_log_area"] = compute_jpdf(
            ds_coupled,
            x_var="ro_particle",
            y_var="log_voronoi_area_norm",
            nbins=nbins,
        )

    if "div_f_particle" in ds_coupled and "log_voronoi_area_norm" in ds_coupled:
        products["jpdf_div_log_area"] = compute_jpdf(
            ds_coupled,
            x_var="div_f_particle",
            y_var="log_voronoi_area_norm",
            nbins=nbins,
        )

    return products


# ============================================================
# PLOTTING HELPERS
# ============================================================

def _edges_from_product(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    x_edges = np.concatenate([np.asarray(ds["x_left"].values), [float(ds["x_right"].values[-1])]])
    y_edges = np.concatenate([np.asarray(ds["y_left"].values), [float(ds["y_right"].values[-1])]])
    return x_edges, y_edges


def savefig_if_enabled(fig, path: str | Path | None, *, save: bool = True, dpi: int = 300) -> Path | None:
    if not save or path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white", transparent=False)
    return path


def plot_particles_on_flow_snapshot(
    ds_flow: xr.Dataset,
    ds_traj: xr.Dataset,
    *,
    obs_idx: int,
    flow_var: str = "ro",
    cluster_ds: xr.Dataset | None = None,
    residence_ds: xr.Dataset | None = None,
    run_case: str = "",
    particle_label: str = "particles",
    flow_time_name: str = "time",
    flow_x_name: str = "x",
    flow_y_name: str = "y",
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    percentile_limit: float = 99.0,
    clustered_only: bool = False,
    color_by: str | None = None,
    point_size: float = 8.0,
    point_alpha: float = 0.85,
    save_path: str | Path | None = None,
    show: bool = True,
):
    """
    Plot particle positions over one background flow diagnostic snapshot.

    Examples
    --------
    background Ro with all particles:
        plot_particles_on_flow_snapshot(ds_flow, ds_traj, obs_idx=55, flow_var="ro")

    background divergence with clustered particles only:
        plot_particles_on_flow_snapshot(..., flow_var="div_f", cluster_ds=cluster_ds,
                                        clustered_only=True)

    residence-time-colored points:
        plot_particles_on_flow_snapshot(..., residence_ds=residence_labels,
                                        color_by="final_residence_time_days")
    """
    if flow_var not in ds_flow:
        raise KeyError(f"{flow_var!r} not found in ds_flow.")

    xname, yname = get_xy_names(ds_traj)
    t_seconds = trajectory_time_values_seconds(ds_traj, [obs_idx])[0]
    t_days = t_seconds / 86400.0

    if flow_time_name in ds_flow.coords or flow_time_name in ds_flow.dims:
        flow_t = ds_flow.sel({flow_time_name: float(t_seconds)}, method="nearest")
    else:
        flow_t = ds_flow.isel({flow_time_name: int(obs_idx)})

    field = np.asarray(flow_t[flow_var].values, dtype=float)
    x = np.asarray(ds_flow[flow_x_name].values, dtype=float)
    y = np.asarray(ds_flow[flow_y_name].values, dtype=float)

    if vmin is None or vmax is None:
        valid = field[np.isfinite(field)]
        if valid.size > 0:
            lim = float(np.nanpercentile(np.abs(valid), percentile_limit))
        else:
            lim = 1.0
        if vmin is None:
            vmin = -lim
        if vmax is None:
            vmax = lim

    xp = np.asarray(ds_traj[xname].isel(obs=obs_idx).values, dtype=float)
    yp = np.asarray(ds_traj[yname].isel(obs=obs_idx).values, dtype=float)
    mask = np.isfinite(xp) & np.isfinite(yp)

    if clustered_only:
        if cluster_ds is None or "cluster_flag" not in cluster_ds:
            raise ValueError("clustered_only=True requires cluster_ds with cluster_flag.")
        cf = np.asarray(cluster_ds["cluster_flag"].sel(obs=obs_idx).values, dtype=bool)
        mask &= cf

    values = None
    if color_by is not None:
        source_ds = residence_ds if residence_ds is not None and color_by in residence_ds else cluster_ds
        if source_ds is None or color_by not in source_ds:
            raise KeyError(f"color_by={color_by!r} not found in residence_ds or cluster_ds.")
        values = np.asarray(source_ds[color_by].sel(obs=obs_idx).values, dtype=float)
        mask &= np.isfinite(values)

    fig, ax = plt.subplots(figsize=(7.0, 6.0), facecolor="white")
    ax.set_facecolor("white")

    pcm = ax.pcolormesh(
        x / 1000.0,
        y / 1000.0,
        field,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label(flow_var)

    if values is None:
        ax.scatter(
            xp[mask] / 1000.0,
            yp[mask] / 1000.0,
            s=point_size,
            c="k",
            alpha=point_alpha,
            linewidths=0.0,
            label=particle_label,
        )
    else:
        sc = ax.scatter(
            xp[mask] / 1000.0,
            yp[mask] / 1000.0,
            s=point_size,
            c=values[mask],
            cmap="viridis",
            alpha=point_alpha,
            linewidths=0.0,
            label=particle_label,
        )
        cbar2 = fig.colorbar(sc, ax=ax, pad=0.08, fraction=0.046)
        cbar2.set_label(color_by)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.grid(True, alpha=0.25)
    ax.set_title(f"T={t_days:.2f} d | {run_case}" if run_case else f"T={t_days:.2f} d")
    ax.legend(loc="best")
    fig.tight_layout()

    savefig_if_enabled(fig, save_path, save=save_path is not None)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def plot_jpdf(
    jpdf_ds: xr.Dataset,
    *,
    field: str = "log10_pdf_norm",
    title: str = "",
    xlabel: str | None = None,
    ylabel: str | None = None,
    cbar_label: str | None = None,
    cmap: str = "turbo",
    vmin: float | None = -3.0,
    vmax: float | None = 0.0,
    diagonal_abs: bool = False,
    zero_lines: bool = True,
    save_path: str | Path | None = None,
    show: bool = True,
):
    """Plot a JPDF product returned by compute_jpdf()."""
    x_edges, y_edges = _edges_from_product(jpdf_ds)
    Z = np.asarray(jpdf_ds[field].values, dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 5.4), facecolor="white")
    ax.set_facecolor("white")
    pcm = ax.pcolormesh(x_edges, y_edges, Z.T, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label(cbar_label or field)

    if zero_lines:
        ax.axvline(0.0, color="0.2", ls="--", lw=0.8, alpha=0.7)
        ax.axhline(0.0, color="0.2", ls="--", lw=0.8, alpha=0.7)
    if diagonal_abs:
        xx = np.linspace(x_edges[0], x_edges[-1], 500)
        ax.plot(xx, np.abs(xx), color="0.2", ls="--", lw=1.0, alpha=0.7)

    ax.set_xlabel(xlabel or jpdf_ds.attrs.get("x_var", "x"))
    ax.set_ylabel(ylabel or jpdf_ds.attrs.get("y_var", "y"))
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    savefig_if_enabled(fig, save_path, save=save_path is not None)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def plot_conditional_field(
    cond_ds: xr.Dataset,
    *,
    field: str | None = None,
    title: str = "",
    xlabel: str | None = None,
    ylabel: str | None = None,
    cbar_label: str | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    percentile_limit: float = 98.0,
    diagonal_abs: bool = False,
    zero_lines: bool = True,
    save_path: str | Path | None = None,
    show: bool = True,
):
    """Plot a conditional probability or conditional mean product."""
    if field is None:
        if "conditional_probability" in cond_ds:
            field = "conditional_probability"
        elif "conditional_mean" in cond_ds:
            field = "conditional_mean"
        else:
            raise KeyError("Could not infer field. Provide field explicitly.")

    x_edges, y_edges = _edges_from_product(cond_ds)
    Z = np.asarray(cond_ds[field].values, dtype=float)

    if vmin is None or vmax is None:
        valid = Z[np.isfinite(Z)]
        if valid.size > 0:
            if field == "conditional_probability":
                if vmin is None:
                    vmin = 0.0
                if vmax is None:
                    vmax = 1.0
            else:
                if vmin is None:
                    vmin = float(np.nanpercentile(valid, 100.0 - percentile_limit))
                if vmax is None:
                    vmax = float(np.nanpercentile(valid, percentile_limit))

    fig, ax = plt.subplots(figsize=(6.2, 5.4), facecolor="white")
    ax.set_facecolor("white")
    pcm = ax.pcolormesh(x_edges, y_edges, Z.T, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label(cbar_label or field)

    if zero_lines:
        ax.axvline(0.0, color="0.2", ls="--", lw=0.8, alpha=0.7)
        ax.axhline(0.0, color="0.2", ls="--", lw=0.8, alpha=0.7)
    if diagonal_abs:
        xx = np.linspace(x_edges[0], x_edges[-1], 500)
        ax.plot(xx, np.abs(xx), color="0.2", ls="--", lw=1.0, alpha=0.7)

    ax.set_xlabel(xlabel or cond_ds.attrs.get("x_var", "x"))
    ax.set_ylabel(ylabel or cond_ds.attrs.get("y_var", "y"))
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    savefig_if_enabled(fig, save_path, save=save_path is not None)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax
