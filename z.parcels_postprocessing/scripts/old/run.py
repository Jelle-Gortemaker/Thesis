from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import xarray as xr
from parcels import ParticleSet

from .fieldset import build_fieldset
from .particles import FloatingParticle
from .kernels import advection_passive_or_inertial, periodic_xy
from .io_utils import ensure_dir, open_trajectory_dataset


@dataclass
class RunConfig:
    input_nc: str
    output_path: str

    runtime_days: float = 14.0
    dt_seconds: int = 300
    outputdt_seconds: int = 10800

    time_step_seconds: int = 10800
    release_time_index: int = 0

    surface_only: bool = True
    mesh: str = "flat"
    periodic: bool = True
    level_indices: tuple[int, ...] = (0,)

    release_mode: Literal["grid", "random"] = "grid"
    nx: int = 50
    ny: int = 50
    n_particles: int = 2500
    seed: int = 42

    xmin: Optional[float] = None
    xmax: Optional[float] = None
    ymin: Optional[float] = None
    ymax: Optional[float] = None

    release_margin_cells: float = 1.0

    # Particle class for this run
    particle_class: Literal["passive", "inertial"] = "passive"
    tau_p_seconds: float = 0.0
    flow_timescale_seconds: Optional[float] = None
    initial_particle_velocity: Literal["fluid", "zero"] = "fluid"


def _grid_spacing_1d(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    a = np.sort(np.unique(a[np.isfinite(a)]))
    if a.size < 2:
        return 0.0
    return float(np.nanmedian(np.diff(a)))


def _make_release_points_grid(xmin, xmax, ymin, ymax, nx, ny):
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys)
    return xx.ravel(), yy.ravel()


def _make_release_points_random(xmin, xmax, ymin, ymax, n_particles, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(xmin, xmax, n_particles)
    y = rng.uniform(ymin, ymax, n_particles)
    return x, y


def prepare_release(config: RunConfig, ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(ds["x"].values, dtype=float)
    y = np.asarray(ds["y"].values, dtype=float)

    dx = _grid_spacing_1d(x)
    dy = _grid_spacing_1d(y)

    xmin_default = float(np.nanmin(x) + config.release_margin_cells * dx)
    xmax_default = float(np.nanmax(x) - config.release_margin_cells * dx)
    ymin_default = float(np.nanmin(y) + config.release_margin_cells * dy)
    ymax_default = float(np.nanmax(y) - config.release_margin_cells * dy)

    xmin = xmin_default if config.xmin is None else float(config.xmin)
    xmax = xmax_default if config.xmax is None else float(config.xmax)
    ymin = ymin_default if config.ymin is None else float(config.ymin)
    ymax = ymax_default if config.ymax is None else float(config.ymax)

    if xmin >= xmax or ymin >= ymax:
        raise ValueError(
            "Invalid release bounds after applying margin. "
            "Reduce release_margin_cells or set xmin/xmax/ymin/ymax manually."
        )

    if config.release_mode == "grid":
        return _make_release_points_grid(xmin, xmax, ymin, ymax, config.nx, config.ny)

    if config.release_mode == "random":
        return _make_release_points_random(
            xmin, xmax, ymin, ymax, config.n_particles, config.seed
        )

    raise ValueError(f"Unknown release_mode: {config.release_mode}")


def _get_release_time(fieldset, release_time_index: int) -> float:
    times = np.asarray(fieldset.U.grid.time, dtype=float)

    if release_time_index < 0 or release_time_index >= len(times):
        raise IndexError(
            f"release_time_index={release_time_index} is out of range "
            f"for {len(times)} available forcing times."
        )

    return float(times[release_time_index])


def _check_runtime_available(fieldset, release_time: float, runtime_seconds: float) -> None:
    times = np.asarray(fieldset.U.grid.time, dtype=float)
    last_time = float(times[-1])
    requested_end = release_time + runtime_seconds

    if requested_end > last_time:
        available_days = (last_time - release_time) / 86400.0
        requested_days = runtime_seconds / 86400.0
        raise ValueError(
            "Requested runtime is too long for the available velocity forcing.\n"
            f"Requested: {requested_days:.2f} days after release.\n"
            f"Available: {available_days:.2f} days after release.\n"
            "Reduce runtime_days or choose an earlier release_time_index."
        )


def _sample_initial_fluid_velocity(
    ds: xr.Dataset,
    lon0: np.ndarray,
    lat0: np.ndarray,
    release_time_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    points = xr.DataArray(np.arange(len(lon0)), dims="particle")
    lon_da = xr.DataArray(lon0, dims="particle", coords={"particle": points})
    lat_da = xr.DataArray(lat0, dims="particle", coords={"particle": points})

    U0 = ds["U"].isel(time=release_time_index).interp(x=lon_da, y=lat_da).values
    V0 = ds["V"].isel(time=release_time_index).interp(x=lon_da, y=lat_da).values

    U0 = np.asarray(U0, dtype=float)
    V0 = np.asarray(V0, dtype=float)

    U0 = np.where(np.isfinite(U0), U0, 0.0)
    V0 = np.where(np.isfinite(V0), V0, 0.0)

    return U0, V0


def _particle_class_parameters(config: RunConfig) -> tuple[int, float, float, str]:
    if config.particle_class == "passive":
        return 0, 0.0, 0.0, "passive"

    if config.particle_class != "inertial":
        raise ValueError(f"Unknown particle_class: {config.particle_class}")

    tau_p = float(config.tau_p_seconds)
    if tau_p <= 0.0:
        raise ValueError(f"Inertial particles require tau_p_seconds > 0, got {tau_p}")

    if config.flow_timescale_seconds is None:
        stokes_number = np.nan
    else:
        tref = float(config.flow_timescale_seconds)
        if tref <= 0.0:
            raise ValueError(f"flow_timescale_seconds must be > 0, got {tref}")
        stokes_number = tau_p / tref

    label = f"inertial_tau{tau_p:g}s_St{stokes_number:g}"
    return 1, tau_p, float(stokes_number), label


def run_parcels_experiment(config: RunConfig) -> dict:
    fieldset, meta, ds = build_fieldset(
        config.input_nc,
        surface_only=config.surface_only,
        mesh=config.mesh,
        time_step_seconds=config.time_step_seconds,
        level_indices=config.level_indices,
        periodic=config.periodic,
    )

    lon0, lat0 = prepare_release(config, ds)

    release_time = _get_release_time(fieldset, config.release_time_index)
    runtime_seconds = float(config.runtime_days) * 86400.0
    _check_runtime_available(fieldset, release_time, runtime_seconds)

    particle_class_id, tau_p, stokes_number, particle_label = _particle_class_parameters(config)

    if config.particle_class == "inertial" and config.initial_particle_velocity == "zero":
        up0 = np.zeros_like(lon0, dtype=float)
        vp0 = np.zeros_like(lat0, dtype=float)
    else:
        up0, vp0 = _sample_initial_fluid_velocity(
            ds,
            lon0,
            lat0,
            config.release_time_index,
        )

    n = len(lon0)
    release_id = np.arange(n, dtype=np.int32)

    output_path = Path(config.output_path)
    ensure_dir(output_path.parent)

    pset = ParticleSet.from_list(
        fieldset=fieldset,
        pclass=FloatingParticle,
        lon=lon0,
        lat=lat0,
        time=np.full(n, release_time, dtype=float),
        release_id=release_id,
        particle_class_id=np.full(n, particle_class_id, dtype=np.int32),
        tau_p=np.full(n, tau_p, dtype=np.float32),
        stokes_number=np.full(n, stokes_number, dtype=np.float32),
        up=up0.astype(np.float32),
        vp=vp0.astype(np.float32),
    )

    kernel = pset.Kernel(advection_passive_or_inertial)

    if config.periodic:
        kernel += pset.Kernel(periodic_xy)

    pfile = pset.ParticleFile(
        name=str(output_path),
        outputdt=timedelta(seconds=config.outputdt_seconds),
    )

    pset.execute(
        kernel,
        runtime=timedelta(days=config.runtime_days),
        dt=timedelta(seconds=config.dt_seconds),
        output_file=pfile,
    )

    return {
        "input_nc": str(config.input_nc),
        "output_path": str(output_path),
        "n_particles": int(n),
        "release_time_index": int(config.release_time_index),
        "release_time_seconds": float(release_time),
        "release_time_days": float(release_time / 86400.0),
        "runtime_days": float(config.runtime_days),
        "dt_seconds": int(config.dt_seconds),
        "outputdt_seconds": int(config.outputdt_seconds),
        "surface_only": bool(config.surface_only),
        "periodic": bool(config.periodic),
        "level_indices": tuple(config.level_indices),
        "particle_class": config.particle_class,
        "particle_class_id": int(particle_class_id),
        "particle_label": particle_label,
        "tau_p_seconds": float(tau_p),
        "stokes_number": float(stokes_number),
        "flow_timescale_seconds": (
            None if config.flow_timescale_seconds is None
            else float(config.flow_timescale_seconds)
        ),
        "initial_particle_velocity": config.initial_particle_velocity,
        "u_name": getattr(meta, "u_name", "UVEL"),
        "v_name": getattr(meta, "v_name", "VVEL"),
    }


def load_trajectories(path: str | Path) -> xr.Dataset:
    return open_trajectory_dataset(path)