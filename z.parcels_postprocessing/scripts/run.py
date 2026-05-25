from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import xarray as xr
from parcels import ParticleSet, AdvectionRK4

from .fieldset import build_fieldset
from .particles import SurfaceParticle, DepthParticle
from .kernels import age_particle, periodic_xy
from .io_utils import ensure_dir, open_trajectory_dataset


@dataclass
class RunConfig:
    input_nc: str
    output_path: str

    # Simulation timing
    runtime_days: float = 14.0
    dt_seconds: int = 300
    outputdt_seconds: int = 10800          # 3 hours
    time_step_seconds: int = 10800         # MITgcm velocity output frequency
    release_time_index: int = 0            # change this for T=0, T=10, ...

    # Field / particle setup
    surface_only: bool = True
    mesh: str = "flat"
    periodic: bool = True
    level_indices: tuple[int, ...] = (0,)

    # Release setup
    release_mode: Literal["grid", "random"] = "grid"
    nx: int = 50
    ny: int = 50
    n_particles: int = 2500
    seed: int = 42

    # Optional release bounds
    xmin: Optional[float] = None
    xmax: Optional[float] = None
    ymin: Optional[float] = None
    ymax: Optional[float] = None

    # Keep particles away from exact boundaries
    release_margin_cells: float = 1.0

    # Only used for true 3D tracking
    depth_value: Optional[float] = None


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
    last_time = float(np.asarray(fieldset.U.grid.time, dtype=float)[-1])
    requested_end = release_time + runtime_seconds

    if requested_end > last_time:
        available_days = (last_time - release_time) / 86400.0
        requested_days = runtime_seconds / 86400.0
        raise ValueError(
            f"Requested runtime is too long for the available velocity forcing.\n"
            f"Requested: {requested_days:.2f} days after release.\n"
            f"Available: {available_days:.2f} days after release.\n"
            f"Reduce runtime_days or choose an earlier release_time_index."
        )


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

    output_path = Path(config.output_path)
    ensure_dir(output_path.parent)

    particle_time = np.full(lon0.shape, release_time, dtype=float)

    if meta.is_3d and not config.surface_only:
        pclass = DepthParticle
        depth0 = np.full_like(
            lon0,
            float(ds["depth"].values[0] if config.depth_value is None else config.depth_value),
            dtype=float,
        )
        pset = ParticleSet.from_list(
            fieldset=fieldset,
            pclass=pclass,
            lon=lon0,
            lat=lat0,
            depth=depth0,
            time=particle_time,
        )
    else:
        pclass = SurfaceParticle
        pset = ParticleSet.from_list(
            fieldset=fieldset,
            pclass=pclass,
            lon=lon0,
            lat=lat0,
            time=particle_time,
        )

    kernel = AdvectionRK4 + pset.Kernel(age_particle)

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
        "n_particles": int(len(lon0)),
        "release_time_index": int(config.release_time_index),
        "release_time_seconds": float(release_time),
        "release_time_days": float(release_time / 86400.0),
        "runtime_days": float(config.runtime_days),
        "dt_seconds": int(config.dt_seconds),
        "outputdt_seconds": int(config.outputdt_seconds),
        "surface_only": bool(config.surface_only),
        "periodic": bool(config.periodic),
        "is_3d_input": bool(meta.is_3d),
        "u_name": meta.u_name,
        "v_name": meta.v_name,
    }


def load_trajectories(path: str | Path) -> xr.Dataset:
    return open_trajectory_dataset(path)