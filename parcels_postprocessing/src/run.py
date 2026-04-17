from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from parcels import ParticleSet, AdvectionRK4

from .fieldset import build_fieldset
from .particles import SurfaceParticle, DepthParticle
from .kernels import age_particle
from .io_utils import ensure_dir


@dataclass
class RunConfig:
    input_nc: str
    output_path: str
    runtime_days: float = 14.0
    dt_seconds: int = 300
    outputdt_seconds: int = 3600
    surface_only: bool = True
    mesh: str = "flat"
    release_mode: Literal["grid", "random"] = "grid"
    nx: int = 50
    ny: int = 50
    n_particles: int = 2500
    seed: int = 42
    xmin: Optional[float] = None
    xmax: Optional[float] = None
    ymin: Optional[float] = None
    ymax: Optional[float] = None
    depth_value: Optional[float] = None


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


def run_parcels_experiment(config: RunConfig) -> dict:
    fieldset, meta, ds = build_fieldset(
        config.input_nc,
        surface_only=config.surface_only,
        mesh=config.mesh,
    )

    x = ds["x"].values
    y = ds["y"].values

    xmin = float(np.min(x)) if config.xmin is None else float(config.xmin)
    xmax = float(np.max(x)) if config.xmax is None else float(config.xmax)
    ymin = float(np.min(y)) if config.ymin is None else float(config.ymin)
    ymax = float(np.max(y)) if config.ymax is None else float(config.ymax)

    if config.release_mode == "grid":
        lon0, lat0 = _make_release_points_grid(xmin, xmax, ymin, ymax, config.nx, config.ny)
    elif config.release_mode == "random":
        lon0, lat0 = _make_release_points_random(
            xmin, xmax, ymin, ymax, config.n_particles, config.seed
        )
    else:
        raise ValueError(f"Unknown release_mode: {config.release_mode}")

    output_path = Path(config.output_path)
    ensure_dir(output_path.parent)

    if meta.is_3d and not config.surface_only:
        pclass = DepthParticle
        if config.depth_value is None:
            depth0 = np.full_like(lon0, float(ds["depth"].values[0]), dtype=float)
        else:
            depth0 = np.full_like(lon0, float(config.depth_value), dtype=float)

        pset = ParticleSet(
            fieldset=fieldset,
            pclass=pclass,
            lon=lon0,
            lat=lat0,
            depth=depth0,
        )
    else:
        pclass = SurfaceParticle
        pset = ParticleSet(
            fieldset=fieldset,
            pclass=pclass,
            lon=lon0,
            lat=lat0,
        )

    kernel = AdvectionRK4 + pset.Kernel(age_particle)

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
        "n_particles": len(lon0),
        "surface_only": config.surface_only,
        "is_3d_input": meta.is_3d,
        "u_name": meta.u_name,
        "v_name": meta.v_name,
        "x_name": meta.x_name,
        "y_name": meta.y_name,
        "time_name": meta.time_name,
        "depth_name": meta.depth_name,
        "release_mode": config.release_mode,
        "domain": {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax},
    }