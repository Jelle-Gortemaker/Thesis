from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Literal, Optional
import json

import numpy as np
import xarray as xr
from parcels import ParticleSet

from .fieldset import build_fieldset
from .particles import FloatingParticle
from .kernels_common import periodic_xy
from .kernels_passive import advection_passive_rk4
from .kernels_mr_sm import advection_mr_sm_rk4
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

    # Particle class for this run.
    particle_class: Literal["passive", "mr_sm"] = "passive"

    # Optional human-readable identifiers. If omitted, they are generated from tau_p.
    particle_tag: Optional[str] = None
    particle_label: Optional[str] = None

    # Tau-only MR-SM control parameter.
    # Passive tracer: tau_p_seconds = 0.0
    # MR-SM particle: tau_p_seconds > 0.0
    tau_p_seconds: float = 0.0

    # f-plane Coriolis parameter [s-1]. Used only for MR-SM.
    f0: float = 0.0

    # Used only for labelling / storing St = tau_p / flow_timescale_seconds.
    flow_timescale_seconds: Optional[float] = None

    # Kept for future extension; current implementation initializes all classes
    # with the local fluid velocity.
    initial_particle_velocity: Literal["fluid"] = "fluid"

    # Always write sidecar metadata next to the trajectory output.
    save_metadata_sidecar: bool = True


def _safe_number(x: float, precision: int = 4) -> str:
    """File-name safe numeric formatting."""
    if x is None:
        return "none"
    if not np.isfinite(float(x)):
        return "nan"
    s = f"{float(x):.{precision}g}"
    return s.replace("-", "m").replace("+", "").replace(".", "p")


def _format_tau_label(tau_seconds: float) -> str:
    """Compact label for a relaxation time in seconds."""
    tau_seconds = float(tau_seconds)

    if tau_seconds >= 86400.0 and np.isclose(tau_seconds % 86400.0, 0.0):
        return f"{tau_seconds / 86400.0:g} d"
    if tau_seconds >= 3600.0 and np.isclose(tau_seconds % 3600.0, 0.0):
        return f"{tau_seconds / 3600.0:g} h"
    if tau_seconds >= 60.0 and np.isclose(tau_seconds % 60.0, 0.0):
        return f"{tau_seconds / 60.0:g} min"
    return f"{tau_seconds:g} s"


def make_particle_tag(
    *,
    particle_class: str,
    tau_p_seconds: float = 0.0,
    stokes_number: float = np.nan,
) -> str:
    """Create a compact file-safe particle class tag."""
    if particle_class == "passive":
        return "passive"

    if particle_class != "mr_sm":
        raise ValueError(f"Unknown particle_class: {particle_class}")

    return (
        "mrsm_"
        f"tau{_safe_number(tau_p_seconds)}s_"
        f"St{_safe_number(stokes_number)}"
    )


def make_particle_label(
    *,
    particle_class: str,
    tau_p_seconds: float = 0.0,
    stokes_number: float = np.nan,
) -> str:
    """Create a compact display label for plots and legends."""
    if particle_class == "passive":
        return "Tracer"

    if particle_class != "mr_sm":
        raise ValueError(f"Unknown particle_class: {particle_class}")

    tau_txt = _format_tau_label(tau_p_seconds)

    if np.isfinite(stokes_number):
        return f"tau={tau_txt}, St={stokes_number:.3g}"

    return f"tau={tau_txt}"


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


def particle_class_parameters(config: RunConfig) -> dict:
    """Return numeric parameters and labels for a single run config."""
    if config.particle_class == "passive":
        return {
            "particle_class_id": 0,
            "drag_mode_id": 0,
            "tau_p_seconds": 0.0,
            "tau_s_seconds": 0.0,
            "tau_eff_nominal_seconds": 0.0,
            "C_Rep": 1.0,
            "stokes_number": 0.0,
            "particle_tag": config.particle_tag or "passive",
            "particle_label": config.particle_label or "Tracer",
        }

    if config.particle_class != "mr_sm":
        raise ValueError(f"Unknown particle_class: {config.particle_class}")

    tau_p = float(config.tau_p_seconds)
    if tau_p <= 0.0:
        raise ValueError("tau_p_seconds must be positive for mr_sm particles.")

    if config.flow_timescale_seconds is None:
        stokes_number = np.nan
    else:
        tref = float(config.flow_timescale_seconds)
        if tref <= 0.0:
            raise ValueError("flow_timescale_seconds must be positive.")
        stokes_number = tau_p / tref

    tag = config.particle_tag or make_particle_tag(
        particle_class=config.particle_class,
        tau_p_seconds=tau_p,
        stokes_number=stokes_number,
    )

    label = config.particle_label or make_particle_label(
        particle_class=config.particle_class,
        tau_p_seconds=tau_p,
        stokes_number=stokes_number,
    )

    return {
        "particle_class_id": 1,
        "drag_mode_id": 0,
        "tau_p_seconds": float(tau_p),
        "tau_s_seconds": float(tau_p),
        "tau_eff_nominal_seconds": float(tau_p),
        "C_Rep": 1.0,
        "stokes_number": float(stokes_number),
        "particle_tag": tag,
        "particle_label": label,
    }


def _metadata_path_for_output(output_path: Path) -> Path:
    if output_path.suffix == ".zarr":
        return output_path.with_suffix(".json")
    return output_path.parent / f"{output_path.name}.json"


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_metadata_sidecar(output_path: Path, info: dict) -> Path:
    metadata_path = _metadata_path_for_output(output_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    clean = {key: _json_ready(value) for key, value in info.items()}

    with open(metadata_path, "w") as f:
        json.dump(clean, f, indent=2)

    return metadata_path


def _add_attrs_to_zarr_if_possible(output_path: Path, info: dict) -> None:
    """Best-effort storage of labels/metadata in the trajectory dataset attrs."""
    try:
        import zarr

        if output_path.suffix != ".zarr" or not output_path.exists():
            return

        root = zarr.open_group(str(output_path), mode="a")
        for key, value in info.items():
            value = _json_ready(value)
            if isinstance(value, (str, int, float, bool)) or value is None:
                root.attrs[key] = value
    except Exception:
        # The JSON sidecar is the authoritative metadata store.
        return


def run_parcels_experiment(config: RunConfig) -> dict:
    add_derivatives = config.particle_class == "mr_sm"

    fieldset, meta, ds = build_fieldset(
        config.input_nc,
        surface_only=config.surface_only,
        mesh=config.mesh,
        time_step_seconds=config.time_step_seconds,
        level_indices=config.level_indices,
        periodic=config.periodic,
        add_derivatives=add_derivatives,
    )

    params = particle_class_parameters(config)

    lon0, lat0 = prepare_release(config, ds)

    release_time = _get_release_time(fieldset, config.release_time_index)
    runtime_seconds = float(config.runtime_days) * 86400.0
    _check_runtime_available(fieldset, release_time, runtime_seconds)

    # Initialize both passive and MR-SM particles with the local fluid velocity.
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
        particle_class_id=np.full(n, int(params["particle_class_id"]), dtype=np.int32),
        drag_mode_id=np.full(n, int(params["drag_mode_id"]), dtype=np.int32),
        B=np.ones(n, dtype=np.float32),
        diameter=np.zeros(n, dtype=np.float32),
        tau_p=np.full(n, float(params["tau_p_seconds"]), dtype=np.float32),
        tau_eff_nominal=np.full(n, float(params["tau_eff_nominal_seconds"]), dtype=np.float32),
        C_Rep=np.ones(n, dtype=np.float32),
        stokes_number=np.full(n, float(params["stokes_number"]), dtype=np.float32),
        Rep=np.zeros(n, dtype=np.float32),
        uslip=np.zeros(n, dtype=np.float32),
        vslip=np.zeros(n, dtype=np.float32),
        C_Rep_current=np.ones(n, dtype=np.float32),
        up=up0.astype(np.float32),
        vp=vp0.astype(np.float32),
    )

    if config.particle_class == "passive":
        kernel = pset.Kernel(advection_passive_rk4)
    elif config.particle_class == "mr_sm":
        kernel = pset.Kernel(advection_mr_sm_rk4)
    else:
        raise ValueError(f"Unknown particle_class: {config.particle_class}")

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

    tau_p_for_metadata = float(params["tau_p_seconds"])

    info = {
        **params,
        "input_nc": str(config.input_nc),
        "output_path": str(output_path),
        "n_particles": int(n),
        "release_time_index": int(config.release_time_index),
        "release_time_seconds": float(release_time),
        "release_time_days": float(release_time / 86400.0),
        "runtime_days": float(config.runtime_days),
        "dt_seconds": int(config.dt_seconds),
        "outputdt_seconds": int(config.outputdt_seconds),
        "time_step_seconds": int(config.time_step_seconds),
        "surface_only": bool(config.surface_only),
        "periodic": bool(config.periodic),
        "level_indices": tuple(config.level_indices),
        "particle_class": config.particle_class,
        "tau_p_seconds": tau_p_for_metadata,
        "f0": float(config.f0),
        "flow_timescale_seconds": (
            None if config.flow_timescale_seconds is None
            else float(config.flow_timescale_seconds)
        ),
        "initial_particle_velocity": config.initial_particle_velocity,
        "u_name": getattr(meta, "u_name", "UVEL"),
        "v_name": getattr(meta, "v_name", "VVEL"),
        "add_derivatives": bool(add_derivatives),
        "config": asdict(config),
    }

    if config.save_metadata_sidecar:
        metadata_path = _write_metadata_sidecar(output_path, info)
        info["metadata_path"] = str(metadata_path)

    _add_attrs_to_zarr_if_possible(output_path, info)

    return info


def load_trajectories(path: str | Path) -> xr.Dataset:
    return open_trajectory_dataset(path)


def load_run_metadata(path: str | Path) -> dict:
    path = Path(path)
    with open(path, "r") as f:
        return json.load(f)
