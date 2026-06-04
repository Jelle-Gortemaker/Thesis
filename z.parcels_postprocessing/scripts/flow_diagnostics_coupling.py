from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr

try:
    from .fieldset import mitgcm_cgrid_to_parcels_dataset
except Exception:  # allows running as a standalone copied script
    from fieldset import mitgcm_cgrid_to_parcels_dataset


# ============================================================
# FLOW DIAGNOSTICS FOR PARTICLE COUPLING
# ============================================================

def _differentiate(da: xr.DataArray, coord: str) -> xr.DataArray:
    """Small wrapper around xarray differentiation with safe attrs."""
    out = da.differentiate(coord, edge_order=2)
    out.attrs = dict(da.attrs)
    return out


def compute_flow_diagnostics_from_parcels_uv(
    ds_uv: xr.Dataset,
    *,
    f0: float,
    include_velocity_derivatives: bool = True,
) -> xr.Dataset:
    """
    Compute flow diagnostics on the same x/y/time grid used by Parcels.

    Input dataset must contain:
        U(time, y, x) [m/s]
        V(time, y, x) [m/s]
        time [s], x [m], y [m]

    Output variables:
        speed      [m/s]
        zeta       [1/s]
        div        [1/s]
        strain     [1/s]
        ro         [-] = zeta/f0
        div_f      [-] = div/f0
        strain_f   [-] = strain/f0
        ow         [1/s^2] = strain^2 - zeta^2
        ow_f2      [-] = (strain/f0)^2 - (zeta/f0)^2

    Optionally also stores dUdt, dVdt, dUdx, dUdy, dVdx, dVdy, which are useful
    for checking the slow-manifold MR forcing and for future diagnostics.
    """
    if "U" not in ds_uv or "V" not in ds_uv:
        raise KeyError("ds_uv must contain U and V.")
    if f0 == 0 or not np.isfinite(float(f0)):
        raise ValueError("f0 must be a finite non-zero Coriolis parameter.")

    U = ds_uv["U"].astype(float)
    V = ds_uv["V"].astype(float)

    dUdx = _differentiate(U, "x")
    dUdy = _differentiate(U, "y")
    dVdx = _differentiate(V, "x")
    dVdy = _differentiate(V, "y")

    zeta = dVdx - dUdy
    div = dUdx + dVdy
    strain = np.sqrt((dUdx - dVdy) ** 2 + (dVdx + dUdy) ** 2)
    speed = np.sqrt(U ** 2 + V ** 2)

    ro = zeta / float(f0)
    div_f = div / float(f0)
    strain_f = strain / float(f0)
    ow = strain ** 2 - zeta ** 2
    ow_f2 = strain_f ** 2 - ro ** 2

    ds_out = xr.Dataset(
        data_vars={
            "speed": speed.astype(np.float32),
            "zeta": zeta.astype(np.float32),
            "div": div.astype(np.float32),
            "strain": strain.astype(np.float32),
            "ro": ro.astype(np.float32),
            "div_f": div_f.astype(np.float32),
            "strain_f": strain_f.astype(np.float32),
            "ow": ow.astype(np.float32),
            "ow_f2": ow_f2.astype(np.float32),
        },
        coords={"time": ds_uv["time"], "y": ds_uv["y"], "x": ds_uv["x"]},
        attrs={
            "description": "Flow diagnostics on the Parcels x/y/time grid for particle-flow coupling.",
            "f0_s-1": float(f0),
            "ow_definition": "ow = strain^2 - zeta^2; positive values are strain-dominated under this convention",
            "time_units": "seconds relative to Parcels release-time convention",
            "x_y_units": "m",
        },
    )

    for name, units, long_name in [
        ("speed", "m s-1", "horizontal speed"),
        ("zeta", "s-1", "relative vorticity"),
        ("div", "s-1", "horizontal divergence"),
        ("strain", "s-1", "horizontal strain magnitude"),
        ("ro", "1", "Rossby number zeta/f0"),
        ("div_f", "1", "horizontal divergence normalized by f0"),
        ("strain_f", "1", "strain magnitude normalized by f0"),
        ("ow", "s-2", "Okubo-Weiss strain^2 - vorticity^2"),
        ("ow_f2", "1", "Okubo-Weiss normalized by f0^2"),
    ]:
        ds_out[name].attrs.update({"units": units, "long_name": long_name})

    if include_velocity_derivatives:
        dUdt = _differentiate(U, "time")
        dVdt = _differentiate(V, "time")
        ds_out["dUdt"] = dUdt.astype(np.float32)
        ds_out["dVdt"] = dVdt.astype(np.float32)
        ds_out["dUdx"] = dUdx.astype(np.float32)
        ds_out["dUdy"] = dUdy.astype(np.float32)
        ds_out["dVdx"] = dVdx.astype(np.float32)
        ds_out["dVdy"] = dVdy.astype(np.float32)

        for name in ["dUdt", "dVdt"]:
            ds_out[name].attrs.update({"units": "m s-2"})
        for name in ["dUdx", "dUdy", "dVdx", "dVdy"]:
            ds_out[name].attrs.update({"units": "s-1"})

    return ds_out


def compute_flow_diagnostics_from_mitgcm(
    path: str | Path,
    *,
    f0: float,
    level_indices: Sequence[int] = (0,),
    time_step_seconds: float = 10800.0,
    use_dataset_time: bool = False,
    chunks: dict | None = None,
    include_velocity_derivatives: bool = True,
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Convert MITgcm UVEL/VVEL to the Parcels grid and compute diagnostics.

    Returns:
        ds_flow : flow diagnostics dataset for coupling
        ds_uv   : intermediate Parcels-ready U/V dataset
    """
    ds_uv, _meta = mitgcm_cgrid_to_parcels_dataset(
        path,
        level_indices=level_indices,
        time_step_seconds=time_step_seconds,
        use_dataset_time=use_dataset_time,
        chunks=chunks,
    )
    ds_flow = compute_flow_diagnostics_from_parcels_uv(
        ds_uv,
        f0=f0,
        include_velocity_derivatives=include_velocity_derivatives,
    )
    ds_flow.attrs.update(
        {
            "source_mitgcm_file": str(path),
            "level_indices": str(list(level_indices)),
            "time_step_seconds": float(time_step_seconds),
            "use_dataset_time": bool(use_dataset_time),
        }
    )
    return ds_flow, ds_uv


def save_flow_diagnostics(
    ds_flow: xr.Dataset,
    path: str | Path,
    *,
    overwrite: bool = True,
) -> Path:
    """Save a flow diagnostics dataset as .zarr or .nc."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".zarr":
        mode = "w" if overwrite else "w-"
        ds_flow.to_zarr(path, mode=mode)
    else:
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        ds_flow.to_netcdf(path)

    return path


def open_flow_diagnostics(path: str | Path) -> xr.Dataset:
    """Open a saved flow diagnostics dataset."""
    path = Path(path)
    if path.suffix == ".zarr":
        return xr.open_zarr(path)
    return xr.open_dataset(path)
