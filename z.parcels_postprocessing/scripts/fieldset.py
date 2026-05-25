from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import xarray as xr
from parcels import FieldSet


@dataclass
class FieldMeta:
    path: Path
    x_name: str = "X"
    y_name: str = "Y"
    xp1_name: str = "Xp1"
    yp1_name: str = "Yp1"
    time_name: str = "T"
    z_name: str = "Zmd000001"
    u_name: str = "UVEL"
    v_name: str = "VVEL"
    mesh: str = "flat"


def _find_z_dim(ds: xr.Dataset) -> str:
    """Find the vertical dimension used by UVEL/VVEL."""
    for dim in ds["UVEL"].dims:
        if dim.startswith("Z") or dim.lower() in {"z", "depth", "k"}:
            return dim
    raise ValueError(f"Could not detect vertical dimension from UVEL dims: {ds['UVEL'].dims}")


def _check_required_mitgcm_cgrid(ds: xr.Dataset) -> None:
    required_coords = ["T", "Y", "Xp1", "Yp1", "X"]
    required_vars = ["UVEL", "VVEL"]

    missing_coords = [c for c in required_coords if c not in ds.coords and c not in ds.variables]
    missing_vars = [v for v in required_vars if v not in ds.variables]

    if missing_coords or missing_vars:
        raise KeyError(
            "Dataset does not match expected MITgcm C-grid format.\n"
            f"Missing coords: {missing_coords}\n"
            f"Missing vars: {missing_vars}\n"
            f"Available coords: {list(ds.coords)}\n"
            f"Available variables: {list(ds.data_vars)}"
        )


def _make_relative_time_seconds(
    ds: xr.Dataset,
    time_step_seconds: float = 10800.0,
    use_dataset_time: bool = False,
) -> np.ndarray:
    """
    Return Parcels time coordinate in seconds.

    Default: use index-based time, so output index 0 becomes t=0.
    This is convenient because your MITgcm diagnostics appear to be centered
    at e.g. 5400, 16200, ... seconds, while for Parcels experiments you want
    release_time_index=0 to mean release at the first available velocity field.
    """
    nt = ds.sizes["T"]

    if use_dataset_time:
        t = np.asarray(ds["T"].values, dtype=float)
        return t - t[0]

    return np.arange(nt, dtype=float) * float(time_step_seconds)


def _center_u_to_x(u: xr.DataArray, x: np.ndarray, y: np.ndarray, time: np.ndarray) -> xr.DataArray:
    """
    Center UVEL from Xp1 faces to X tracer centers.

    Input expected after selecting depth:
    UVEL(T, Y, Xp1)
    """
    u_left = u.isel(Xp1=slice(0, -1)).data
    u_right = u.isel(Xp1=slice(1, None)).data
    u_center = 0.5 * (u_left + u_right)

    return xr.DataArray(
        u_center,
        dims=("time", "y", "x"),
        coords={"time": time, "y": y, "x": x},
        name="U",
        attrs={"units": "m s-1", "long_name": "zonal velocity centered to tracer grid"},
    )


def _center_v_to_y(v: xr.DataArray, x: np.ndarray, y: np.ndarray, time: np.ndarray) -> xr.DataArray:
    """
    Center VVEL from Yp1 faces to Y tracer centers.

    Input expected after selecting depth:
    VVEL(T, Yp1, X)
    """
    v_lower = v.isel(Yp1=slice(0, -1)).data
    v_upper = v.isel(Yp1=slice(1, None)).data
    v_center = 0.5 * (v_lower + v_upper)

    return xr.DataArray(
        v_center,
        dims=("time", "y", "x"),
        coords={"time": time, "y": y, "x": x},
        name="V",
        attrs={"units": "m s-1", "long_name": "meridional velocity centered to tracer grid"},
    )


def mitgcm_cgrid_to_parcels_dataset(
    path: str | Path,
    level_indices: Sequence[int] = (0,),
    time_step_seconds: float = 10800.0,
    use_dataset_time: bool = False,
    chunks: Optional[dict] = None,
) -> tuple[xr.Dataset, FieldMeta]:
    """
    Convert your MITgcm C-grid NetCDF to a Parcels-ready flat-grid dataset.

    Output:
        ds_parcels["U"](time, y, x)
        ds_parcels["V"](time, y, x)

    By default, only k=0 is used. If multiple level_indices are provided,
    velocities are centered per level first and then averaged.
    """
    path = Path(path)
    ds = xr.open_dataset(path, chunks=chunks)
    _check_required_mitgcm_cgrid(ds)

    z_dim = _find_z_dim(ds)

    meta = FieldMeta(
        path=path,
        z_name=z_dim,
    )

    x = np.asarray(ds["X"].values, dtype=float)
    y = np.asarray(ds["Y"].values, dtype=float)
    time = _make_relative_time_seconds(
        ds,
        time_step_seconds=time_step_seconds,
        use_dataset_time=use_dataset_time,
    )

    u_layers = []
    v_layers = []

    nz = ds.sizes[z_dim]
    for k in level_indices:
        if k < 0 or k >= nz:
            raise IndexError(f"Requested level k={k}, but dataset has nz={nz}")

        u_k = ds["UVEL"].isel({z_dim: int(k)})
        v_k = ds["VVEL"].isel({z_dim: int(k)})

        u_c = _center_u_to_x(u_k, x=x, y=y, time=time)
        v_c = _center_v_to_y(v_k, x=x, y=y, time=time)

        u_layers.append(u_c)
        v_layers.append(v_c)

    if len(u_layers) == 1:
        U = u_layers[0]
        V = v_layers[0]
    else:
        U = xr.concat(u_layers, dim="level").mean("level", skipna=True)
        V = xr.concat(v_layers, dim="level").mean("level", skipna=True)

    ds_parcels = xr.Dataset(
        data_vars={"U": U, "V": V},
        coords={"time": time, "y": y, "x": x},
        attrs={
            "source": str(path),
            "description": "MITgcm C-grid UVEL/VVEL centered to tracer grid for Parcels",
            "level_indices": str(list(level_indices)),
            "time_step_seconds": float(time_step_seconds),
            "use_dataset_time": bool(use_dataset_time),
        },
    )

    return ds_parcels, meta


def build_fieldset(
    path: str | Path,
    surface_only: bool = True,
    mesh: str = "flat",
    chunks: Optional[dict] = None,
    level_indices: Sequence[int] = (0,),
    time_step_seconds: float = 10800.0,
    use_dataset_time: bool = False,
    periodic: bool = True,
) -> tuple[FieldSet, FieldMeta, xr.Dataset]:
    """
    Build a Parcels FieldSet from your fixed MITgcm C-grid format.

    surface_only is kept for compatibility with your run.py, but this function
    always returns a 2D surface/near-surface fieldset.
    """
    ds_parcels, meta = mitgcm_cgrid_to_parcels_dataset(
        path=path,
        level_indices=level_indices,
        time_step_seconds=time_step_seconds,
        use_dataset_time=use_dataset_time,
        chunks=chunks,
    )

    variables = {"U": "U", "V": "V"}
    dimensions = {"lon": "x", "lat": "y", "time": "time"}

    fieldset = FieldSet.from_xarray_dataset(
        ds_parcels,
        variables=variables,
        dimensions=dimensions,
        mesh=mesh,
    )

    # Physical periodic box edges from tracer-cell centers.
    x = np.asarray(ds_parcels["x"].values, dtype=float)
    y = np.asarray(ds_parcels["y"].values, dtype=float)

    dx = float(np.nanmedian(np.diff(x)))
    dy = float(np.nanmedian(np.diff(y)))

    x_edge_min = float(x[0] - 0.5 * dx)
    x_edge_max = float(x[-1] + 0.5 * dx)
    y_edge_min = float(y[0] - 0.5 * dy)
    y_edge_max = float(y[-1] + 0.5 * dy)

    if periodic:
        # Adds halo cells so interpolation near periodic boundaries remains valid.
        fieldset.add_periodic_halo(zonal=True, meridional=True)

        fieldset.add_constant("x_edge_min", x_edge_min)
        fieldset.add_constant("x_edge_max", x_edge_max)
        fieldset.add_constant("y_edge_min", y_edge_min)
        fieldset.add_constant("y_edge_max", y_edge_max)
        fieldset.add_constant("Lx", x_edge_max - x_edge_min)
        fieldset.add_constant("Ly", y_edge_max - y_edge_min)

    return fieldset, meta, ds_parcels


def summarize_dataset(path: str | Path) -> dict:
    path = Path(path)
    ds = xr.open_dataset(path)
    _check_required_mitgcm_cgrid(ds)
    z_dim = _find_z_dim(ds)

    return {
        "path": str(path),
        "dims": dict(ds.sizes),
        "coords": list(ds.coords),
        "data_vars": list(ds.data_vars),
        "time_name": "T",
        "z_name": z_dim,
        "x_center": "X",
        "y_center": "Y",
        "x_u_face": "Xp1",
        "y_v_face": "Yp1",
        "u_name": "UVEL",
        "v_name": "VVEL",
        "u_dims": ds["UVEL"].dims,
        "v_dims": ds["VVEL"].dims,
    }


def quick_qc_parcels_input(ds: xr.Dataset) -> dict:
    qc = {
        "dims": dict(ds.sizes),
        "coords": list(ds.coords),
        "data_vars": list(ds.data_vars),
        "U_dims": ds["U"].dims,
        "V_dims": ds["V"].dims,
        "U_shape": tuple(ds["U"].shape),
        "V_shape": tuple(ds["V"].shape),
        "x_min": float(ds["x"].min()),
        "x_max": float(ds["x"].max()),
        "y_min": float(ds["y"].min()),
        "y_max": float(ds["y"].max()),
        "time0": float(ds["time"].values[0]),
        "time_last": float(ds["time"].values[-1]),
        "dt_head": np.diff(ds["time"].values[:5]).tolist(),
        "U_nan_fraction_t0": float(np.isnan(ds["U"].isel(time=0)).mean()),
        "V_nan_fraction_t0": float(np.isnan(ds["V"].isel(time=0)).mean()),
    }
    return qc