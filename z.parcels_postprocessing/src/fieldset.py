from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from parcels import FieldSet


@dataclass
class FieldMeta:
    path: Path
    x_name: str
    y_name: str
    time_name: str
    depth_name: Optional[str]
    u_name: str
    v_name: str
    is_3d: bool
    mesh: str = "flat"


def _pick_first_existing(candidates: list[str], available: list[str]) -> Optional[str]:
    lower_map = {name.lower(): name for name in available}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _detect_coord_names(ds: xr.Dataset) -> tuple[str, str, Optional[str], Optional[str]]:
    names = list(ds.coords) + list(ds.dims)

    x_name = _pick_first_existing(["X", "XC", "x", "lon", "Xp1", "XG"], names)
    y_name = _pick_first_existing(["Y", "YC", "y", "lat", "Yp1", "YG"], names)
    time_name = _pick_first_existing(["T", "time", "TIME", "iter"], names)
    depth_name = _pick_first_existing(["Z", "depth", "DEPTH", "Zl", "Zu"], names)

    if x_name is None or y_name is None:
        raise ValueError(f"Could not detect X/Y coordinates. Available: {names}")

    return x_name, y_name, time_name, depth_name


def _detect_velocity_names(ds: xr.Dataset) -> tuple[str, str]:
    data_vars = list(ds.data_vars)

    u_name = _pick_first_existing(["U", "UVEL", "uo", "u"], data_vars)
    v_name = _pick_first_existing(["V", "VVEL", "vo", "v"], data_vars)

    if u_name is None or v_name is None:
        raise ValueError(f"Could not detect U/V variables. Available: {data_vars}")

    return u_name, v_name


def inspect_mitgcm_nc(path: str | Path) -> FieldMeta:
    path = Path(path)
    ds = xr.open_dataset(path)

    x_name, y_name, time_name, depth_name = _detect_coord_names(ds)
    u_name, v_name = _detect_velocity_names(ds)

    u_dims = ds[u_name].dims
    is_3d = depth_name is not None and depth_name in u_dims

    if time_name is None:
        raise ValueError("Could not detect time coordinate.")

    return FieldMeta(
        path=path,
        x_name=x_name,
        y_name=y_name,
        time_name=time_name,
        depth_name=depth_name if is_3d else None,
        u_name=u_name,
        v_name=v_name,
        is_3d=is_3d,
        mesh="flat",
    )


def _rename_to_parcels_convention(ds: xr.Dataset, meta: FieldMeta) -> xr.Dataset:
    rename_map = {
        meta.x_name: "x",
        meta.y_name: "y",
        meta.time_name: "time",
    }
    if meta.depth_name is not None:
        rename_map[meta.depth_name] = "depth"

    return ds.rename({k: v for k, v in rename_map.items() if k in ds.coords or k in ds.dims})


def _extract_surface_if_needed(ds: xr.Dataset, meta: FieldMeta, surface_only: bool) -> tuple[xr.Dataset, bool]:
    if not meta.is_3d:
        return ds, False

    if not surface_only:
        return ds, True

    depth_name = meta.depth_name
    assert depth_name is not None

    depth_values = np.asarray(ds[depth_name].values)
    idx = int(np.argmin(np.abs(depth_values)))
    ds2 = ds.isel({depth_name: idx})
    return ds2, False


def _ensure_parcels_coords(ds: xr.Dataset) -> xr.Dataset:
    coord_map = {}
    for cname in ["x", "y", "time", "depth"]:
        if cname in ds.coords:
            coord_map[cname] = ds.coords[cname]
        elif cname in ds.variables:
            coord_map[cname] = ds[cname]
    if coord_map:
        ds = ds.assign_coords(coord_map)
    return ds


def build_fieldset(
    path: str | Path,
    surface_only: bool = True,
    mesh: str = "flat",
    chunks: Optional[dict] = None,
) -> tuple[FieldSet, FieldMeta, xr.Dataset]:
    path = Path(path)
    ds = xr.open_dataset(path, chunks=chunks)
    meta = inspect_mitgcm_nc(path)

    ds_work, keep_depth = _extract_surface_if_needed(ds, meta, surface_only=surface_only)
    ds_work = _rename_to_parcels_convention(ds_work, meta)
    ds_work = _ensure_parcels_coords(ds_work)

    variables = {"U": meta.u_name, "V": meta.v_name}

    if keep_depth:
        dimensions = {"lon": "x", "lat": "y", "time": "time", "depth": "depth"}
    else:
        dimensions = {"lon": "x", "lat": "y", "time": "time"}

    fieldset = FieldSet.from_xarray_dataset(
        ds_work,
        variables=variables,
        dimensions=dimensions,
        mesh=mesh,
    )

    return fieldset, meta, ds_work


def summarize_dataset(path: str | Path) -> dict:
    meta = inspect_mitgcm_nc(path)
    ds = xr.open_dataset(path)

    return {
        "path": str(path),
        "dims": dict(ds.sizes),
        "coords": list(ds.coords),
        "data_vars": list(ds.data_vars),
        "x_name": meta.x_name,
        "y_name": meta.y_name,
        "time_name": meta.time_name,
        "depth_name": meta.depth_name,
        "u_name": meta.u_name,
        "v_name": meta.v_name,
        "is_3d": meta.is_3d,
    }


def quick_qc_parcels_input(ds: xr.Dataset, u_name: str, v_name: str) -> dict:
    qc = {}

    qc["dims"] = dict(ds.sizes)
    qc["coords"] = list(ds.coords)
    qc["u_dims"] = ds[u_name].dims
    qc["v_dims"] = ds[v_name].dims
    qc["u_shape"] = tuple(ds[u_name].shape)
    qc["v_shape"] = tuple(ds[v_name].shape)

    for cname in ["x", "y", "time"]:
        qc[f"has_{cname}"] = cname in ds.coords

    qc["has_depth"] = "depth" in ds.coords

    qc["x_min"] = float(ds["x"].min()) if "x" in ds.coords else None
    qc["x_max"] = float(ds["x"].max()) if "x" in ds.coords else None
    qc["y_min"] = float(ds["y"].min()) if "y" in ds.coords else None
    qc["y_max"] = float(ds["y"].max()) if "y" in ds.coords else None

    if "time" in ds.coords:
        qc["n_time"] = int(ds.sizes["time"])
        qc["time0"] = str(ds["time"].values[0])
        qc["time_last"] = str(ds["time"].values[-1])

    if "depth" in ds.coords:
        qc["depth_values_head"] = np.asarray(ds["depth"].values[:5]).tolist()

    qc["u_nan_fraction_t0"] = float(np.isnan(ds[u_name].isel(time=0)).mean())
    qc["v_nan_fraction_t0"] = float(np.isnan(ds[v_name].isel(time=0)).mean())

    return qc