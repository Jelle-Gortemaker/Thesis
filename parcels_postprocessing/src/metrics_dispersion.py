from __future__ import annotations

import numpy as np
import xarray as xr


def _get_xy(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    x_name = "lon" if "lon" in ds.data_vars else "x"
    y_name = "lat" if "lat" in ds.data_vars else "y"

    if x_name not in ds:
        for cand in ["lon", "x"]:
            if cand in ds.variables:
                x_name = cand
                break
    if y_name not in ds:
        for cand in ["lat", "y"]:
            if cand in ds.variables:
                y_name = cand
                break

    x = np.asarray(ds[x_name].values)
    y = np.asarray(ds[y_name].values)

    return x, y


def absolute_dispersion(ds: xr.Dataset) -> xr.Dataset:
    x, y = _get_xy(ds)

    x0 = x[:, [0]]
    y0 = y[:, [0]]

    msd = np.nanmean((x - x0) ** 2 + (y - y0) ** 2, axis=0)

    return xr.Dataset(
        data_vars={
            "absolute_dispersion": (("obs",), msd),
        }
    )


def pair_dispersion(ds: xr.Dataset, max_pairs: int = 2000, seed: int = 42) -> xr.Dataset:
    x, y = _get_xy(ds)

    n_particles, n_times = x.shape
    if n_particles < 2:
        raise ValueError("Need at least 2 particles for pair dispersion.")

    rng = np.random.default_rng(seed)

    all_i = []
    all_j = []
    for _ in range(min(max_pairs, n_particles * (n_particles - 1) // 2)):
        i, j = rng.choice(n_particles, size=2, replace=False)
        all_i.append(i)
        all_j.append(j)

    i = np.asarray(all_i)
    j = np.asarray(all_j)

    dx = x[i, :] - x[j, :]
    dy = y[i, :] - y[j, :]

    r2 = dx**2 + dy**2
    pair_disp = np.nanmean(r2, axis=0)

    return xr.Dataset(
        data_vars={
            "pair_dispersion": (("obs",), pair_disp),
        }
    )