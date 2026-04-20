from __future__ import annotations

import numpy as np
import xarray as xr


def _get_xy(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    x_name = "lon" if "lon" in ds.variables else "x"
    y_name = "lat" if "lat" in ds.variables else "y"
    return np.asarray(ds[x_name].values), np.asarray(ds[y_name].values)


def absolute_dispersion(ds: xr.Dataset) -> xr.Dataset:
    x, y = _get_xy(ds)
    x0 = x[:, [0]]
    y0 = y[:, [0]]
    msd = np.nanmean((x - x0) ** 2 + (y - y0) ** 2, axis=0)
    return xr.Dataset({"absolute_dispersion": (("obs",), msd)})


def pair_dispersion(ds: xr.Dataset, max_pairs: int = 2000, seed: int = 42) -> xr.Dataset:
    x, y = _get_xy(ds)
    n_particles, _ = x.shape
    rng = np.random.default_rng(seed)

    n_pairs = min(max_pairs, n_particles * (n_particles - 1) // 2)
    i = np.empty(n_pairs, dtype=int)
    j = np.empty(n_pairs, dtype=int)

    for k in range(n_pairs):
        ii, jj = rng.choice(n_particles, size=2, replace=False)
        i[k] = ii
        j[k] = jj

    dx = x[i, :] - x[j, :]
    dy = y[i, :] - y[j, :]
    r2 = dx**2 + dy**2
    pair_disp = np.nanmean(r2, axis=0)

    return xr.Dataset({"pair_dispersion": (("obs",), pair_disp)})