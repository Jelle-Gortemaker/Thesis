from __future__ import annotations

import numpy as np
import xarray as xr


def _get_xy(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    x_name = "lon" if "lon" in ds.variables else "x"
    y_name = "lat" if "lat" in ds.variables else "y"
    x = np.asarray(ds[x_name].values)
    y = np.asarray(ds[y_name].values)
    return x, y


def density_map_metrics(
    ds: xr.Dataset,
    nx_bins: int = 50,
    ny_bins: int = 50,
    hotspot_percentile: float = 95.0,
) -> tuple[xr.Dataset, xr.DataArray]:
    x, y = _get_xy(ds)
    n_particles, n_times = x.shape

    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    ymin = np.nanmin(y)
    ymax = np.nanmax(y)

    xedges = np.linspace(xmin, xmax, nx_bins + 1)
    yedges = np.linspace(ymin, ymax, ny_bins + 1)

    H_all = np.zeros((n_times, ny_bins, nx_bins), dtype=float)
    patchiness = np.full(n_times, np.nan)
    hotspot_fraction = np.full(n_times, np.nan)
    hotspot_threshold = np.full(n_times, np.nan)

    for t in range(n_times):
        xt = x[:, t]
        yt = y[:, t]
        mask = np.isfinite(xt) & np.isfinite(yt)

        H, _, _ = np.histogram2d(xt[mask], yt[mask], bins=[xedges, yedges])
        H = H.T
        H_all[t] = H

        mean = np.mean(H)
        var = np.var(H)
        patchiness[t] = var / (mean**2) if mean > 0 else np.nan

        positive = H[H > 0]
        if positive.size > 0:
            thr = np.percentile(positive, hotspot_percentile)
            hotspot_threshold[t] = thr
            hotspot_fraction[t] = np.mean(H >= thr)

    metrics = xr.Dataset(
        data_vars={
            "patchiness_index": (("obs",), patchiness),
            "hotspot_fraction": (("obs",), hotspot_fraction),
            "hotspot_threshold": (("obs",), hotspot_threshold),
        }
    )

    density_maps = xr.DataArray(
        H_all,
        dims=("obs", "ybin", "xbin"),
        name="particle_density",
    )

    return metrics, density_maps