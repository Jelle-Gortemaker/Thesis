from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.spatial import Voronoi, cKDTree


def _get_xy(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    x_name = "lon" if "lon" in ds.variables else "x"
    y_name = "lat" if "lat" in ds.variables else "y"
    return np.asarray(ds[x_name].values), np.asarray(ds[y_name].values)


def nearest_neighbor_metric(ds: xr.Dataset) -> xr.Dataset:
    x, y = _get_xy(ds)
    _, n_times = x.shape

    mean_nn = np.full(n_times, np.nan)
    std_nn = np.full(n_times, np.nan)

    for t in range(n_times):
        pts = np.column_stack([x[:, t], y[:, t]])
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) < 2:
            continue

        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=2)
        nn = dists[:, 1]
        mean_nn[t] = np.mean(nn)
        std_nn[t] = np.std(nn)

    return xr.Dataset({
        "nn_mean": (("obs",), mean_nn),
        "nn_std": (("obs",), std_nn),
    })


def _voronoi_finite_polygon_areas(points: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    xmin, xmax, ymin, ymax = bbox
    x = points[:, 0]
    y = points[:, 1]

    mirrored = [
        points,
        np.column_stack([2 * xmin - x, y]),
        np.column_stack([2 * xmax - x, y]),
        np.column_stack([x, 2 * ymin - y]),
        np.column_stack([x, 2 * ymax - y]),
    ]
    pts_aug = np.vstack(mirrored)
    vor = Voronoi(pts_aug)

    areas = np.full(len(points), np.nan)

    for i in range(len(points)):
        region_index = vor.point_region[i]
        region = vor.regions[region_index]
        if -1 in region or len(region) == 0:
            continue

        poly = vor.vertices[region]
        if poly.shape[0] < 3:
            continue

        px = poly[:, 0]
        py = poly[:, 1]
        areas[i] = 0.5 * np.abs(np.dot(px, np.roll(py, -1)) - np.dot(py, np.roll(px, -1)))

    return areas


def voronoi_metrics(ds: xr.Dataset) -> xr.Dataset:
    x, y = _get_xy(ds)
    _, n_times = x.shape

    bbox = (np.nanmin(x), np.nanmax(x), np.nanmin(y), np.nanmax(y))

    area_mean = np.full(n_times, np.nan)
    area_std = np.full(n_times, np.nan)
    area_cv = np.full(n_times, np.nan)

    for t in range(n_times):
        pts = np.column_stack([x[:, t], y[:, t]])
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) < 4:
            continue

        areas = _voronoi_finite_polygon_areas(pts, bbox)
        areas = areas[np.isfinite(areas)]
        if len(areas) == 0:
            continue

        area_mean[t] = np.mean(areas)
        area_std[t] = np.std(areas)
        area_cv[t] = area_std[t] / area_mean[t] if area_mean[t] > 0 else np.nan

    return xr.Dataset({
        "voronoi_area_mean": (("obs",), area_mean),
        "voronoi_area_std": (("obs",), area_std),
        "voronoi_area_cv": (("obs",), area_cv),
    })