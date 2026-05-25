from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import Voronoi


@dataclass(frozen=True)
class PeriodicDomain:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def Lx(self) -> float:
        return self.x_max - self.x_min

    @property
    def Ly(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.Lx * self.Ly


def domain_from_parcels_dataset(ds_parcels: xr.Dataset) -> PeriodicDomain:
    """
    Infer periodic box edges from the Parcels-ready x/y tracer grid.
    """
    x = np.asarray(ds_parcels["x"].values, dtype=float)
    y = np.asarray(ds_parcels["y"].values, dtype=float)

    dx = float(np.nanmedian(np.diff(x)))
    dy = float(np.nanmedian(np.diff(y)))

    return PeriodicDomain(
        x_min=float(x[0] - 0.5 * dx),
        x_max=float(x[-1] + 0.5 * dx),
        y_min=float(y[0] - 0.5 * dy),
        y_max=float(y[-1] + 0.5 * dy),
    )


def get_xy_names(ds: xr.Dataset) -> tuple[str, str]:
    xname = "lon" if "lon" in ds.variables else "x"
    yname = "lat" if "lat" in ds.variables else "y"
    return xname, yname


def wrap_to_domain(x: np.ndarray, y: np.ndarray, domain: PeriodicDomain) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    xw = ((x - domain.x_min) % domain.Lx) + domain.x_min
    yw = ((y - domain.y_min) % domain.Ly) + domain.y_min

    return xw, yw


def polygon_area(poly: np.ndarray) -> float:
    if poly is None or len(poly) < 3:
        return 0.0

    x = poly[:, 0]
    y = poly[:, 1]

    return 0.5 * abs(
        np.dot(x, np.roll(y, -1)) -
        np.dot(y, np.roll(x, -1))
    )


def _clip_polygon_axis(poly: np.ndarray, axis: int, bound: float, keep_greater: bool) -> np.ndarray:
    if poly is None or len(poly) == 0:
        return np.empty((0, 2), dtype=float)

    out = []

    def inside(p):
        return p[axis] >= bound if keep_greater else p[axis] <= bound

    def intersect(p1, p2):
        denom = p2[axis] - p1[axis]
        if abs(denom) < 1e-14:
            return p2.copy()

        t = (bound - p1[axis]) / denom
        return p1 + t * (p2 - p1)

    prev = poly[-1]
    prev_inside = inside(prev)

    for curr in poly:
        curr_inside = inside(curr)

        if curr_inside:
            if not prev_inside:
                out.append(intersect(prev, curr))
            out.append(curr)
        elif prev_inside:
            out.append(intersect(prev, curr))

        prev = curr
        prev_inside = curr_inside

    if len(out) == 0:
        return np.empty((0, 2), dtype=float)

    return np.asarray(out, dtype=float)


def clip_polygon_to_box(poly: np.ndarray, domain: PeriodicDomain) -> np.ndarray:
    """
    Sutherland-Hodgman polygon clipping to the rectangular domain.
    """
    clipped = np.asarray(poly, dtype=float)

    clipped = _clip_polygon_axis(clipped, axis=0, bound=domain.x_min, keep_greater=True)
    clipped = _clip_polygon_axis(clipped, axis=0, bound=domain.x_max, keep_greater=False)
    clipped = _clip_polygon_axis(clipped, axis=1, bound=domain.y_min, keep_greater=True)
    clipped = _clip_polygon_axis(clipped, axis=1, bound=domain.y_max, keep_greater=False)

    return clipped


def voronoi_finite_polygons_2d(vor: Voronoi, radius: float | None = None) -> list[np.ndarray]:
    """
    Convert 2D Voronoï regions to finite polygons.

    Returns one polygon per input point. Infinite regions are closed by adding
    artificial far points.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Only 2D Voronoï diagrams are supported.")

    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2.0

    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    regions = []

    for p1, region_index in enumerate(vor.point_region):
        vertices = vor.regions[region_index]

        if len(vertices) == 0:
            regions.append(np.empty((0, 2), dtype=float))
            continue

        if all(v >= 0 for v in vertices):
            polygon = np.asarray([vor.vertices[v] for v in vertices], dtype=float)
        else:
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in all_ridges.get(p1, []):
                if v2 < 0:
                    v1, v2 = v2, v1

                if v1 >= 0:
                    continue

                tangent = vor.points[p2] - vor.points[p1]
                tangent /= np.linalg.norm(tangent)

                normal = np.array([-tangent[1], tangent[0]])

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, normal)) * normal

                far_point = vor.vertices[v2] + direction * radius

                new_vertices.append(far_point.tolist())
                new_region.append(len(new_vertices) - 1)

            polygon = np.asarray([new_vertices[v] for v in new_region], dtype=float)

        if len(polygon) >= 3:
            centroid = polygon.mean(axis=0)
            angles = np.arctan2(polygon[:, 1] - centroid[1], polygon[:, 0] - centroid[0])
            polygon = polygon[np.argsort(angles)]

        regions.append(polygon)

    return regions


def periodic_voronoi_snapshot(
    x: np.ndarray,
    y: np.ndarray,
    domain: PeriodicDomain,
    min_area: float = 0.0,
) -> dict:
    """
    Compute a periodic Voronoï tessellation by tiling particles in a 3x3 grid.

    The returned polygons are clipped to the central domain. For particles whose
    periodic Voronoï cell crosses a boundary, multiple clipped polygon pieces may
    exist. Areas are summed per original particle.

    Returns:
        {
            "pieces": list of {"release_index", "polygon", "area"},
            "cell_area": array with one total area per particle,
        }
    """
    xw, yw = wrap_to_domain(x, y, domain)

    valid = np.isfinite(xw) & np.isfinite(yw)
    xw = xw[valid]
    yw = yw[valid]

    n = len(xw)
    if n < 4:
        raise ValueError("At least 4 valid points are needed for Voronoï tessellation.")

    base_points = np.column_stack([xw, yw])

    tiled_points = []
    tiled_original_ids = []

    shifts = [
        (-domain.Lx, -domain.Ly), (0.0, -domain.Ly), (domain.Lx, -domain.Ly),
        (-domain.Lx, 0.0),        (0.0, 0.0),        (domain.Lx, 0.0),
        (-domain.Lx, domain.Ly),  (0.0, domain.Ly),  (domain.Lx, domain.Ly),
    ]

    for sx, sy in shifts:
        shifted = base_points + np.array([sx, sy])
        tiled_points.append(shifted)
        tiled_original_ids.append(np.arange(n, dtype=int))

    tiled_points = np.vstack(tiled_points)
    tiled_original_ids = np.concatenate(tiled_original_ids)

    vor = Voronoi(tiled_points)
    regions = voronoi_finite_polygons_2d(vor, radius=2.0 * max(domain.Lx, domain.Ly))

    cell_area = np.zeros(n, dtype=float)
    pieces = []

    for point_index, polygon in enumerate(regions):
        if polygon is None or len(polygon) < 3:
            continue

        clipped = clip_polygon_to_box(polygon, domain)

        if len(clipped) < 3:
            continue

        area = polygon_area(clipped)

        if area <= min_area:
            continue

        original_id = int(tiled_original_ids[point_index])
        cell_area[original_id] += area

        pieces.append(
            {
                "release_index": original_id,
                "polygon": clipped,
                "area": area,
            }
        )

    return {
        "pieces": pieces,
        "cell_area": cell_area,
        "valid_count": n,
        "domain_area": domain.area,
    }


def compute_voronoi_summary(
    ds_traj: xr.Dataset,
    obs_indices: Iterable[int],
    domain: PeriodicDomain,
) -> xr.Dataset:
    """
    Compute Voronoï area summary statistics for selected trajectory snapshots.

    Stores compact summary only, not all polygons.
    """
    xname, yname = get_xy_names(ds_traj)

    obs_indices = list(obs_indices)

    mean_area = []
    std_area = []
    cv_area = []
    p05_area = []
    p50_area = []
    p95_area = []
    n_cells = []
    total_area = []
    obs_out = []

    for obs_idx in obs_indices:
        x = ds_traj[xname].isel(obs=obs_idx).values
        y = ds_traj[yname].isel(obs=obs_idx).values

        result = periodic_voronoi_snapshot(x, y, domain)
        areas = result["cell_area"]

        valid_areas = areas[np.isfinite(areas) & (areas > 0.0)]

        obs_out.append(obs_idx)
        n_cells.append(len(valid_areas))
        total_area.append(float(np.nansum(valid_areas)))

        mean = float(np.nanmean(valid_areas))
        std = float(np.nanstd(valid_areas))

        mean_area.append(mean)
        std_area.append(std)
        cv_area.append(std / mean if mean > 0.0 else np.nan)
        p05_area.append(float(np.nanpercentile(valid_areas, 5)))
        p50_area.append(float(np.nanpercentile(valid_areas, 50)))
        p95_area.append(float(np.nanpercentile(valid_areas, 95)))

    return xr.Dataset(
        data_vars={
            "voronoi_area_mean": ("obs", mean_area),
            "voronoi_area_std": ("obs", std_area),
            "voronoi_area_cv": ("obs", cv_area),
            "voronoi_area_p05": ("obs", p05_area),
            "voronoi_area_p50": ("obs", p50_area),
            "voronoi_area_p95": ("obs", p95_area),
            "voronoi_n_cells": ("obs", n_cells),
            "voronoi_total_area": ("obs", total_area),
        },
        coords={
            "obs": np.asarray(obs_out, dtype=int),
        },
        attrs={
            "description": "Periodic Voronoi area summary statistics.",
            "domain_area": float(domain.area),
            "x_min": float(domain.x_min),
            "x_max": float(domain.x_max),
            "y_min": float(domain.y_min),
            "y_max": float(domain.y_max),
        },
    )


def plot_voronoi_snapshot(
    ax,
    pieces: list[dict],
    x: np.ndarray,
    y: np.ndarray,
    domain: PeriodicDomain,
    *,
    line_color="0.35",
    point_color=None,
    point_marker="o",
    point_size=8,
    line_width=0.35,
    line_alpha=0.75,
    point_alpha=0.9,
    km: bool = True,
    label: str | None = None,
):
    """
    Plot a Voronoï snapshot on an existing axes.
    """
    scale = 1000.0 if km else 1.0

    segments = []

    for piece in pieces:
        poly = piece["polygon"]

        if poly is None or len(poly) < 3:
            continue

        poly_plot = poly / scale
        closed = np.vstack([poly_plot, poly_plot[0]])
        segments.extend([[closed[i], closed[i + 1]] for i in range(len(closed) - 1)])

    lc = LineCollection(
        segments,
        colors=line_color,
        linewidths=line_width,
        alpha=line_alpha,
    )
    ax.add_collection(lc)

    xw, yw = wrap_to_domain(x, y, domain)

    ax.scatter(
        xw / scale,
        yw / scale,
        s=point_size,
        color=point_color,
        marker=point_marker,
        alpha=point_alpha,
        label=label,
        zorder=3,
    )

    ax.set_xlim(domain.x_min / scale, domain.x_max / scale)
    ax.set_ylim(domain.y_min / scale, domain.y_max / scale)
    ax.set_aspect("equal", adjustable="box")