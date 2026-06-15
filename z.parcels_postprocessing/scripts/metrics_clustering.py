from __future__ import annotations

from dataclasses import dataclass
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
    """Infer periodic box edges from the Parcels-ready x/y tracer grid."""
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
    """Sutherland-Hodgman polygon clipping to the rectangular domain."""
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
                norm = np.linalg.norm(tangent)
                if norm == 0:
                    continue
                tangent /= norm

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

    Returns cell_area with the same length as the input x/y arrays. Invalid
    particles get NaN. This makes the output convenient for residence-time
    calculations along particle trajectories.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xw_all, yw_all = wrap_to_domain(x, y, domain)

    valid = np.isfinite(xw_all) & np.isfinite(yw_all)
    original_ids = np.flatnonzero(valid)
    xw = xw_all[valid]
    yw = yw_all[valid]

    n_valid = len(xw)
    cell_area_full = np.full(len(x), np.nan, dtype=float)

    if n_valid < 4:
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
        tiled_original_ids.append(original_ids)

    tiled_points = np.vstack(tiled_points)
    tiled_original_ids = np.concatenate(tiled_original_ids)

    vor = Voronoi(tiled_points)
    regions = voronoi_finite_polygons_2d(vor, radius=2.0 * max(domain.Lx, domain.Ly))

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
        if np.isfinite(cell_area_full[original_id]):
            cell_area_full[original_id] += area
        else:
            cell_area_full[original_id] = area

        pieces.append(
            {
                "release_index": original_id,
                "polygon": clipped,
                "area": area,
            }
        )

    return {
        "pieces": pieces,
        "cell_area": cell_area_full,
        "valid_count": n_valid,
        "valid_original_ids": original_ids,
        "domain_area": domain.area,
    }


def _normalized_valid_areas(areas: np.ndarray) -> np.ndarray:
    areas = np.asarray(areas, dtype=float)
    valid = areas[np.isfinite(areas) & (areas > 0.0)]
    if valid.size == 0:
        return valid
    mean = np.nanmean(valid)
    if not np.isfinite(mean) or mean <= 0:
        return valid * np.nan
    return valid / mean


def poisson_voronoi_area_pdf_2d(area_norm: np.ndarray) -> np.ndarray:
    """
    Analytical approximation for a random 2D Poisson-Voronoï tessellation.

    The normalized cell area x=A/<A> is approximated by the Ferenc-Néda / Kiang
    gamma-like distribution:

        p(x) = (343/15) * sqrt(7/(2*pi)) * x**(5/2) * exp(-7*x/2)

    This has mean 1 and a standard deviation close to 0.52, matching the
    reference value commonly used for 2D Poisson Voronoï clustering tests.
    """
    x = np.asarray(area_norm, dtype=float)
    pdf = np.zeros_like(x, dtype=float)
    mask = x > 0.0
    pdf[mask] = (343.0 / 15.0) * np.sqrt(7.0 / (2.0 * np.pi)) * x[mask] ** 2.5 * np.exp(-3.5 * x[mask])
    return pdf


def poisson_voronoi_logarea_pdf_2d(log_area_norm: np.ndarray) -> np.ndarray:
    """PDF of y=ln(A/<A>) corresponding to poisson_voronoi_area_pdf_2d."""
    y = np.asarray(log_area_norm, dtype=float)
    x = np.exp(y)
    return x * poisson_voronoi_area_pdf_2d(x)


def _hist_pdf(values: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    hist, edges = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return hist.astype(float), centers.astype(float)


def find_first_pdf_intersection(x: np.ndarray, pdf: np.ndarray, pdf_ref: np.ndarray) -> float:
    """
    Find the first low-area intersection between measured and random PDFs.

    The cluster threshold in the Voronoï method is the first crossing where the
    particle PDF stops being above the Poisson reference as area increases.
    Very-low-probability tail bins are ignored because finite particle counts
    make those bins noisy.
    """
    x = np.asarray(x, dtype=float)
    pdf = np.asarray(pdf, dtype=float)
    pdf_ref = np.asarray(pdf_ref, dtype=float)

    ref_max = np.nanmax(pdf_ref) if np.any(np.isfinite(pdf_ref)) else np.nan
    ref_ok = pdf_ref > (1e-3 * ref_max if np.isfinite(ref_max) and ref_max > 0 else 0.0)
    mask = np.isfinite(x) & np.isfinite(pdf) & np.isfinite(pdf_ref) & ref_ok

    x = x[mask]
    diff = (pdf - pdf_ref)[mask]

    if x.size < 5:
        return np.nan

    order = np.argsort(x)
    x = x[order]
    diff = diff[order]

    # Light smoothing makes the intersection less sensitive to empty histogram
    # bins while retaining the low-area crossing.
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    kernel /= kernel.sum()
    diff_s = np.convolve(diff, kernel, mode="same")

    positive_seen = False
    for i in range(1, len(x)):
        if diff_s[i - 1] > 0:
            positive_seen = True
        if positive_seen and diff_s[i - 1] > 0 and diff_s[i] <= 0:
            denom = diff_s[i] - diff_s[i - 1]
            if denom == 0:
                return float(x[i])
            frac = -diff_s[i - 1] / denom
            return float(x[i - 1] + frac * (x[i] - x[i - 1]))

    return np.nan

def find_second_pdf_intersection(x: np.ndarray, pdf: np.ndarray, pdf_ref: np.ndarray) -> float:
    """
    Find the high-area ('void') intersection between measured and Poisson PDFs.

    This is the second crossing:
    first crossing -> cluster threshold nu_c
    second crossing -> void threshold nu_v
    """
    x = np.asarray(x, dtype=float)
    pdf = np.asarray(pdf, dtype=float)
    pdf_ref = np.asarray(pdf_ref, dtype=float)

    ref_max = np.nanmax(pdf_ref) if np.any(np.isfinite(pdf_ref)) else np.nan
    ref_ok = pdf_ref > (1e-3 * ref_max if np.isfinite(ref_max) and ref_max > 0 else 0.0)
    mask = np.isfinite(x) & np.isfinite(pdf) & np.isfinite(pdf_ref) & ref_ok

    x = x[mask]
    diff = (pdf - pdf_ref)[mask]

    if x.size < 5:
        return np.nan

    order = np.argsort(x)
    x = x[order]
    diff = diff[order]

    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
    kernel /= kernel.sum()
    diff_s = np.convolve(diff, kernel, mode="same")

    crossings = []
    for i in range(1, len(x)):
        if diff_s[i - 1] == 0:
            crossings.append(float(x[i - 1]))
        elif diff_s[i - 1] * diff_s[i] < 0:
            denom = diff_s[i] - diff_s[i - 1]
            if denom == 0:
                xc = float(x[i])
            else:
                frac = -diff_s[i - 1] / denom
                xc = float(x[i - 1] + frac * (x[i] - x[i - 1]))
            crossings.append(xc)

    if len(crossings) >= 2:
        return crossings[1]

    return np.nan


def voronoi_area_pdf_for_snapshot(
    areas: np.ndarray,
    *,
    bins: np.ndarray | None = None,
    log_area: bool = True,
) -> dict:
    """
    Compute the normalized Voronoï area PDF and Poisson reference for one snapshot.

    If log_area=True, the PDF is computed for ln(A/<A>), as in the commonly used
    Monchaux-style Voronoï PDF plot. The cluster threshold is returned in both
    log-space and normalized-area space.
    """
    area_norm = _normalized_valid_areas(areas)

    if area_norm.size == 0:
        raise ValueError("No valid Voronoï areas available for PDF calculation.")

    if log_area:
        values = np.log(area_norm)
        if bins is None:
            bins = np.linspace(-4.0, 4.0, 81)
        pdf, centers = _hist_pdf(values, bins)
        pdf_ref = poisson_voronoi_logarea_pdf_2d(centers)
        threshold_log = find_first_pdf_intersection(centers, pdf, pdf_ref)
        threshold_area = float(np.exp(threshold_log)) if np.isfinite(threshold_log) else np.nan
        x_name = "log_area_norm"
    else:
        values = area_norm
        if bins is None:
            bins = np.linspace(0.0, 5.0, 81)
        pdf, centers = _hist_pdf(values, bins)
        pdf_ref = poisson_voronoi_area_pdf_2d(centers)
        threshold_area = find_first_pdf_intersection(centers, pdf, pdf_ref)
        threshold_log = float(np.log(threshold_area)) if np.isfinite(threshold_area) and threshold_area > 0 else np.nan
        x_name = "area_norm"

    with np.errstate(divide="ignore", invalid="ignore"):
        relative_pdf = pdf / pdf_ref
    relative_pdf[~np.isfinite(relative_pdf)] = np.nan

    return {
        "area_norm": area_norm,
        "pdf": pdf,
        "pdf_poisson": pdf_ref,
        "relative_pdf": relative_pdf,
        "bin_edges": np.asarray(bins, dtype=float),
        "bin_centers": centers,
        "x_name": x_name,
        "log_area": bool(log_area),
        "cluster_threshold_area_norm": threshold_area,
        "cluster_threshold_log_area_norm": threshold_log,
        "sigma_area_norm": float(np.nanstd(area_norm)),
        "sigma_log_area_norm": float(np.nanstd(np.log(area_norm))),
    }

def compute_voronoi_pdf_aggregate(
    ds_traj: xr.Dataset,
    obs_indices: Iterable[int],
    domain: PeriodicDomain,
    *,
    bins: np.ndarray | None = None,
) -> xr.Dataset:
    """
    Aggregate Voronoï normalized-area PDFs over multiple snapshots,
    in Monchaux-style variable nu = A/<A>.
    """
    xname, yname = get_xy_names(ds_traj)
    obs_indices = list(obs_indices)

    all_area_norm = []

    for obs_idx in obs_indices:
        x = ds_traj[xname].isel(obs=obs_idx).values
        y = ds_traj[yname].isel(obs=obs_idx).values

        result = periodic_voronoi_snapshot(x, y, domain)
        area_norm = _normalized_valid_areas(result["cell_area"])

        if area_norm.size > 0:
            all_area_norm.append(area_norm)

    if len(all_area_norm) == 0:
        raise ValueError("No valid Voronoï areas found for aggregate PDF.")

    area_norm_all = np.concatenate(all_area_norm)

    if bins is None:
        bins = np.logspace(-2.0, 1.2, 60)

    count, edges = np.histogram(area_norm_all, bins=bins, density=False)
    pdf, _ = np.histogram(area_norm_all, bins=bins, density=True)

    centers = np.sqrt(edges[:-1] * edges[1:])
    pdf_ref = poisson_voronoi_area_pdf_2d(centers)

    with np.errstate(divide="ignore", invalid="ignore"):
        relative_pdf = pdf / pdf_ref
    relative_pdf[~np.isfinite(relative_pdf)] = np.nan

    min_count_for_threshold = 5
    valid_for_threshold = count >= min_count_for_threshold

    nu_c = find_first_pdf_intersection(
        centers[valid_for_threshold],
        pdf[valid_for_threshold],
        pdf_ref[valid_for_threshold],
    )

    nu_v = find_second_pdf_intersection(
        centers[valid_for_threshold],
        pdf[valid_for_threshold],
        pdf_ref[valid_for_threshold],
    )

    return xr.Dataset(
        data_vars={
            "pdf": ("bin", pdf.astype(float)),
            "pdf_poisson": ("bin", pdf_ref.astype(float)),
            "relative_pdf": ("bin", relative_pdf.astype(float)),
            "count": ("bin", count.astype(int)),
            "cluster_threshold_nu_c": xr.DataArray(float(nu_c)),
            "void_threshold_nu_v": xr.DataArray(float(nu_v)),
            "n_samples": xr.DataArray(int(area_norm_all.size)),
        },
        coords={
            "bin": np.arange(len(centers), dtype=int),
            "nu": ("bin", centers.astype(float)),
            "bin_left": ("bin", edges[:-1].astype(float)),
            "bin_right": ("bin", edges[1:].astype(float)),
        },
        attrs={
            "description": "Aggregate Monchaux-style Voronoi area PDF in nu=A/<A>.",
            "n_obs_aggregated": int(len(obs_indices)),
            "min_count_for_threshold": int(min_count_for_threshold),
        },
    )

def plot_voronoi_pdf_monchaux(
    pdf_ds: xr.Dataset,
    *,
    axes=None,
    particle_label: str = "particles",
    color="tab:blue",
    title_prefix: str | None = None,
    min_count: int = 5,
):
    """
    Monchaux-style two-panel Voronoï PDF figure.

    Panel (a): PDF of nu = A/<A>
    Panel (b): relative PDF = PDF / PDF_poisson
    """
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), facecolor="white")
    else:
        fig = axes[0].figure

    ax1, ax2 = axes

    nu = np.asarray(pdf_ds["nu"].values, dtype=float)
    pdf = np.asarray(pdf_ds["pdf"].values, dtype=float)
    pdf_ref = np.asarray(pdf_ds["pdf_poisson"].values, dtype=float)
    rel = np.asarray(pdf_ds["relative_pdf"].values, dtype=float)

    if "count" in pdf_ds:
        count = np.asarray(pdf_ds["count"].values, dtype=float)
    else:
        count = np.ones_like(nu)

    nu_c = float(pdf_ds["cluster_threshold_nu_c"].values)
    nu_v = float(pdf_ds["void_threshold_nu_v"].values)

    valid_particle = (
        np.isfinite(nu)
        & np.isfinite(pdf)
        & np.isfinite(pdf_ref)
        & np.isfinite(rel)
        & (nu > 0.0)
        & (pdf > 0.0)
        & (pdf_ref > 0.0)
        & (count >= min_count)
    )

    valid_ref = (
        np.isfinite(nu)
        & np.isfinite(pdf_ref)
        & (nu > 0.0)
        & (pdf_ref > 0.0)
    )

    pdf_plot = np.where(valid_particle, pdf, np.nan)
    ref_plot = np.where(valid_ref, pdf_ref, np.nan)
    rel_plot = np.where(valid_particle & (rel > 0.0), rel, np.nan)

    ax1.plot(nu, pdf_plot, color=color, lw=2.0, label=particle_label)
    ax1.plot(nu, ref_plot, color="0.25", lw=1.4, ls="--", label="Poisson")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(1e-2, 2e1)
    ax1.set_ylim(1e-4, 2e0)

    ax1.set_xlabel(r"$\nu=A/\langle A\rangle$")
    ax1.set_ylabel("PDF")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")

    ax2.plot(nu, rel_plot, color="0.35", lw=1.8)
    ax2.axhline(1.0, color="0.15", lw=1.0, ls="-.")

    xmin, xmax = 1e-2, 2e1

    if np.isfinite(nu_c):
        ax2.axvline(nu_c, color="0.2", lw=1.0, ls=":")
        ax2.axvspan(xmin, nu_c, color="0.35", alpha=0.25)
        ax2.text(nu_c, 70.0, r"$\nu_c$", ha="center", va="bottom")
        ax2.text(1.7e-2, 20.0, "Clusters", color="white")

    if np.isfinite(nu_v):
        ax2.axvline(nu_v, color="0.2", lw=1.0, ls=":")
        ax2.axvspan(nu_v, xmax, color="0.75", alpha=0.35)
        ax2.text(nu_v, 70.0, r"$\nu_v$", ha="center", va="bottom")
        ax2.text(max(nu_v * 1.2, 2.0), 20.0, "Voids", color="0.2")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(1e-1, 1e2)

    ax2.set_xlabel(r"$\nu=A/\langle A\rangle$")
    ax2.set_ylabel("relative PDF")
    ax2.grid(True, alpha=0.25)

    if title_prefix is not None:
        fig.suptitle(title_prefix)

    fig.tight_layout()
    return fig, axes


def compute_voronoi_summary(
    ds_traj: xr.Dataset,
    obs_indices: Iterable[int],
    domain: PeriodicDomain,
) -> xr.Dataset:
    """
    Compute compact Voronoï area summary statistics for selected snapshots.

    Stores summary only, not all polygons. For particle-level cluster flags and
    residence times, use compute_voronoi_clustering_timeseries().
    """
    xname, yname = get_xy_names(ds_traj)
    obs_indices = list(obs_indices)

    mean_area = []
    std_area = []
    cv_area = []
    p05_area = []
    p50_area = []
    p95_area = []
    sigma_area_norm = []
    n_cells = []
    total_area = []
    obs_out = []

    for obs_idx in obs_indices:
        x = ds_traj[xname].isel(obs=obs_idx).values
        y = ds_traj[yname].isel(obs=obs_idx).values

        result = periodic_voronoi_snapshot(x, y, domain)
        areas = result["cell_area"]
        valid_areas = areas[np.isfinite(areas) & (areas > 0.0)]
        area_norm = _normalized_valid_areas(valid_areas)

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
        sigma_area_norm.append(float(np.nanstd(area_norm)))

    return xr.Dataset(
        data_vars={
            "voronoi_area_mean": ("obs", mean_area),
            "voronoi_area_std": ("obs", std_area),
            "voronoi_area_cv": ("obs", cv_area),
            "voronoi_area_p05": ("obs", p05_area),
            "voronoi_area_p50": ("obs", p50_area),
            "voronoi_area_p95": ("obs", p95_area),
            "voronoi_area_norm_std": ("obs", sigma_area_norm),
            "voronoi_n_cells": ("obs", n_cells),
            "voronoi_total_area": ("obs", total_area),
        },
        coords={"obs": np.asarray(obs_out, dtype=int)},
        attrs={
            "description": "Periodic Voronoi area summary statistics.",
            "domain_area": float(domain.area),
            "poisson_area_norm_std_2d": 0.52,
            "x_min": float(domain.x_min),
            "x_max": float(domain.x_max),
            "y_min": float(domain.y_min),
            "y_max": float(domain.y_max),
        },
    )


def compute_voronoi_pdf(
    ds_traj: xr.Dataset,
    obs_indices: Iterable[int],
    domain: PeriodicDomain,
    *,
    bins: np.ndarray | None = None,
    log_area: bool = True,
) -> xr.Dataset:
    """Compute Voronoï area PDFs for selected observations."""
    xname, yname = get_xy_names(ds_traj)
    obs_indices = list(obs_indices)

    pdfs = []
    pdf_refs = []
    rels = []
    thresholds = []
    thresholds_log = []
    sigma_area = []
    sigma_log = []
    clustered_fraction = []
    obs_out = []
    centers_out = None
    edges_out = None

    for obs_idx in obs_indices:
        x = ds_traj[xname].isel(obs=obs_idx).values
        y = ds_traj[yname].isel(obs=obs_idx).values
        result = periodic_voronoi_snapshot(x, y, domain)
        out = voronoi_area_pdf_for_snapshot(result["cell_area"], bins=bins, log_area=log_area)

        if centers_out is None:
            centers_out = out["bin_centers"]
            edges_out = out["bin_edges"]

        thr = out["cluster_threshold_area_norm"]
        area_norm = out["area_norm"]
        if np.isfinite(thr):
            frac = float(np.mean(area_norm < thr))
        else:
            frac = np.nan

        obs_out.append(obs_idx)
        pdfs.append(out["pdf"])
        pdf_refs.append(out["pdf_poisson"])
        rels.append(out["relative_pdf"])
        thresholds.append(thr)
        thresholds_log.append(out["cluster_threshold_log_area_norm"])
        sigma_area.append(out["sigma_area_norm"])
        sigma_log.append(out["sigma_log_area_norm"])
        clustered_fraction.append(frac)

    coord_name = "log_area_norm" if log_area else "area_norm"

    ds_out = xr.Dataset(
        data_vars={
            "pdf": (("obs", "bin"), np.asarray(pdfs)),
            "pdf_poisson": (("obs", "bin"), np.asarray(pdf_refs)),
            "relative_pdf": (("obs", "bin"), np.asarray(rels)),
            "cluster_threshold_area_norm": ("obs", np.asarray(thresholds, dtype=float)),
            "cluster_threshold_log_area_norm": ("obs", np.asarray(thresholds_log, dtype=float)),
            "clustered_fraction": ("obs", np.asarray(clustered_fraction, dtype=float)),
            "sigma_area_norm": ("obs", np.asarray(sigma_area, dtype=float)),
            "sigma_log_area_norm": ("obs", np.asarray(sigma_log, dtype=float)),
        },
        coords={
            "obs": np.asarray(obs_out, dtype=int),
            "bin": np.arange(len(centers_out), dtype=int),
            coord_name: ("bin", centers_out),
            "bin_left": ("bin", edges_out[:-1]),
            "bin_right": ("bin", edges_out[1:]),
        },
        attrs={
            "description": "Voronoi area PDF and 2D Poisson-Voronoi reference.",
            "log_area": bool(log_area),
            "cluster_threshold_method": "first low-area PDF intersection with 2D Poisson-Voronoi reference",
            "poisson_area_norm_std_2d": 0.52,
        },
    )
    return ds_out


def compute_voronoi_clustering_timeseries(
    ds_traj: xr.Dataset,
    obs_indices: Iterable[int] | None,
    domain: PeriodicDomain,
    *,
    bins: np.ndarray | None = None,
    log_area: bool = True,
    threshold_method: str = "pdf_intersection",
    fixed_threshold_area_norm: float = 0.5,
) -> xr.Dataset:
    """
    Compute particle-level Voronoï area, normalized area, and cluster flags.

    threshold_method:
        "pdf_intersection" uses the Monchaux-style first PDF intersection.
        "fixed" uses fixed_threshold_area_norm.
    """
    xname, yname = get_xy_names(ds_traj)
    n_particles = ds_traj.sizes.get("trajectory", ds_traj.sizes.get("particle"))
    n_obs_total = ds_traj.sizes["obs"]

    if obs_indices is None:
        obs_indices = range(n_obs_total)
    obs_indices = list(obs_indices)

    areas = np.full((n_particles, len(obs_indices)), np.nan, dtype=float)
    area_norm = np.full_like(areas, np.nan)
    cluster_flag = np.zeros_like(areas, dtype=bool)
    thresholds = np.full(len(obs_indices), np.nan, dtype=float)

    for j, obs_idx in enumerate(obs_indices):
        x = ds_traj[xname].isel(obs=obs_idx).values
        y = ds_traj[yname].isel(obs=obs_idx).values

        result = periodic_voronoi_snapshot(x, y, domain)
        a = np.asarray(result["cell_area"], dtype=float)
        valid = np.isfinite(a) & (a > 0.0)

        areas[:, j] = a

        mean_a = np.nanmean(a[valid]) if np.any(valid) else np.nan
        if np.isfinite(mean_a) and mean_a > 0.0:
            area_norm[:, j] = a / mean_a

        if threshold_method == "pdf_intersection":
            pdf_out = voronoi_area_pdf_for_snapshot(a, bins=bins, log_area=log_area)
            thr = pdf_out["cluster_threshold_area_norm"]
            if not np.isfinite(thr):
                thr = fixed_threshold_area_norm
        elif threshold_method == "fixed":
            thr = fixed_threshold_area_norm
        else:
            raise ValueError("threshold_method must be 'pdf_intersection' or 'fixed'.")

        thresholds[j] = thr
        cluster_flag[:, j] = np.isfinite(area_norm[:, j]) & (area_norm[:, j] < thr)

    return xr.Dataset(
        data_vars={
            "voronoi_area": (("trajectory", "obs"), areas),
            "voronoi_area_norm": (("trajectory", "obs"), area_norm),
            "cluster_flag": (("trajectory", "obs"), cluster_flag.astype(np.int8)),
            "cluster_threshold_area_norm": ("obs", thresholds),
        },
        coords={
            "trajectory": np.arange(n_particles, dtype=int),
            "obs": np.asarray(obs_indices, dtype=int),
        },
        attrs={
            "description": "Particle-level Voronoi clustering diagnostics.",
            "threshold_method": threshold_method,
            "fixed_threshold_area_norm": float(fixed_threshold_area_norm),
            "cluster_flag_definition": "voronoi_area_norm < cluster_threshold_area_norm",
        },
    )


def _obs_time_days(ds_traj: xr.Dataset, obs_indices: Iterable[int]) -> np.ndarray:
    obs_indices = list(obs_indices)
    if "time" not in ds_traj:
        return np.asarray(obs_indices, dtype=float)

    t = ds_traj["time"].isel(trajectory=0, obs=obs_indices).values
    t0 = ds_traj["time"].isel(trajectory=0, obs=obs_indices[0]).values

    if np.issubdtype(np.asarray(t).dtype, np.datetime64):
        return ((t - t0) / np.timedelta64(1, "s")).astype(float) / 86400.0

    return (np.asarray(t, dtype=float) - float(t0)) / 86400.0


def compute_cluster_residence_times(
    cluster_ds: xr.Dataset,
    ds_traj: xr.Dataset | None = None,
    *,
    min_duration_obs: int = 1,
) -> xr.Dataset:
    """
    Convert a cluster-flag timeseries into residence-time events.

    A residence event is one consecutive period during which one particle is
    classified as clustered. Durations are reported both in observation counts
    and days. If ds_traj is omitted, durations are expressed in obs counts only
    and days are set equal to counts.
    """
    flags = np.asarray(cluster_ds["cluster_flag"].values, dtype=bool)
    obs = np.asarray(cluster_ds["obs"].values, dtype=int)

    if ds_traj is not None:
        t_days = _obs_time_days(ds_traj, obs)
    else:
        t_days = np.arange(len(obs), dtype=float)

    events = []

    for traj_idx in range(flags.shape[0]):
        f = flags[traj_idx]
        i = 0
        while i < len(f):
            if not f[i]:
                i += 1
                continue

            start = i
            while i + 1 < len(f) and f[i + 1]:
                i += 1
            end = i

            duration_obs = end - start + 1
            if duration_obs >= min_duration_obs:
                if len(t_days) > 1:
                    # Include one output interval after the end index when possible.
                    dt_mean = float(np.nanmedian(np.diff(t_days)))
                else:
                    dt_mean = 1.0
                duration_days = duration_obs * dt_mean
                events.append((
                    int(traj_idx),
                    int(obs[start]),
                    int(obs[end]),
                    int(duration_obs),
                    float(duration_days),
                    float(t_days[start]),
                    float(t_days[end]),
                ))

            i += 1

    if len(events) == 0:
        data = np.empty((0, 7), dtype=float)
    else:
        data = np.asarray(events, dtype=float)

    return xr.Dataset(
        data_vars={
            "trajectory": ("event", data[:, 0].astype(int) if data.size else np.array([], dtype=int)),
            "start_obs": ("event", data[:, 1].astype(int) if data.size else np.array([], dtype=int)),
            "end_obs": ("event", data[:, 2].astype(int) if data.size else np.array([], dtype=int)),
            "duration_obs": ("event", data[:, 3].astype(int) if data.size else np.array([], dtype=int)),
            "duration_days": ("event", data[:, 4] if data.size else np.array([], dtype=float)),
            "start_day": ("event", data[:, 5] if data.size else np.array([], dtype=float)),
            "end_day": ("event", data[:, 6] if data.size else np.array([], dtype=float)),
        },
        coords={"event": np.arange(len(events), dtype=int)},
        attrs={
            "description": "Residence-time events for consecutive clustered periods.",
            "min_duration_obs": int(min_duration_obs),
        },
    )


def residence_summary(residence_ds: xr.Dataset) -> xr.Dataset:
    """Compact summary statistics of cluster residence-time events."""
    if "duration_days" not in residence_ds or residence_ds.sizes.get("event", 0) == 0:
        vals = np.array([], dtype=float)
    else:
        vals = np.asarray(residence_ds["duration_days"].values, dtype=float)
        vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        stats = {"n_events": 0, "mean_days": np.nan, "median_days": np.nan, "p95_days": np.nan, "max_days": np.nan}
    else:
        stats = {
            "n_events": int(vals.size),
            "mean_days": float(np.nanmean(vals)),
            "median_days": float(np.nanmedian(vals)),
            "p95_days": float(np.nanpercentile(vals, 95)),
            "max_days": float(np.nanmax(vals)),
        }

    return xr.Dataset({k: xr.DataArray(v) for k, v in stats.items()})


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
    """Plot a Voronoï snapshot on an existing axes."""
    scale = 1000.0 if km else 1.0
    segments = []

    for piece in pieces:
        poly = piece["polygon"]
        if poly is None or len(poly) < 3:
            continue
        poly_plot = poly / scale
        closed = np.vstack([poly_plot, poly_plot[0]])
        segments.extend([[closed[i], closed[i + 1]] for i in range(len(closed) - 1)])

    lc = LineCollection(segments, colors=line_color, linewidths=line_width, alpha=line_alpha)
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


def plot_voronoi_pdf(
    pdf_ds: xr.Dataset,
    *,
    obs_idx: int | None = None,
    ax=None,
    label: str | None = None,
    color=None,
    title: str | None = None,
    show_poisson: bool = True,
):
    """Plot a Voronoï PDF dataset returned by compute_voronoi_pdf()."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4), facecolor="white")

    if obs_idx is None:
        obs_idx = int(pdf_ds["obs"].values[0])

    row = pdf_ds.sel(obs=obs_idx)
    xcoord = "log_area_norm" if "log_area_norm" in pdf_ds.coords else "area_norm"
    x = row[xcoord].values

    ax.plot(x, row["pdf"].values, lw=1.8, label=label or "particles", color=color)

    if show_poisson:
        ax.plot(x, row["pdf_poisson"].values, lw=1.4, ls="--", color="0.25", label="Poisson")

    thr_log = float(row["cluster_threshold_log_area_norm"].values)
    if np.isfinite(thr_log) and xcoord == "log_area_norm":
        ax.axvline(thr_log, color="0.35", lw=1.0, ls=":")
    elif xcoord == "area_norm":
        thr = float(row["cluster_threshold_area_norm"].values)
        if np.isfinite(thr):
            ax.axvline(thr, color="0.35", lw=1.0, ls=":")

    ax.set_yscale("log")
    ax.set_xlabel(r"$\ln(A/\langle A\rangle)$" if xcoord == "log_area_norm" else r"$A/\langle A\rangle$")
    ax.set_ylabel("PDF")
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_residence_time_pdf(
    residence_ds: xr.Dataset,
    *,
    ax=None,
    bins: int | np.ndarray = 30,
    label: str | None = None,
    color=None,
    title: str | None = None,
):
    """Plot residence-time distribution from compute_cluster_residence_times()."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4), facecolor="white")

    if residence_ds.sizes.get("event", 0) == 0:
        ax.text(0.5, 0.5, "No residence events", ha="center", va="center", transform=ax.transAxes)
        return ax

    vals = np.asarray(residence_ds["duration_days"].values, dtype=float)
    vals = vals[np.isfinite(vals)]

    ax.hist(vals, bins=bins, density=True, histtype="step", lw=1.8, label=label, color=color)
    ax.set_xlabel("cluster residence time [days]")
    ax.set_ylabel("PDF")
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax





# ============================================================
# CLUSTERED-STATE PERSISTENCE + LAGRANGIAN AUTOCORRELATION
# ============================================================

def _bridge_short_false_gaps_1d(
    flags: np.ndarray,
    max_gap_obs: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bridge short internal False gaps in a Boolean clustered-state series.

    Only gaps bounded by True values on both sides are filled. Leading and
    trailing False values are never changed.

    Returns
    -------
    tolerant_flags
        Clustered-state flags after bridging eligible gaps.
    gap_filled
        Boolean mask identifying observations changed from False to True.
    """
    flags = np.asarray(flags, dtype=bool)
    tolerant = flags.copy()
    gap_filled = np.zeros_like(flags, dtype=bool)

    max_gap_obs = int(max_gap_obs)
    if max_gap_obs < 0:
        raise ValueError("max_gap_obs must be >= 0.")
    if max_gap_obs == 0 or flags.size < 3:
        return tolerant, gap_filled

    i = 0
    n = flags.size
    while i < n:
        if flags[i]:
            i += 1
            continue

        gap_start = i
        while i < n and not flags[i]:
            i += 1
        gap_end = i - 1
        gap_length = gap_end - gap_start + 1

        bounded_left = gap_start > 0 and flags[gap_start - 1]
        bounded_right = i < n and flags[i]

        if bounded_left and bounded_right and gap_length <= max_gap_obs:
            tolerant[gap_start:i] = True
            gap_filled[gap_start:i] = True

    return tolerant, gap_filled


def apply_clustered_state_gap_tolerance(
    cluster_ds: xr.Dataset,
    *,
    max_unclustered_gap_obs: int = 1,
) -> xr.Dataset:
    """
    Add tolerant clustered-state flags to a Voronoï clustering time series.

    A short internal unclustered gap is bridged when it is bounded by clustered
    observations on both sides. With max_unclustered_gap_obs=1, the sequence

        True, False, True

    is treated as one continuous clustered-state episode.

    The original ``cluster_flag`` is retained. The returned dataset adds:

        cluster_flag_tolerant(trajectory, obs)
        gap_filled_flag(trajectory, obs)
    """
    if "cluster_flag" not in cluster_ds:
        raise KeyError("cluster_ds must contain 'cluster_flag'.")

    raw = np.asarray(cluster_ds["cluster_flag"].values, dtype=bool)
    tolerant = np.zeros_like(raw, dtype=bool)
    filled = np.zeros_like(raw, dtype=bool)

    for traj_idx in range(raw.shape[0]):
        tolerant[traj_idx], filled[traj_idx] = _bridge_short_false_gaps_1d(
            raw[traj_idx],
            max_gap_obs=max_unclustered_gap_obs,
        )

    out = cluster_ds.copy()
    out["cluster_flag_tolerant"] = xr.DataArray(
        tolerant.astype(np.int8),
        dims=cluster_ds["cluster_flag"].dims,
        coords=cluster_ds["cluster_flag"].coords,
        attrs={
            "description": "Clustered-state flag after bridging short internal unclustered gaps.",
            "max_unclustered_gap_obs": int(max_unclustered_gap_obs),
        },
    )
    out["gap_filled_flag"] = xr.DataArray(
        filled.astype(np.int8),
        dims=cluster_ds["cluster_flag"].dims,
        coords=cluster_ds["cluster_flag"].coords,
        attrs={
            "description": "1 where an unclustered observation was bridged by the tolerance rule.",
        },
    )
    out.attrs["max_unclustered_gap_obs"] = int(max_unclustered_gap_obs)
    out.attrs["tolerant_cluster_flag_definition"] = (
        "cluster_flag with internal False gaps of length <= "
        f"{int(max_unclustered_gap_obs)} bridged when bounded by True values"
    )
    return out


def compute_clustered_state_residence_times(
    cluster_ds: xr.Dataset,
    ds_traj: xr.Dataset | None = None,
    *,
    min_duration_obs: int = 1,
    flag_name: str | None = None,
) -> xr.Dataset:
    """
    Compute particle clustered-state residence episodes.

    This is an individual-particle metric: an episode is a continuous interval
    during which a particle is classified as locally clustered according to its
    normalized Voronoï-cell area. It is not the lifetime of a tracked spatial
    cluster object.

    If ``cluster_flag_tolerant`` is present, it is used by default; otherwise
    ``cluster_flag`` is used. The event duration is the occupied output-frame
    span, ``duration_obs * median_output_interval``. When a short unclustered
    gap was bridged, that frame is included in the episode span and reported
    separately through ``gap_obs_count``.
    """
    if flag_name is None:
        flag_name = (
            "cluster_flag_tolerant"
            if "cluster_flag_tolerant" in cluster_ds
            else "cluster_flag"
        )

    if flag_name not in cluster_ds:
        raise KeyError(f"cluster_ds does not contain {flag_name!r}.")

    tolerant_flags = np.asarray(cluster_ds[flag_name].values, dtype=bool)
    raw_flags = np.asarray(
        cluster_ds.get("cluster_flag", cluster_ds[flag_name]).values,
        dtype=bool,
    )
    obs = np.asarray(cluster_ds["obs"].values, dtype=int)

    if ds_traj is not None:
        t_days = _obs_time_days(ds_traj, obs)
    else:
        t_days = np.arange(len(obs), dtype=float)

    if len(t_days) > 1:
        dt_days = float(np.nanmedian(np.diff(t_days)))
    else:
        dt_days = 1.0

    if not np.isfinite(dt_days) or dt_days <= 0.0:
        raise ValueError("Could not determine a positive trajectory output interval.")

    events: list[tuple] = []

    for traj_idx in range(tolerant_flags.shape[0]):
        f = tolerant_flags[traj_idx]
        raw = raw_flags[traj_idx]
        i = 0

        while i < len(f):
            if not f[i]:
                i += 1
                continue

            start = i
            while i + 1 < len(f) and f[i + 1]:
                i += 1
            end = i

            duration_obs = end - start + 1
            if duration_obs >= int(min_duration_obs):
                clustered_obs_count = int(np.count_nonzero(raw[start:end + 1]))
                gap_obs_count = int(duration_obs - clustered_obs_count)
                duration_days = float(duration_obs * dt_days)

                left_censored = start == 0
                right_censored = end == len(f) - 1

                events.append(
                    (
                        int(traj_idx),
                        int(obs[start]),
                        int(obs[end]),
                        int(duration_obs),
                        clustered_obs_count,
                        gap_obs_count,
                        duration_days,
                        float(t_days[start]),
                        float(t_days[end]),
                        int(left_censored),
                        int(right_censored),
                    )
                )

            i += 1

    if len(events) == 0:
        data = np.empty((0, 11), dtype=float)
    else:
        data = np.asarray(events, dtype=float)

    return xr.Dataset(
        data_vars={
            "trajectory": ("event", data[:, 0].astype(int) if data.size else np.array([], dtype=int)),
            "start_obs": ("event", data[:, 1].astype(int) if data.size else np.array([], dtype=int)),
            "end_obs": ("event", data[:, 2].astype(int) if data.size else np.array([], dtype=int)),
            "duration_obs": ("event", data[:, 3].astype(int) if data.size else np.array([], dtype=int)),
            "clustered_obs_count": ("event", data[:, 4].astype(int) if data.size else np.array([], dtype=int)),
            "gap_obs_count": ("event", data[:, 5].astype(int) if data.size else np.array([], dtype=int)),
            "duration_days": ("event", data[:, 6] if data.size else np.array([], dtype=float)),
            "start_day": ("event", data[:, 7] if data.size else np.array([], dtype=float)),
            "end_day": ("event", data[:, 8] if data.size else np.array([], dtype=float)),
            "left_censored": ("event", data[:, 9].astype(np.int8) if data.size else np.array([], dtype=np.int8)),
            "right_censored": ("event", data[:, 10].astype(np.int8) if data.size else np.array([], dtype=np.int8)),
        },
        coords={"event": np.arange(len(events), dtype=int)},
        attrs={
            "description": "Individual-particle clustered-state residence episodes based on Voronoi area.",
            "metric_name": "particle clustered-state residence time",
            "flag_name": flag_name,
            "min_duration_obs": int(min_duration_obs),
            "output_interval_days": float(dt_days),
            "duration_definition": "duration_obs * median trajectory output interval",
            "interpretation": (
                "Consecutive time an individual particle remains locally clustered; "
                "not the lifetime of a tracked spatial cluster object."
            ),
        },
    )


def clustered_state_residence_summary(
    residence_ds: xr.Dataset,
    *,
    complete_events_only: bool = False,
) -> xr.Dataset:
    """Compact statistics of particle clustered-state residence episodes."""
    n_total = int(residence_ds.sizes.get("event", 0))

    if n_total == 0 or "duration_days" not in residence_ds:
        vals = np.array([], dtype=float)
        complete_mask = np.array([], dtype=bool)
    else:
        vals_all = np.asarray(residence_ds["duration_days"].values, dtype=float)
        left = np.asarray(
            residence_ds.get("left_censored", xr.zeros_like(residence_ds["duration_days"])).values,
            dtype=bool,
        )
        right = np.asarray(
            residence_ds.get("right_censored", xr.zeros_like(residence_ds["duration_days"])).values,
            dtype=bool,
        )
        complete_mask = ~(left | right)
        use_mask = complete_mask if complete_events_only else np.ones_like(complete_mask, dtype=bool)
        vals = vals_all[use_mask & np.isfinite(vals_all)]

    n_complete = int(np.count_nonzero(complete_mask))
    n_censored = int(n_total - n_complete)

    if vals.size == 0:
        stats = {
            "n_events": n_total,
            "n_complete_events": n_complete,
            "n_censored_events": n_censored,
            "mean_days": np.nan,
            "median_days": np.nan,
            "p95_days": np.nan,
            "max_days": np.nan,
        }
    else:
        stats = {
            "n_events": n_total,
            "n_complete_events": n_complete,
            "n_censored_events": n_censored,
            "mean_days": float(np.nanmean(vals)),
            "median_days": float(np.nanmedian(vals)),
            "p95_days": float(np.nanpercentile(vals, 95)),
            "max_days": float(np.nanmax(vals)),
        }

    out = xr.Dataset({k: xr.DataArray(v) for k, v in stats.items()})
    out.attrs.update(
        {
            "description": "Summary of particle clustered-state residence episodes.",
            "complete_events_only": int(bool(complete_events_only)),
        }
    )
    return out


def _first_crossing_time(x: np.ndarray, y: np.ndarray, level: float) -> float:
    """Linearly interpolate the first downward crossing of ``level``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    for i in range(1, len(y)):
        if not (np.isfinite(y[i - 1]) and np.isfinite(y[i])):
            continue
        if y[i - 1] >= level and y[i] <= level:
            dy = y[i] - y[i - 1]
            if dy == 0.0:
                return float(x[i])
            frac = (level - y[i - 1]) / dy
            return float(x[i - 1] + frac * (x[i] - x[i - 1]))

    return np.nan


def compute_lagrangian_voronoi_concentration_autocorrelation(
    cluster_ds: xr.Dataset,
    ds_traj: xr.Dataset | None = None,
    *,
    max_lag_obs: int | None = None,
    min_pairs: int = 100,
    fit_min_rho: float = 0.2,
    fit_max_rho: float = 0.9,
) -> xr.Dataset:
    r"""
    Compute the Lagrangian autocorrelation of Voronoï concentration.

    Following Li et al. (2025), the local particle concentration is

        c_i(t) = 1 / A_i(t),

    where A_i is the Voronoï-cell area of particle i. The autocorrelation is

        rho_c(tau) = <c'(t)c'(t+tau)> / <c'^2>,

    with c' obtained by subtracting the ensemble-time mean concentration.
    A characteristic clustering-persistence timescale is obtained by fitting

        rho_c(tau) = exp(-tau / T_c)

    over the positive decaying range selected by fit_min_rho and fit_max_rho.
    The fit is constrained through rho_c(0)=1.
    """
    if "voronoi_area" not in cluster_ds:
        raise KeyError("cluster_ds must contain 'voronoi_area'.")

    area = np.asarray(cluster_ds["voronoi_area"].values, dtype=float)
    concentration = np.full_like(area, np.nan, dtype=float)
    valid_area = np.isfinite(area) & (area > 0.0)
    concentration[valid_area] = 1.0 / area[valid_area]

    mean_c = float(np.nanmean(concentration))
    fluct = concentration - mean_c
    variance = float(np.nanmean(fluct**2))

    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError("Voronoï concentration variance is not positive.")

    obs = np.asarray(cluster_ds["obs"].values, dtype=int)
    n_obs = len(obs)

    if ds_traj is not None:
        t_days = _obs_time_days(ds_traj, obs)
    else:
        t_days = np.arange(n_obs, dtype=float)

    if n_obs > 1:
        dt_days = float(np.nanmedian(np.diff(t_days)))
    else:
        dt_days = 1.0

    if max_lag_obs is None:
        max_lag_obs = max(1, n_obs // 2)
    max_lag_obs = int(min(max_lag_obs, n_obs - 1))

    lags = np.arange(max_lag_obs + 1, dtype=int)
    rho = np.full(max_lag_obs + 1, np.nan, dtype=float)
    n_pairs = np.zeros(max_lag_obs + 1, dtype=np.int64)

    for lag in lags:
        if lag == 0:
            a = fluct
            b = fluct
        else:
            a = fluct[:, :-lag]
            b = fluct[:, lag:]

        pair_valid = np.isfinite(a) & np.isfinite(b)
        n_pairs[lag] = int(np.count_nonzero(pair_valid))

        if n_pairs[lag] >= int(min_pairs):
            rho[lag] = float(np.nanmean((a * b)[pair_valid]) / variance)

    if np.isfinite(rho[0]):
        rho[0] = 1.0

    lag_days = lags.astype(float) * dt_days

    # Restrict fitting to the positive branch before the first non-positive lag.
    positive_end = len(rho)
    nonpositive = np.flatnonzero((lags > 0) & np.isfinite(rho) & (rho <= 0.0))
    if nonpositive.size:
        positive_end = int(nonpositive[0])

    fit_mask = (
        (lags > 0)
        & (lags < positive_end)
        & np.isfinite(rho)
        & (rho > 0.0)
        & (rho >= float(fit_min_rho))
        & (rho <= float(fit_max_rho))
        & (n_pairs >= int(min_pairs))
    )

    timescale_fit_days = np.nan
    fit_r2 = np.nan
    fitted_rho = np.full_like(rho, np.nan, dtype=float)

    if np.count_nonzero(fit_mask) >= 2:
        x = lag_days[fit_mask]
        y = np.log(rho[fit_mask])

        # Least-squares fit through the origin: log(rho) = -tau/T_c.
        denom = float(np.dot(x, x))
        slope = float(np.dot(x, y) / denom) if denom > 0.0 else np.nan

        if np.isfinite(slope) and slope < 0.0:
            timescale_fit_days = float(-1.0 / slope)
            fitted_rho = np.exp(-lag_days / timescale_fit_days)

            y_pred = slope * x
            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            fit_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else np.nan

    timescale_efold_days = _first_crossing_time(
        lag_days,
        rho,
        np.exp(-1.0),
    )

    # Positive-lobe integral timescale, truncated at first non-positive value.
    positive_mask = np.isfinite(rho[:positive_end]) & (rho[:positive_end] > 0.0)
    if np.count_nonzero(positive_mask) >= 2:
        timescale_integral_days = float(
            np.trapezoid(rho[:positive_end][positive_mask], lag_days[:positive_end][positive_mask])
        )
    else:
        timescale_integral_days = np.nan

    return xr.Dataset(
        data_vars={
            "autocorrelation": ("lag", rho),
            "exponential_fit": ("lag", fitted_rho),
            "n_pairs": ("lag", n_pairs),
            "timescale_expfit_days": xr.DataArray(timescale_fit_days),
            "timescale_efold_days": xr.DataArray(timescale_efold_days),
            "timescale_integral_days": xr.DataArray(timescale_integral_days),
            "fit_r2": xr.DataArray(fit_r2),
            "mean_concentration": xr.DataArray(mean_c),
            "concentration_variance": xr.DataArray(variance),
        },
        coords={
            "lag": lags,
            "lag_days": ("lag", lag_days),
        },
        attrs={
            "description": "Lagrangian autocorrelation of inverse Voronoi-cell area.",
            "metric_name": "Voronoi-concentration clustering-persistence timescale",
            "concentration_definition": "c = 1 / voronoi_area",
            "autocorrelation_definition": "<c'(t)c'(t+tau)> / <c'^2>",
            "timescale_definition": "rho(tau)=exp(-tau/Tc), fitted through rho(0)=1",
            "output_interval_days": float(dt_days),
            "max_lag_obs": int(max_lag_obs),
            "min_pairs": int(min_pairs),
            "fit_min_rho": float(fit_min_rho),
            "fit_max_rho": float(fit_max_rho),
        },
    )


def plot_clustered_state_residence_pdf(
    residence_ds: xr.Dataset,
    *,
    ax=None,
    bins: int | np.ndarray = 30,
    label: str | None = None,
    color=None,
    title: str | None = None,
    complete_events_only: bool = False,
):
    """Plot the PDF of particle clustered-state residence durations."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4), facecolor="white")

    if residence_ds.sizes.get("event", 0) == 0:
        ax.text(0.5, 0.5, "No clustered-state episodes", ha="center", va="center", transform=ax.transAxes)
        return ax

    vals = np.asarray(residence_ds["duration_days"].values, dtype=float)
    

    if complete_events_only and "left_censored" in residence_ds and "right_censored" in residence_ds:
        left = np.asarray(residence_ds["left_censored"].values, dtype=bool)
        right = np.asarray(residence_ds["right_censored"].values, dtype=bool)
        vals = vals[~(left | right)]

    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        ax.text(0.5, 0.5, "No complete clustered-state episodes", ha="center", va="center", transform=ax.transAxes)
        return ax
    
    weights = np.ones_like(vals) / len(vals)

    ax.hist(vals, bins=bins, weights=weights, histtype="step", linewidth=2.0, label=label)

    ax.set_xlabel("Residence time [days]")
    ax.set_ylabel("Fraction of clustering events")   
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_lagrangian_voronoi_concentration_autocorrelation(
    autocorr_ds: xr.Dataset,
    *,
    ax=None,
    label: str | None = None,
    color=None,
    title: str | None = None,
    show_fit: bool = True,
):
    """Plot the Lagrangian autocorrelation of c=1/A_V and its exponential fit."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4), facecolor="white")

    lag_days = np.asarray(autocorr_ds["lag_days"].values, dtype=float)
    rho = np.asarray(autocorr_ds["autocorrelation"].values, dtype=float)

    ax.plot(lag_days, rho, lw=1.8, color=color, label=label)

    if show_fit and "exponential_fit" in autocorr_ds:
        fit = np.asarray(autocorr_ds["exponential_fit"].values, dtype=float)
        if np.any(np.isfinite(fit)):
            fit_label = None
            tc = float(autocorr_ds["timescale_expfit_days"].values)
            if label is not None and np.isfinite(tc):
                fit_label = rf"{label} fit: $T_c={tc:.2f}$ d"
            ax.plot(lag_days, fit, lw=1.2, ls="--", color=color, alpha=0.8, label=fit_label)

    ax.axhline(0.0, color="0.35", lw=0.8)
    ax.axhline(np.exp(-1.0), color="0.5", lw=0.8, ls=":")
    ax.set_xlabel(r"lag $\tau$ [days]")
    ax.set_ylabel(r"$\rho_c^L(\tau)$")
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


# Backward-compatible aliases for older notebook cells.
compute_cluster_residence_times = compute_clustered_state_residence_times
residence_summary = clustered_state_residence_summary
plot_residence_time_pdf = plot_clustered_state_residence_pdf


# ============================================================
# PARTICLE-LEVEL RESIDENCE LABELS FOR FLOW COUPLING
# ============================================================

def compute_cluster_residence_labels(
    cluster_ds: xr.Dataset,
    ds_traj: xr.Dataset | None = None,
    *,
    min_duration_obs: int = 1,
    noncluster_event_id: int = -1,
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Compute particle-level residence labels from a cluster-flag time series.

    This is the recommended residence-time product for coupling particle
    clustering to flow diagnostics. It returns both:

    1. labels_ds with dimensions (trajectory, obs), suitable for merging with
       sampled flow diagnostics;
    2. events_ds with one row per cluster-residence event.

    Definitions
    -----------
    A cluster-residence event is one consecutive period during which one
    particle has cluster_flag=True. For each clustered particle snapshot:

    current_residence_time_days
        Time already spent inside the current event up to this snapshot.

    final_residence_time_days
        Total duration of the event to which this snapshot belongs. This is
        only known after the full trajectory has been analysed.

    Parameters
    ----------
    cluster_ds
        Dataset returned by compute_voronoi_clustering_timeseries(). Must
        contain cluster_flag(trajectory, obs).
    ds_traj
        Optional trajectory dataset with time(trajectory, obs). If omitted,
        durations are expressed in observation intervals and converted to days
        using one obs = one day.
    min_duration_obs
        Minimum event length in number of saved observations. Shorter events
        are left unlabelled.
    noncluster_event_id
        Event ID assigned outside clusters.
    """
    if "cluster_flag" not in cluster_ds:
        raise KeyError("cluster_ds must contain 'cluster_flag'.")

    flags = np.asarray(cluster_ds["cluster_flag"].values, dtype=bool)
    obs_vals = np.asarray(cluster_ds["obs"].values, dtype=int)

    if flags.ndim != 2:
        raise ValueError("cluster_flag must have dimensions (trajectory, obs).")

    n_traj, n_obs = flags.shape

    if ds_traj is not None:
        t_days = _obs_time_days(ds_traj, obs_vals)
    else:
        t_days = np.arange(n_obs, dtype=float)

    if n_obs > 1:
        dt_days = float(np.nanmedian(np.diff(t_days)))
        if not np.isfinite(dt_days) or dt_days <= 0.0:
            dt_days = 1.0
    else:
        dt_days = 1.0

    event_id = np.full((n_traj, n_obs), noncluster_event_id, dtype=np.int32)
    current_residence_obs = np.zeros((n_traj, n_obs), dtype=np.int32)
    final_residence_obs = np.zeros((n_traj, n_obs), dtype=np.int32)
    current_residence_days = np.zeros((n_traj, n_obs), dtype=np.float32)
    final_residence_days = np.zeros((n_traj, n_obs), dtype=np.float32)

    events = []
    eid = 0

    for traj_idx in range(n_traj):
        f = flags[traj_idx]
        i = 0
        while i < n_obs:
            if not f[i]:
                i += 1
                continue

            start = i
            while i + 1 < n_obs and f[i + 1]:
                i += 1
            end = i

            duration_obs = end - start + 1

            if duration_obs >= int(min_duration_obs):
                duration_days = float(duration_obs * dt_days)
                local = np.arange(1, duration_obs + 1, dtype=np.int32)

                event_id[traj_idx, start:end + 1] = eid
                current_residence_obs[traj_idx, start:end + 1] = local
                final_residence_obs[traj_idx, start:end + 1] = duration_obs
                current_residence_days[traj_idx, start:end + 1] = local.astype(np.float32) * dt_days
                final_residence_days[traj_idx, start:end + 1] = duration_days

                events.append(
                    {
                        "event_id": eid,
                        "trajectory": int(traj_idx),
                        "start_obs": int(obs_vals[start]),
                        "end_obs": int(obs_vals[end]),
                        "duration_obs": int(duration_obs),
                        "duration_days": duration_days,
                        "start_day": float(t_days[start]),
                        "end_day": float(t_days[end]),
                    }
                )
                eid += 1

            i += 1

    labels_ds = xr.Dataset(
        data_vars={
            "cluster_event_id": (("trajectory", "obs"), event_id),
            "current_residence_obs": (("trajectory", "obs"), current_residence_obs),
            "final_residence_obs": (("trajectory", "obs"), final_residence_obs),
            "current_residence_time_days": (("trajectory", "obs"), current_residence_days),
            "final_residence_time_days": (("trajectory", "obs"), final_residence_days),
        },
        coords={
            "trajectory": np.asarray(cluster_ds["trajectory"].values, dtype=int),
            "obs": obs_vals,
        },
        attrs={
            "description": "Particle-level cluster residence labels for coupled flow-particle analysis.",
            "min_duration_obs": int(min_duration_obs),
            "noncluster_event_id": int(noncluster_event_id),
            "dt_days_assumed": float(dt_days),
            "current_residence_time_days": "time already spent in current cluster event at each saved observation",
            "final_residence_time_days": "total duration of the cluster event to which each saved observation belongs",
        },
    )

    if len(events) == 0:
        events_ds = xr.Dataset(
            data_vars={
                "trajectory": ("event", np.array([], dtype=np.int32)),
                "start_obs": ("event", np.array([], dtype=np.int32)),
                "end_obs": ("event", np.array([], dtype=np.int32)),
                "duration_obs": ("event", np.array([], dtype=np.int32)),
                "duration_days": ("event", np.array([], dtype=np.float32)),
                "start_day": ("event", np.array([], dtype=np.float32)),
                "end_day": ("event", np.array([], dtype=np.float32)),
            },
            coords={"event": np.array([], dtype=np.int32)},
        )
    else:
        events_ds = xr.Dataset(
            data_vars={
                "trajectory": ("event", np.asarray([e["trajectory"] for e in events], dtype=np.int32)),
                "start_obs": ("event", np.asarray([e["start_obs"] for e in events], dtype=np.int32)),
                "end_obs": ("event", np.asarray([e["end_obs"] for e in events], dtype=np.int32)),
                "duration_obs": ("event", np.asarray([e["duration_obs"] for e in events], dtype=np.int32)),
                "duration_days": ("event", np.asarray([e["duration_days"] for e in events], dtype=np.float32)),
                "start_day": ("event", np.asarray([e["start_day"] for e in events], dtype=np.float32)),
                "end_day": ("event", np.asarray([e["end_day"] for e in events], dtype=np.float32)),
            },
            coords={"event": np.asarray([e["event_id"] for e in events], dtype=np.int32)},
        )

    events_ds.attrs.update(
        {
            "description": "Cluster residence events derived from particle-level cluster flags.",
            "min_duration_obs": int(min_duration_obs),
            "dt_days_assumed": float(dt_days),
        }
    )

    return labels_ds, events_ds


def compute_clustering_products(
    ds_traj: xr.Dataset,
    obs_indices: Iterable[int] | None,
    domain: PeriodicDomain,
    *,
    bins: np.ndarray | None = None,
    threshold_method: str = "pdf_intersection",
    fixed_threshold_area_norm: float = 0.5,
    min_residence_duration_obs: int = 1,
) -> dict[str, xr.Dataset]:
    """
    Convenience wrapper for the clustering-only workflow.

    Returns a dictionary with:
        summary
        pdf
        particle_voronoi
        residence_labels
        residence_events
        residence_summary
    """
    if obs_indices is None:
        obs_indices = list(range(ds_traj.sizes["obs"]))
    else:
        obs_indices = list(obs_indices)

    summary = compute_voronoi_summary(ds_traj, obs_indices, domain)
    pdf = compute_voronoi_pdf(ds_traj, obs_indices, domain, bins=bins, log_area=True)
    particle_voronoi = compute_voronoi_clustering_timeseries(
        ds_traj,
        obs_indices,
        domain,
        bins=bins,
        log_area=True,
        threshold_method=threshold_method,
        fixed_threshold_area_norm=fixed_threshold_area_norm,
    )
    residence_labels, residence_events = compute_cluster_residence_labels(
        particle_voronoi,
        ds_traj,
        min_duration_obs=min_residence_duration_obs,
    )
    res_summary = residence_summary(residence_events)

    return {
        "summary": summary,
        "pdf": pdf,
        "particle_voronoi": particle_voronoi,
        "residence_labels": residence_labels,
        "residence_events": residence_events,
        "residence_summary": res_summary,
    }
