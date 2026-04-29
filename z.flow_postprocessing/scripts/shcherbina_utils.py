from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# BASIC DATA HELPERS
# ============================================================

def find_dim(dims, candidates):
    """
    Find a dimension name from a list of possible candidates.

    Useful for MITgcm MNC output where dimensions can be named
    e.g. T, iter, Z, Zmd000059, Xp1, Yp1, etc.
    """
    dims = list(dims)

    for c in candidates:
        if c in dims:
            return c

    for d in dims:
        if any(d.startswith(prefix) for prefix in candidates if isinstance(prefix, str)):
            return d

    for d in dims:
        dl = d.lower()
        if dl.startswith("zmd") or dl == "z":
            return d

    raise KeyError(f"Could not find any of {candidates} in dims={dims}")


def clean_fill(a, big=1e10):
    """
    Replace very large MITgcm fill values by NaN.
    """
    a = np.asarray(a, dtype=np.float32)
    a[np.abs(a) > big] = np.nan
    return a


def get_time_days(it, day_per_index=0.5):
    """
    Convert model output index to days.
    """
    return float(it) * float(day_per_index)


def format_threshold(val):
    """
    Clean threshold formatting for plot titles.
    """
    return f"{val:g}"


# ============================================================
# KINEMATIC DIAGNOSTICS
# ============================================================

def uv_to_center(u, v):
    """
    Convert staggered MITgcm U/V to common tracer-cell centers.

    Parameters
    ----------
    u : ndarray
        Shape (..., y, x_u)
    v : ndarray
        Shape (..., y_v, x)

    Returns
    -------
    u_c, v_c : ndarray
        Both cropped to common horizontal shape (..., y, x).
    """
    u = np.asarray(u)
    v = np.asarray(v)

    u_c = 0.5 * (u[..., :, :-1] + u[..., :, 1:])
    v_c = 0.5 * (v[..., :-1, :] + v[..., 1:, :])

    ny = min(u_c.shape[-2], v_c.shape[-2])
    nx = min(u_c.shape[-1], v_c.shape[-1])

    return u_c[..., :ny, :nx], v_c[..., :ny, :nx]


def gradients_uv(u_c, v_c, dx, dy):
    """
    Compute speed, relative vorticity, horizontal divergence,
    and strain magnitude from centered velocity components.

    Definitions:
        zeta   = dv/dx - du/dy
        div    = du/dx + dv/dy
        strain = sqrt((du/dx - dv/dy)^2 + (dv/dx + du/dy)^2)
    """
    u_c = np.asarray(u_c, dtype=np.float64)
    v_c = np.asarray(v_c, dtype=np.float64)

    du_dy, du_dx = np.gradient(u_c, dy, dx, axis=(-2, -1))
    dv_dy, dv_dx = np.gradient(v_c, dy, dx, axis=(-2, -1))

    zeta = dv_dx - du_dy
    div = du_dx + dv_dy

    sn = du_dx - dv_dy
    ss = dv_dx + du_dy
    strain = np.sqrt(sn**2 + ss**2)

    speed = np.hypot(u_c, v_c)

    return speed, zeta, div, strain


def compute_surface_kinematics(ds, time_dim, z_dim, it, levels, dx, dy, f0):
    """
    Compute Ro, div/f, and strain/f from UVEL/VVEL.

    If multiple vertical levels are provided, diagnostics are computed
    per layer first and then averaged. This avoids vertically averaging
    velocities before taking horizontal derivatives.
    """
    layer_results = []

    for k in levels:
        u = clean_fill(ds["UVEL"].isel({time_dim: it, z_dim: k}).values)
        v = clean_fill(ds["VVEL"].isel({time_dim: it, z_dim: k}).values)

        u_c, v_c = uv_to_center(u[None, ...], v[None, ...])
        u_c = u_c[0]
        v_c = v_c[0]

        _, zeta_k, div_k, strain_k = gradients_uv(u_c, v_c, dx, dy)

        layer_results.append({
            "ro": zeta_k / f0,
            "div_f": div_k / f0,
            "strain_f": strain_k / f0,
        })

    ro = np.nanmean(np.stack([d["ro"] for d in layer_results], axis=0), axis=0)
    div_f = np.nanmean(np.stack([d["div_f"] for d in layer_results], axis=0), axis=0)
    strain_f = np.nanmean(np.stack([d["strain_f"] for d in layer_results], axis=0), axis=0)

    return ro, div_f, strain_f


# ============================================================
# STATISTICS
# ============================================================

def normalize_hist(y):
    """
    Normalize histogram values by their maximum.
    """
    y = np.asarray(y, dtype=float)
    m = np.nanmax(y)
    if np.isfinite(m) and m > 0:
        return y / m
    return y


def nan_skew(a):
    """
    NaN-safe Pearson moment skewness.
    """
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]

    if a.size == 0:
        return np.nan

    mu = np.mean(a)
    sig = np.std(a)

    if sig == 0 or not np.isfinite(sig):
        return np.nan

    return np.mean((a - mu) ** 3) / sig ** 3


def stats_dict(a):
    """
    Summary statistics for flattened finite values.
    """
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]

    if a.size == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "skew": np.nan,
        }

    return {
        "mean": np.mean(a),
        "median": np.median(a),
        "std": np.std(a),
        "skew": nan_skew(a),
    }


# ============================================================
# DOMAIN SUBDIVISION / LOCAL PRo
# ============================================================

def build_square_index_ranges(ny, nx, nsy, nsx):
    """
    Divide a 2D domain into nsy x nsx rectangular/square-ish regions.

    Numbering starts bottom-left and proceeds row-wise:
        1, 2, 3, ...
    """
    y_edges = np.linspace(0, ny, nsy + 1, dtype=int)
    x_edges = np.linspace(0, nx, nsx + 1, dtype=int)

    squares = []
    sq_id = 1

    for jy in range(nsy):
        for ix in range(nsx):
            y0, y1 = y_edges[jy], y_edges[jy + 1]
            x0, x1 = x_edges[ix], x_edges[ix + 1]

            squares.append({
                "square_id": sq_id,
                "ix": ix,
                "jy": jy,
                "x0": x0,
                "x1": x1,
                "y0": y0,
                "y1": y1,
            })

            sq_id += 1

    return squares, x_edges, y_edges


def compute_pro_per_square(ro, squares, threshold):
    """
    Compute PRo = area fraction where |Ro| > threshold for each square.
    """
    vals = []

    for sq in squares:
        sub = ro[sq["y0"]:sq["y1"], sq["x0"]:sq["x1"]]
        finite = np.isfinite(sub)

        if np.any(finite):
            pro = np.mean(np.abs(sub[finite]) > threshold)
        else:
            pro = np.nan

        vals.append({
            "square_id": sq["square_id"],
            "PRo": float(pro),
        })

    return vals


# ============================================================
# PLOTTING HELPERS
# ============================================================

def savefig_if_needed(fig, filename_base, save=False, out_dir=None, dpi=None):
    """
    Save figure if save=True.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    filename_base : str
        Filename without extension.
    save : bool
    out_dir : path-like
    dpi : int or None
    """
    if not save:
        return None

    if out_dir is None:
        raise ValueError("out_dir must be provided when save=True.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if dpi is None:
        dpi = plt.rcParams["figure.dpi"]

    png_path = out_dir / f"{filename_base}.png"
    fig.canvas.draw()
    fig.savefig(png_path, bbox_inches="tight", dpi=dpi)
    print(f"Saved: {png_path}")

    return png_path


def plot_map(field, title="", cbar_label="", cmap=None, vmin=None, vmax=None,
             dx=500.0, dy=500.0, figsize=(7, 5)):
    """
    Map plot on a uniform Cartesian grid.

    Returns
    -------
    fig, ax
        Returning these explicitly avoids blank saved figures.
    """
    field = np.asarray(field)
    ny, nx = field.shape[-2], field.shape[-1]

    x_edges_km = np.arange(nx + 1) * dx / 1000.0
    y_edges_km = np.arange(ny + 1) * dy / 1000.0

    fig, ax = plt.subplots(figsize=figsize)

    pcm = ax.pcolormesh(
        x_edges_km,
        y_edges_km,
        field,
        shading="flat",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_title(title)
    ax.set_xlim(x_edges_km[0], x_edges_km[-1])
    ax.set_ylim(y_edges_km[0], y_edges_km[-1])

    fig.tight_layout()

    return fig, ax


def plot_pdf_panel(ax, data, bins, title, xlabel, color_line="red", show_zero_line=True):
    """
    Histogram bins + normalized PDF line + summary stats,
    in a Shcherbina-style presentation.
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]

    hist, edges = np.histogram(data, bins=bins, density=True)
    hist_n = normalize_hist(hist)

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    ax.bar(
        centers,
        hist_n,
        width=widths,
        align="center",
        facecolor="0.85",
        edgecolor="0.25",
        linewidth=0.7,
    )

    ax.plot(centers, hist_n, color=color_line, lw=1.3)

    if show_zero_line:
        ax.axvline(0.0, color="k", ls="--", lw=0.8, alpha=0.8)

    s = stats_dict(data)

    txt = (
        f"Mean    {s['mean']:6.2f}\n"
        f"Median  {s['median']:6.2f}\n"
        f"St. dev.{s['std']:6.2f}\n"
        f"Skew.   {s['skew']:6.2f}"
    )

    ax.text(
        0.98,
        0.98,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("PDF")
    ax.set_ylim(bottom=0)
    ax.grid(False)


def add_regime_labels(ax, x_edges, y_edges):
    """
    Add AVD / SD / CVD labels in the vorticity-strain plane.
    """
    x0, x1 = x_edges[0], x_edges[-1]
    y0, y1 = y_edges[0], y_edges[-1]

    xrng = x1 - x0
    yrng = y1 - y0

    ax.text(x0 + 0.10 * xrng, y0 + 0.10 * yrng, "AVD", fontsize=14)
    ax.text(x0 + 0.38 * xrng, y0 + 0.72 * yrng, "SD", fontsize=14)
    ax.text(x0 + 0.84 * xrng, y0 + 0.10 * yrng, "CVD", fontsize=14, ha="right")


def plot_square_overlay(ro, squares, dx, dy, title="", cmap="RdBu_r",
                        vmin=-2, vmax=2, figsize=(8, 6)):
    """
    Plot Rossby number with discretized square overlay and square IDs.
    """
    ro = np.asarray(ro)
    ny, nx = ro.shape

    x_edges_km = np.arange(nx + 1) * dx / 1000.0
    y_edges_km = np.arange(ny + 1) * dy / 1000.0

    fig, ax = plt.subplots(figsize=figsize)

    pcm = ax.pcolormesh(
        x_edges_km,
        y_edges_km,
        ro,
        shading="flat",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(r"$\zeta/f$")

    for sq in squares:
        x0_km = x_edges_km[sq["x0"]]
        x1_km = x_edges_km[sq["x1"]]
        y0_km = y_edges_km[sq["y0"]]
        y1_km = y_edges_km[sq["y1"]]

        rect = Rectangle(
            (x0_km, y0_km),
            x1_km - x0_km,
            y1_km - y0_km,
            fill=False,
            edgecolor="k",
            linewidth=0.8,
        )
        ax.add_patch(rect)

        xc = 0.5 * (x0_km + x1_km)
        yc = 0.5 * (y0_km + y1_km)

        ax.text(
            xc,
            yc,
            str(sq["square_id"]),
            ha="center",
            va="center",
            fontsize=8,
            color="k",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, pad=0.8),
        )

    ax.set_xlim(x_edges_km[0], x_edges_km[-1])
    ax.set_ylim(y_edges_km[0], y_edges_km[-1])
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_title(title)

    fig.tight_layout()

    return fig, ax


def crop_to_common_shape(*arrays):
    """
    Crop all arrays to the smallest common horizontal (..., y, x) shape.
    Useful when WVEL/THETA and velocity-gradient fields differ by one
    point because of MITgcm staggering.
    """
    if len(arrays) == 0:
        return []

    ny = min(np.asarray(a).shape[-2] for a in arrays)
    nx = min(np.asarray(a).shape[-1] for a in arrays)

    return [np.asarray(a)[..., :ny, :nx] for a in arrays]


def conditioned_mean_2d(x, y, z, x_edges, y_edges):
    """
    Compute conditional mean E[z | x-bin, y-bin].

    This is used for Balwada-style conditional means, e.g.
    mean vertical velocity conditioned on surface vorticity and strain.

    Returns
    -------
    mean_z : ndarray
        Shape (len(x_edges)-1, len(y_edges)-1), consistent with:
        pcolormesh(x_edges, y_edges, mean_z.T)
    count : ndarray
        Number of samples per bin.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

    x = x[mask]
    y = y[mask]
    z = z[mask]

    sum_z, _, _ = np.histogram2d(
        x,
        y,
        bins=[x_edges, y_edges],
        weights=z
    )

    count, _, _ = np.histogram2d(
        x,
        y,
        bins=[x_edges, y_edges]
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_z = sum_z / count

    mean_z[count == 0] = np.nan

    return mean_z, count



# ============================================================
# REYNOLDS NUMBER DIAGNOSTICS
# ============================================================

def get_centered_uv_snapshot(
    ds,
    time_dim,
    z_dim,
    time_index,
    level_index,
    u_var="UVEL",
    v_var="VVEL",
):
    """
    Extract UVEL/VVEL at one time and vertical level,
    clean fill values, and interpolate/crop to common tracer-cell centers.

    Reuses:
        clean_fill()
        uv_to_center()
    """
    if u_var not in ds:
        raise KeyError(f"{u_var} not found in dataset.")
    if v_var not in ds:
        raise KeyError(f"{v_var} not found in dataset.")

    u = clean_fill(ds[u_var].isel({time_dim: time_index, z_dim: level_index}).values)
    v = clean_fill(ds[v_var].isel({time_dim: time_index, z_dim: level_index}).values)

    u_c, v_c = uv_to_center(u[None, ...], v[None, ...])

    return u_c[0], v_c[0]


def velocity_anomaly(u, v, method="domain_mean", filter_size=None):
    """
    Compute velocity anomalies.

    Parameters
    ----------
    method : str
        "domain_mean":
            u' = u - mean(u)
            v' = v - mean(v)

        "box_filter":
            subtract a simple running mean using scipy.ndimage.uniform_filter.
    filter_size : int or None
        Filter size in grid cells for method="box_filter".

    Returns
    -------
    u_prime, v_prime : 2D arrays
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    if method == "domain_mean":
        return u - np.nanmean(u), v - np.nanmean(v)

    if method == "box_filter":
        if filter_size is None:
            raise ValueError("filter_size must be provided for method='box_filter'.")

        try:
            from scipy.ndimage import uniform_filter
        except ImportError as exc:
            raise ImportError("scipy is required for method='box_filter'.") from exc

        u_fill = np.where(np.isfinite(u), u, np.nanmean(u))
        v_fill = np.where(np.isfinite(v), v, np.nanmean(v))

        u_smooth = uniform_filter(u_fill, size=filter_size, mode="nearest")
        v_smooth = uniform_filter(v_fill, size=filter_size, mode="nearest")

        return u - u_smooth, v - v_smooth

    raise ValueError("method must be 'domain_mean' or 'box_filter'.")


def rms_velocity(u, v):
    """
    RMS horizontal velocity magnitude.

    U_rms = sqrt(<u^2 + v^2>)
    """
    u, v = crop_to_common_shape(u, v)
    mask = np.isfinite(u) & np.isfinite(v)

    if not np.any(mask):
        return np.nan

    return float(np.sqrt(np.nanmean(u[mask] ** 2 + v[mask] ** 2)))


def reynolds_number_from_UL(U, L, nu):
    """
    Reynolds number:

        Re = U L / nu

    Parameters
    ----------
    U : float
        Velocity scale [m/s]
    L : float
        Length scale [m]
    nu : float
        Effective viscosity [m2/s]
    """
    if nu is None or nu <= 0:
        raise ValueError("nu must be a positive scalar viscosity [m2/s].")

    return float(U * L / nu)


def effective_nu_from_biharmonic(A4, L):
    """
    Convert biharmonic viscosity A4 [m4/s] to a scale-dependent
    effective harmonic viscosity:

        nu_eff(L) = A4 / L^2

    This is useful when estimating Reynolds numbers for biharmonic runs.
    """
    if A4 is None or A4 < 0:
        raise ValueError("A4 must be non-negative.")
    if L is None or L <= 0:
        raise ValueError("L must be positive.")

    return float(A4 / L**2)


def reynolds_fixed_length_scales(
    u,
    v,
    length_scales_m,
    nu,
    remove_mean=False,
    anomaly_method="domain_mean",
    filter_size=None,
):
    """
    Method 1 and Method 2.

    Method 1:
        remove_mean=False
        Uses total velocity RMS.

    Method 2:
        remove_mean=True
        Uses anomaly velocity RMS.

    Parameters
    ----------
    u, v : 2D arrays
        Centered horizontal velocities [m/s].
    length_scales_m : dict
        Example:
            {
                "dx": 500.0,
                "5dx": 2500.0,
                "10dx": 5000.0,
                "box": 64000.0,
            }
    nu : float
        Viscosity [m2/s].
    remove_mean : bool
        Whether to use velocity anomalies.
    anomaly_method : str
        "domain_mean" or "box_filter".
    filter_size : int or None
        Only used for box_filter anomalies.

    Returns
    -------
    out : dict
    """
    u, v = crop_to_common_shape(u, v)

    if remove_mean:
        u_use, v_use = velocity_anomaly(
            u,
            v,
            method=anomaly_method,
            filter_size=filter_size,
        )
    else:
        u_use, v_use = u, v

    U_rms = rms_velocity(u_use, v_use)

    out = {
        "U_rms": U_rms,
        "nu_m2_s": float(nu),
        "remove_mean": bool(remove_mean),
        "anomaly_method": anomaly_method if remove_mean else "none",
        "filter_size": filter_size if remove_mean else None,
    }

    for name, L in length_scales_m.items():
        out[f"L_{name}_m"] = float(L)
        out[f"Re_{name}"] = reynolds_number_from_UL(U_rms, L, nu)

    return out


def get_peak_spectral_density_length(
    eds_ds,
    variable="spectral_density",
    min_wavelength_m=None,
    max_wavelength_m=None,
):
    """
    Extract the peak spectral-density length scale from the output of
    calculate_EDS_init().

    The EDS dataset should contain:
        eds_ds[variable]
        eds_ds["characteristic_length"]
        eds_ds["wavenumber"]

    Returns
    -------
    out : dict
    """
    if variable not in eds_ds:
        raise KeyError(f"{variable} not found in EDS dataset.")

    if "characteristic_length" not in eds_ds.coords:
        raise KeyError("EDS dataset must contain coordinate 'characteristic_length'.")

    if "wavenumber" not in eds_ds.coords:
        raise KeyError("EDS dataset must contain coordinate 'wavenumber'.")

    E = np.asarray(eds_ds[variable].values, dtype=float)
    L = np.asarray(eds_ds["characteristic_length"].values, dtype=float)
    k = np.asarray(eds_ds["wavenumber"].values, dtype=float)

    mask = np.isfinite(E) & np.isfinite(L) & np.isfinite(k)

    if min_wavelength_m is not None:
        mask &= L >= float(min_wavelength_m)

    if max_wavelength_m is not None:
        mask &= L <= float(max_wavelength_m)

    if not np.any(mask):
        raise ValueError("No valid spectral bins remain after wavelength filtering.")

    E_valid = E[mask]
    L_valid = L[mask]
    k_valid = k[mask]

    imax = int(np.nanargmax(E_valid))

    return {
        "L_peak_m": float(L_valid[imax]),
        "L_peak_km": float(L_valid[imax] / 1000.0),
        "k_peak_cpm": float(k_valid[imax]),
        "spectral_density_peak": float(E_valid[imax]),
        "variable": variable,
        "min_wavelength_m": min_wavelength_m,
        "max_wavelength_m": max_wavelength_m,
    }


def reynolds_peak_spectral_scale(
    u,
    v,
    eds_ds,
    nu,
    remove_mean=True,
    anomaly_method="domain_mean",
    filter_size=None,
    variable="spectral_density",
    min_wavelength_m=None,
    max_wavelength_m=None,
):
    """
    Method 3.

    Reynolds number using the dominant length scale from the peak
    of the spectral density:

        Re_peak = U_rms * L_peak / nu

    Usually use remove_mean=True because your EDS function normally
    removes the mean before computing the spectrum.
    """
    peak = get_peak_spectral_density_length(
        eds_ds,
        variable=variable,
        min_wavelength_m=min_wavelength_m,
        max_wavelength_m=max_wavelength_m,
    )

    u, v = crop_to_common_shape(u, v)

    if remove_mean:
        u_use, v_use = velocity_anomaly(
            u,
            v,
            method=anomaly_method,
            filter_size=filter_size,
        )
    else:
        u_use, v_use = u, v

    U_rms = rms_velocity(u_use, v_use)
    Re_peak = reynolds_number_from_UL(U_rms, peak["L_peak_m"], nu)

    out = {
        "U_rms": U_rms,
        "nu_m2_s": float(nu),
        "remove_mean": bool(remove_mean),
        "anomaly_method": anomaly_method if remove_mean else "none",
        "filter_size": filter_size if remove_mean else None,
        "Re_peak": Re_peak,
    }

    out.update(peak)

    return out


def summarize_reynolds_strategies(
    u,
    v,
    dx,
    dy,
    nu,
    eds_ds=None,
    box_length_m=None,
    spectral_min_wavelength_m=None,
    spectral_max_wavelength_m=None,
    anomaly_method="domain_mean",
    filter_size=None,
):
    """
    Convenience wrapper for:
        Method 1: total velocity RMS, fixed length scales
        Method 2: anomaly velocity RMS, fixed length scales
        Method 3: anomaly velocity RMS, spectral peak length scale

    Returns
    -------
    summary : dict
        Flat dictionary suitable for pandas.DataFrame([summary])
    """
    length_scales = {
        "dx": float(dx),
        "5dx": float(5 * dx),
        "10dx": float(10 * dx),
    }

    if box_length_m is not None:
        length_scales["box"] = float(box_length_m)

    out = {}

    method1 = reynolds_fixed_length_scales(
        u,
        v,
        length_scales_m=length_scales,
        nu=nu,
        remove_mean=False,
    )

    for key, val in method1.items():
        out[f"method1_{key}"] = val

    method2 = reynolds_fixed_length_scales(
        u,
        v,
        length_scales_m=length_scales,
        nu=nu,
        remove_mean=True,
        anomaly_method=anomaly_method,
        filter_size=filter_size,
    )

    for key, val in method2.items():
        out[f"method2_{key}"] = val

    if eds_ds is not None:
        method3 = reynolds_peak_spectral_scale(
            u,
            v,
            eds_ds=eds_ds,
            nu=nu,
            remove_mean=True,
            anomaly_method=anomaly_method,
            filter_size=filter_size,
            min_wavelength_m=spectral_min_wavelength_m,
            max_wavelength_m=spectral_max_wavelength_m,
        )

        for key, val in method3.items():
            out[f"method3_{key}"] = val

    out["dx_m"] = float(dx)
    out["dy_m"] = float(dy)
    out["nu_m2_s"] = float(nu)

    return out