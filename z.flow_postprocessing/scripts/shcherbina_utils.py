from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def find_dim(dims, candidates):
    dims = list(dims)

    for c in candidates:
        if c in dims:
            return c

    # MITgcm MNC often uses names like Zmd000059, Xp1, Yp1, etc.
    for d in dims:
        if any(d.startswith(prefix) for prefix in candidates if isinstance(prefix, str)):
            return d

    # extra fallback for vertical dimensions
    for d in dims:
        dl = d.lower()
        if dl.startswith("zmd") or dl == "z":
            return d

    raise KeyError(f"Could not find any of {candidates} in dims={dims}")


def clean_fill(a, big=1e10):
    a = np.asarray(a, dtype=np.float32)
    a[np.abs(a) > big] = np.nan
    return a


def uv_to_center(u, v):
    """
    Convert staggered MITgcm U/V to tracer-cell centers.

    Parameters
    ----------
    u : ndarray
        Shape (..., y, x_u)
    v : ndarray
        Shape (..., y_v, x)

    Returns
    -------
    u_c, v_c : ndarray
        Both cropped to common tracer shape (..., y, x)
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
    Compute speed, relative vorticity, divergence, and strain from centered velocities.
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


def center_crop(*arrays):
    """
    Crop all arrays to the smallest common (..., y, x) horizontal shape.
    """
    if len(arrays) == 0:
        return []

    ny = min(a.shape[-2] for a in arrays)
    nx = min(a.shape[-1] for a in arrays)
    return [a[..., :ny, :nx] for a in arrays]


def normalize_hist(y):
    y = np.asarray(y, dtype=float)
    m = np.nanmax(y)
    if np.isfinite(m) and m > 0:
        return y / m
    return y


def flatten_finite(*arrays):
    """
    Return flattened arrays using the common finite mask.
    """
    if len(arrays) == 0:
        return []

    mask = np.ones_like(np.asarray(arrays[0]), dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)

    return [np.asarray(a)[mask] for a in arrays]


def compute_density(theta, salt, rho0=1035.0, tAlpha=2.0e-4, sBeta=7.4e-4,
                    tRef=420.0, sRef=2940.0):
    """
    Linear EOS density approximation consistent with MITgcm LINEAR EOS settings.
    """
    theta = np.asarray(theta, dtype=np.float64)
    salt = np.asarray(salt, dtype=np.float64)
    return rho0 * (1.0 - tAlpha * (theta - tRef) + sBeta * (salt - sRef))


def plot_map(field, title="", cbar_label="", cmap=None, vmin=None, vmax=None,
             dx=500.0, dy=500.0, figsize=(7, 5)):
    """
    Quick map plot on a uniform Cartesian grid.
    """
    field = np.asarray(field)
    ny, nx = field.shape[-2], field.shape[-1]

    x = np.arange(nx) * dx / 1000.0
    y = np.arange(ny) * dy / 1000.0
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=figsize)
    plt.pcolormesh(X, Y, field, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label=cbar_label)
    plt.xlabel("x [km]")
    plt.ylabel("y [km]")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_pdf(x, y, xlabel="", title="", figsize=(6, 4)):
    """
    Quick normalized PDF line plot.
    """
    plt.figure(figsize=figsize)
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel("normalized PDF")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_jpdf(x, y, H, xlabel="", ylabel="", title="", figsize=(7, 6),
              vmin=-2, vmax=0, cmap="turbo"):
    """
    Plot log10-normalized 2D PDF.
    """
    H = np.asarray(H, dtype=float)
    hmax = np.nanmax(H)

    if np.isfinite(hmax) and hmax > 0:
        Hn = H / hmax
    else:
        Hn = H

    Hlog = np.log10(np.where(Hn > 0, Hn, np.nan))

    plt.figure(figsize=figsize)
    plt.pcolormesh(x, y, Hlog.T, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label=r"$\log_{10}(\mathrm{PDF}/\mathrm{PDF}_{max})$")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()