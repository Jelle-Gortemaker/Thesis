import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


# Make the shared root-level theme import work from both scripts and notebooks.
for _parent in Path(__file__).resolve().parents:
    if (_parent / "theme" / "plot_theme.py").exists():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

try:
    import theme.plot_theme as ptheme
except Exception:  # fallback for standalone use
    class _FallbackTheme:
        STYLE = "default"
        DPI = 120
        SAVE_DPI = 300

        FONT_SIZE = 14
        TITLE_SIZE = 16
        LABEL_SIZE = 14
        TICK_SIZE = 11
        LEGEND_SIZE = 12
        ANNOTATION_SIZE = 10

        LINE_WIDTH = 1.2
        THIN_LINE_WIDTH = 0.8
        THICK_LINE_WIDTH = 2.0
        MARKER_SIZE = 4
        BAND_ALPHA = 0.15
        GRID_ALPHA = 0.35
        REFERENCE_ALPHA = 0.80

        MAP_GRIDLINE_ALPHA = 0.35
        MAP_GRIDLINE_LINESTYLE = "--"
        MAP_GRIDLINE_LINEWIDTH = 0.6
        MAP_LABEL_SIZE = TICK_SIZE

        QUIVER_COLOR = "0.35"
        QUIVER_ALPHA = 0.45
        QUIVER_WIDTH = 0.0022

        TARGET_BOX_LINEWIDTH = THICK_LINE_WIDTH
        TARGET_BOX_COLOR = "#D61418"
        TOC_LIGHT = "white"

        @staticmethod
        def apply_theme():
            plt.style.use("default")
            plt.rcParams.update({
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "savefig.facecolor": "white",
                "savefig.transparent": False,
                "axes.grid": True,
                "grid.alpha": 0.35,
            })

        @staticmethod
        def get_figsize(kind="wide"):
            return {
                "map": (7, 6),
                "wide": (10, 5),
                "tall": (8.5, 12.5),
                "panel": (15, 5),
                "square": (6.5, 6.5),
            }.get(kind, (10, 5))

        @staticmethod
        def get_cmap(category=None):
            cmap_map = {
                "vorticity": "RdBu_r",
                "diverging": "RdBu_r",
                "rossby": "RdBu_r",
                "density": "turbo",
                "jpdf": "turbo",
                "winter": "Blues",
                "summer": "Reds",
                "spring": "Greens",
                "autumn": "Oranges",
            }
            return plt.get_cmap(cmap_map.get(category, "viridis"))

        @staticmethod
        def get_color(name="default"):
            color_map = {
                "default": "#003755",
                "secondary": "#01CBE1",
                "highlight": "#D61418",
                "reference": "0.25",
                "spectrum_shell": "#003755",
                "spectrum_density": "#01CBE1",
                "std_band_shell": "#003755",
                "std_band_density": "#01CBE1",
            }
            return color_map.get(name, name)

        @staticmethod
        def save_figure(fig, path, save=True, close=False):
            if save and path is not None:
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
            if close:
                plt.close(fig)

    ptheme = _FallbackTheme()


# ============================================================
# SHARED PLOTTING HELPERS
# ============================================================

def _apply_consistent_style():
    """Apply the shared thesis style and enforce a white background."""
    ptheme.apply_theme()
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
    })


def _figsize(kind="wide", fallback=(10, 5)):
    try:
        return ptheme.get_figsize(kind)
    except Exception:
        return fallback


def _cmap(category="default"):
    try:
        return ptheme.get_cmap(category)
    except Exception:
        return plt.get_cmap("viridis")


def _color(name="default", fallback="C0"):
    try:
        return ptheme.get_color(name)
    except Exception:
        return fallback


def _set_white_background(fig, *axes):
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")


def _style_cartesian_axis(ax, grid=True):
    """Consistent style for non-map axes."""
    ax.set_facecolor("white")

    for spine in ax.spines.values():
        spine.set_linewidth(getattr(ptheme, "THIN_LINE_WIDTH", 0.8))
        spine.set_color("0.2")

    ax.tick_params(
        direction="out",
        length=4,
        width=getattr(ptheme, "THIN_LINE_WIDTH", 0.8),
        colors="0.15",
        top=False,
        right=False,
    )

    if grid:
        ax.grid(
            True,
            which="both",
            linewidth=getattr(ptheme, "MAP_GRIDLINE_LINEWIDTH", 0.6),
            linestyle=getattr(ptheme, "MAP_GRIDLINE_LINESTYLE", "--"),
            alpha=getattr(ptheme, "GRID_ALPHA", 0.35),
        )
    else:
        ax.grid(False)


def _save_and_show(fig, save_path=None, save=False, show=True, close=False):
    if save and save_path is not None:
        ptheme.save_figure(fig, save_path, save=True, close=False)
    if show:
        plt.show()
    if close:
        plt.close(fig)


def _valid_xy(x, y):
    return np.isfinite(x) & np.isfinite(y)
def process_glorys_data(filepath, averaging_period, depth_idx=0, time_idx=0, time_range=None):
    """
    Loads, optionally subsets in time, averages, and extracts the surface layer from GLORYS data.

    Parameters
    ----------
    filepath : str
        Path to NetCDF file.
    averaging_period : str
        Resampling period, e.g. '1D', '7D', '1M'.
    depth_idx : int, optional
        Depth index to extract.
    time_idx : int, optional
        Time index to extract after resampling.
    time_range : tuple[str, str] or None, optional
        Time window to use before resampling, e.g. ('2020-08-05', '2020-08-15').
        If None, the full dataset is used.
    """
    ds = xr.open_dataset(filepath)

    if time_range is not None:
        start_time, end_time = time_range
        ds = ds.sel(time=slice(start_time, end_time))

    ds_resampled = ds.resample(time=averaging_period).mean()
    surface_ds = ds_resampled.isel(depth=depth_idx, time=time_idx)
    return surface_ds

def get_subdomain(lat_min, lon_min, lat_max, lon_max):
    """
    Simply returns the manual bounds in the [lon_min, lon_max, lat_min, lat_max] 
    format expected by the plotting and slicing functions.
    """
    return [lon_min, lon_max, lat_min, lat_max]



def get_eddy_intensity(results, alpha=0.2):
    """
    Returns vorticity only where OW passes the threshold.
    """
    ow_param = results.w
    vorticity = results.vorticity

    dims_to_std = [d for d in ['latitude', 'longitude', 'lat', 'lon'] if d in ow_param.coords]

    sigma_W = ow_param.std(dim=dims_to_std)
    ow_threshold = -alpha * sigma_W

    mask = xr.where(ow_param < ow_threshold, 1, 0)
    signed_intensity = vorticity.where(mask == 1)

    return signed_intensity, mask

def calculate_okubo_weiss(ds_slice):
    """Calculates Okubo-Weiss parameter and relative vorticity."""
    R = 6371000
    lon_dim = 'longitude' if 'longitude' in ds_slice.dims else 'lon'
    lat_dim = 'latitude' if 'latitude' in ds_slice.dims else 'lat'

    dlon_deg = np.mean(np.gradient(ds_slice[lon_dim].values))
    dlat_deg = np.mean(np.gradient(ds_slice[lat_dim].values))

    dx = np.deg2rad(dlon_deg) * R * np.cos(np.deg2rad(ds_slice[lat_dim].values))
    dy = np.deg2rad(dlat_deg) * R

    grad_u = np.gradient(ds_slice.uo.values)
    grad_v = np.gradient(ds_slice.vo.values)

    du_di, du_dj = grad_u[0], grad_u[1]
    dv_di, dv_dj = grad_v[0], grad_v[1]

    du_dx = du_dj / dx[:, None]
    dv_dx = dv_dj / dx[:, None]
    du_dy = du_di / dy
    dv_dy = dv_di / dy

    sn = du_dx - dv_dy
    ss = dv_dx + du_dy
    omega = dv_dx - du_dy
    w = sn**2 + ss**2 - omega**2

    return xr.Dataset({
        'w': ((lat_dim, lon_dim), w),
        'vorticity': ((lat_dim, lon_dim), omega),
    }, coords=ds_slice.coords)


def plot_ocean_field(
    data,
    u=None,
    v=None,
    title="",
    cmap=None,
    cmap_category="default",
    label="",
    target_box=None,
    extent=None,
    lon_ticks=None,
    lat_ticks=None,
    vabs=None,
    **kwargs,
):
    """
    Consistent Cartopy map plot.

    Rules:
    - white background
    - x-axis tick labels only on top
    - y-axis tick labels only on left
    - fixed map extent if provided, otherwise GPGP default
    - optional fixed symmetric colorbar magnitude through vabs
    - no Cartopy gridline labels, to avoid duplicated labels
    """
    _apply_consistent_style()

    if cmap is None:
        cmap = _cmap(cmap_category)

    if extent is None:
        extent = [-155, -130, 20, 45]

    if lon_ticks is None:
        lon_ticks = np.arange(extent[0], extent[1] + 1, 5)

    if lat_ticks is None:
        lat_ticks = np.arange(extent[2], extent[3] + 1, 5)

    # Pull kwargs that should not be passed to pcolormesh.
    skip = kwargs.pop("skip", 5)
    vector_scale = kwargs.pop("vector_scale", 15)
    kwargs.pop("center", None)
    kwargs.pop("robust", None)

    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)

    if vabs is not None:
        vmin = -float(vabs)
        vmax = float(vabs)

    lon_name = "longitude" if "longitude" in data.coords else "lon"
    lat_name = "latitude" if "latitude" in data.coords else "lat"

    da = data.squeeze().transpose(lat_name, lon_name)
    lon = da[lon_name].values
    lat = da[lat_name].values
    field = da.values

    fig = plt.figure(figsize=_figsize("wide"), facecolor="white")

    # Keep enough space for left labels and the colorbar.
    ax = fig.add_axes([0.08, 0.10, 0.72, 0.80], projection=ccrs.PlateCarree())
    _set_white_background(fig, ax)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE, linewidth=getattr(ptheme, "THIN_LINE_WIDTH", 0.8))
    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="none", zorder=0)

    im = ax.pcolormesh(
        lon,
        lat,
        field,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        **kwargs,
    )

    # Colorbar just to the right of the map; right-side y labels are disabled.
    cax = fig.add_axes([0.68, 0.10, 0.035, 0.80])
    cbar = fig.colorbar(im, cax=cax, extend="both")
    cbar.set_label(label)
    cbar.ax.set_facecolor("white")

    ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    # Hard enforce: x labels only top; y labels only left.
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.yaxis.set_ticks_position("left")
    ax.yaxis.set_label_position("left")

    ax.tick_params(
        axis="x",
        which="both",
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False,
        pad=4,
    )
    ax.tick_params(
        axis="y",
        which="both",
        left=True,
        right=False,
        labelleft=True,
        labelright=False,
        pad=6,
    )

    ax.grid(
        True,
        linewidth=getattr(ptheme, "MAP_GRIDLINE_LINEWIDTH", 0.6),
        linestyle=getattr(ptheme, "MAP_GRIDLINE_LINESTYLE", "--"),
        alpha=getattr(ptheme, "MAP_GRIDLINE_ALPHA", 0.35),
    )

    if u is not None and v is not None:
        u_lon_name = "longitude" if "longitude" in u.coords else "lon"
        u_lat_name = "latitude" if "latitude" in u.coords else "lat"

        ax.quiver(
            u[u_lon_name][::skip],
            u[u_lat_name][::skip],
            u[::skip, ::skip],
            v[::skip, ::skip],
            color=getattr(ptheme, "QUIVER_COLOR", "0.35"),
            alpha=getattr(ptheme, "QUIVER_ALPHA", 0.45),
            width=getattr(ptheme, "QUIVER_WIDTH", 0.0022),
            scale=vector_scale,
            transform=ccrs.PlateCarree(),
        )

    if target_box is not None:
        lon_min, lon_max, lat_min, lat_max = target_box

        rect = patches.Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            linewidth=getattr(ptheme, "TARGET_BOX_LINEWIDTH", getattr(ptheme, "THICK_LINE_WIDTH", 2.0)),
            edgecolor=getattr(ptheme, "TARGET_BOX_COLOR", _color("highlight", "#D61418")),
            facecolor="none",
            transform=ccrs.PlateCarree(),
            zorder=10,
        )
        ax.add_patch(rect)

        ax.text(
            lon_min,
            lat_max + 0.15,
            "Target Box",
            color=getattr(ptheme, "TARGET_BOX_COLOR", _color("highlight", "#D61418")),
            fontsize=getattr(ptheme, "ANNOTATION_SIZE", 10),
            fontweight="bold",
            transform=ccrs.PlateCarree(),
            zorder=11,
        )

    ax.set_title(title)

    return fig, ax


def plot_eddy_intensity(
    intensity_data,
    u=None,
    v=None,
    title="",
    target_box=None,
    extent=None,
    vabs=1.0e-5,
):
    """
    Plot eddy-core relative vorticity with fixed symmetric colorbar limits.

    For animations or comparisons, pass the same vabs to all calls. A good
    choice is a robust percentile of the detected, non-NaN vortex-core values.
    """
    if extent is None:
        extent = [-155, -130, 20, 45]

    return plot_ocean_field(
        intensity_data,
        u=u,
        v=v,
        title=title,
        target_box=target_box,
        extent=extent,
        cmap=_cmap("vorticity"),
        label=r"Relative vorticity [s$^{-1}$]",
        vabs=vabs,
        vector_scale=15,
    )

def preprocess_netcdf(surface_ds, les_box, base_name, active=True, subtract_mean=False):
    """
    Slice GLORYS data and save it in a format compatible with
    VortexFitting's current `file_type='dns'` reader.
    """

    if not active:
        print("Vortex box saving is DEACTIVATED.")
        return None

    # 1. Use the passed surface_ds directly
    ds = surface_ds

    # 2. Slice requested box
    lon_w, lon_e, lat_s, lat_n = les_box
    sliced = ds.sel(longitude=slice(lon_w, lon_e), latitude=slice(lat_s, lat_n))

    # 3. Ensure time dimension exists
    if "time" not in sliced.dims:
        sliced = sliced.expand_dims(time=[0])

    # 4. Keep depth as a singleton dimension (required by dns reader)
    if "depth" in sliced.dims:
        sliced = sliced.isel(depth=slice(0, 1))
    else:
        sliced = sliced.expand_dims(depth=[0.0])

    # 5. Sort latitude so it increases monotonically
    sliced = sliced.sortby("latitude")

    # 6. Convert lon/lat to local Cartesian meters
    R = 6371000.0
    lon_vals = sliced["longitude"].values
    lat_vals = sliced["latitude"].values

    lon0 = float(lon_vals[0])
    lat0 = float(lat_vals[0])
    mean_lat = float(np.mean(lat_vals))

    x_m = (lon_vals - lon0) * (np.pi / 180.0) * R * np.cos(np.deg2rad(mean_lat))
    y_m = (lat_vals - lat0) * (np.pi / 180.0) * R

    # 7. Rename velocity variables only
    compat = sliced.rename({
        "uo": "velocity_x",
        "vo": "velocity_y"
    })

    # 8. Replace coordinate VALUES, but keep coordinate NAMES
    compat = compat.assign_coords(
        longitude=("longitude", x_m.astype(np.float64)),
        latitude=("latitude", y_m.astype(np.float64))
    )

    # 9. Optional background-flow subtraction
    if subtract_mean:
        compat["velocity_x"] = compat["velocity_x"] - compat["velocity_x"].mean(
            dim=("latitude", "longitude")
        )
        compat["velocity_y"] = compat["velocity_y"] - compat["velocity_y"].mean(
            dim=("latitude", "longitude")
        )

    # 10. Add dummy vertical velocity component
    compat["velocity_z"] = xr.zeros_like(compat["velocity_x"])

    # 11. Keep only what the dns reader needs
    compat = compat[["velocity_x", "velocity_y", "velocity_z"]]

    # 12. Enforce exact dimension order expected by the dns reader
    compat = compat.transpose("time", "depth", "latitude", "longitude")

    # 13. Add metadata
    compat["longitude"].attrs["units"] = "m"
    compat["latitude"].attrs["units"] = "m"
    compat["depth"].attrs["units"] = "m"
    compat["velocity_x"].attrs["units"] = "m s-1"
    compat["velocity_y"].attrs["units"] = "m s-1"
    compat["velocity_z"].attrs["units"] = "m s-1"

    # 14. Build output filename (using a default base_name since input_path is gone)
    coord_suffix = f"_{abs(int(lat_s))}-{abs(int(lat_n))}N_{abs(int(lon_w))}-{abs(int(lon_e))}W"
    output_filename = f"../data/{base_name}{coord_suffix}.nc"

    # 15. Save
    compat.to_netcdf(output_filename)

    # 16. Diagnostics
    print(f"Saved to: {output_filename}")
    print("velocity_x dims :", compat["velocity_x"].dims)
    print("velocity_x shape:", compat["velocity_x"].shape)
    print("longitude range :", float(compat["longitude"].values[0]), "to", float(compat["longitude"].values[-1]), "m")
    print("latitude range  :", float(compat["latitude"].values[0]), "to", float(compat["latitude"].values[-1]), "m")

    return output_filename


from pathlib import Path
import numpy as np
import xarray as xr


def infer_horizontal_dims(da, depth_dim):
    ignored_dims = {depth_dim, "time", "time_counter", "t"}
    spatial_dims = [dim for dim in da.dims if dim not in ignored_dims]

    if len(spatial_dims) != 2:
        raise ValueError(
            f"Could not infer exactly two horizontal dimensions from {da.name}. "
            f"DataArray dims are {da.dims}, got horizontal dims {spatial_dims}."
        )

    y_dim = next(
        (dim for dim in spatial_dims if dim.lower().startswith("y") or "lat" in dim.lower()),
        None,
    )
    x_dim = next(
        (dim for dim in spatial_dims if dim.lower().startswith("x") or "lon" in dim.lower()),
        None,
    )

    if y_dim is None or x_dim is None:
        y_dim, x_dim = spatial_dims

    return y_dim, x_dim


def get_vertical_edges_from_centers(z):
    z = np.asarray(z, dtype=float)

    if np.any(np.diff(z) <= 0):
        raise ValueError("Depth coordinate must increase downward.")

    edges = np.zeros(len(z) + 1)
    edges[1:-1] = 0.5 * (z[:-1] + z[1:])
    edges[0] = 0.0
    edges[-1] = z[-1] + 0.5 * (z[-1] - z[-2])

    return edges


def average_in_delR_layers(da, source_edges, delR, depth_dim):
    delR = np.asarray(delR, dtype=float)
    target_edges = np.concatenate(([0.0], np.cumsum(delR)))

    averaged_layers = []

    for k in range(len(delR)):
        top = target_edges[k]
        bottom = target_edges[k + 1]

        overlap = np.maximum(
            0.0,
            np.minimum(source_edges[1:], bottom) - np.maximum(source_edges[:-1], top),
        )

        if np.sum(overlap) <= 0:
            raise ValueError(
                f"No source vertical cells overlap target layer {k + 1}: "
                f"{top}–{bottom} m."
            )

        weights = xr.DataArray(
            overlap,
            dims=[depth_dim],
            coords={depth_dim: da[depth_dim]},
        )

        layer_mean = (da * weights).sum(dim=depth_dim) / weights.sum(dim=depth_dim)
        averaged_layers.append(layer_mean)

    out = xr.concat(averaged_layers, dim=depth_dim)
    out = out.assign_coords({depth_dim: 0.5 * (target_edges[:-1] + target_edges[1:])})

    return out


def slice_vertical_layers_delR(
    input_nc,
    output_nc,
    delR,
    depth_dim="z",
    variables=None,
    time_index=None,
    dtype=">f4",
    fill_value=0.0,
    Nx=None,
    Ny=None,
):
    """
    Average u and v velocities inside user-defined delR layers and write:
      1. MITgcm-ready u velocity binary
      2. MITgcm-ready v velocity binary
      3. NetCDF file with the averaged velocity fields

    Important:
    - The NetCDF output preserves the original horizontal dimensions, e.g.
      u_mit(z, y_u, x_u) and v_mit(z, y_v, x_v).
    - The binary files are cropped to (Nr, Ny, Nx), because MITgcm initial
      condition files should not contain the extra staggered boundary point.

    Parameters
    ----------
    input_nc : str or Path
        Input NetCDF file.
    output_nc : str or Path
        Output NetCDF file. Binary files use the same name stem.
    delR : list or array
        Target MITgcm layer thicknesses in metres, from surface downward.
        Example: [20, 480] or [25, 25, 50, 400].
    depth_dim : str
        Name of the source vertical coordinate/dimension.
    variables : list[str]
        Velocity variable names. For your file, use ["u_mit", "v_mit"].
    time_index : int or None
        Optional time index. Use None if there is no time dimension.
    dtype : str
        Binary output dtype. Use ">f4" for readBinaryPrec=32, ">f8" for readBinaryPrec=64.
    fill_value : float
        Value used to replace NaNs before writing binary files.
    Nx, Ny : int or None
        Target MITgcm tracer-grid size. If None, inferred from the minimum
        horizontal sizes of u and v after averaging.
    """

    input_nc = Path(input_nc)
    output_nc = Path(output_nc)

    if variables is None:
        variables = ["u_mit", "v_mit"]

    if len(variables) != 2:
        raise ValueError("variables should contain exactly two names: [u_variable, v_variable].")

    if not input_nc.exists():
        raise FileNotFoundError(f"Input file not found: {input_nc}")

    delR = np.asarray(delR, dtype=float)

    if np.any(delR <= 0):
        raise ValueError("All delR values must be positive.")

    u_var, v_var = variables

    ds = xr.open_dataset(input_nc)

    if depth_dim not in ds.dims:
        ds.close()
        raise ValueError(
            f"Depth dimension '{depth_dim}' not found. Available dimensions: {list(ds.dims)}"
        )

    if u_var not in ds.data_vars or v_var not in ds.data_vars:
        available = list(ds.data_vars)
        ds.close()
        raise ValueError(
            f"Velocity variables not found. Requested {variables}. Available: {available}"
        )

    if time_index is not None:
        for possible_time_dim in ["time", "time_counter", "t"]:
            if possible_time_dim in ds.dims:
                ds = ds.isel({possible_time_dim: time_index})
                break

    z = ds[depth_dim].values
    source_edges = get_vertical_edges_from_centers(z)

    max_requested_depth = np.sum(delR)

    if max_requested_depth > source_edges[-1]:
        ds.close()
        raise ValueError(
            f"Requested total depth {max_requested_depth:.2f} m, but source data only reaches "
            f"approximately {source_edges[-1]:.2f} m."
        )

    u = ds[u_var]
    v = ds[v_var]

    u_avg = average_in_delR_layers(u, source_edges, delR, depth_dim)
    v_avg = average_in_delR_layers(v, source_edges, delR, depth_dim)

    u_y_dim, u_x_dim = infer_horizontal_dims(u_avg, depth_dim)
    v_y_dim, v_x_dim = infer_horizontal_dims(v_avg, depth_dim)

    u_out_da = u_avg.transpose(depth_dim, u_y_dim, u_x_dim)
    v_out_da = v_avg.transpose(depth_dim, v_y_dim, v_x_dim)

    u_arr_full = np.nan_to_num(u_out_da.values, nan=fill_value)
    v_arr_full = np.nan_to_num(v_out_da.values, nan=fill_value)

    u_arr_full = np.ascontiguousarray(u_arr_full)
    v_arr_full = np.ascontiguousarray(v_arr_full)

    Nr = len(delR)

    # Infer MITgcm binary target size.
    # For staggered data:
    #   u might be (Nr, Ny, Nx+1)
    #   v might be (Nr, Ny+1, Nx)
    # The MITgcm init binaries should be cropped to (Nr, Ny, Nx).
    if Ny is None:
        Ny = min(u_arr_full.shape[1], v_arr_full.shape[1])

    if Nx is None:
        Nx = min(u_arr_full.shape[2], v_arr_full.shape[2])

    u_arr_bin = u_arr_full[:, :Ny, :Nx]
    v_arr_bin = v_arr_full[:, :Ny, :Nx]

    expected_shape = (Nr, Ny, Nx)

    if u_arr_bin.shape != expected_shape:
        ds.close()
        raise ValueError(
            f"Unexpected u binary shape after cropping: {u_arr_bin.shape}. "
            f"Expected {expected_shape}."
        )

    if v_arr_bin.shape != expected_shape:
        ds.close()
        raise ValueError(
            f"Unexpected v binary shape after cropping: {v_arr_bin.shape}. "
            f"Expected {expected_shape}."
        )

    output_nc.parent.mkdir(parents=True, exist_ok=True)

    stem = output_nc.with_suffix("")
    u_bin = stem.parent / f"{stem.name}_uvel.bin"
    v_bin = stem.parent / f"{stem.name}_vvel.bin"

    u_arr_bin.astype(dtype).tofile(u_bin)
    v_arr_bin.astype(dtype).tofile(v_bin)

    ds_out = xr.Dataset(
        {
            f"{u_var}_delRavg": u_out_da,
            f"{v_var}_delRavg": v_out_da,
        }
    )

    ds_out = ds_out.assign_coords(
        {
            "delR": (depth_dim, delR),
            "layer_top": (depth_dim, np.concatenate(([0.0], np.cumsum(delR)[:-1]))),
            "layer_bottom": (depth_dim, np.cumsum(delR)),
        }
    )

    ds_out.attrs["description"] = "Velocity averaged over user-defined delR layers."
    ds_out.attrs["source_file"] = str(input_nc)
    ds_out.attrs["binary_dtype"] = dtype
    ds_out.attrs["binary_Nr"] = int(Nr)
    ds_out.attrs["binary_Ny"] = int(Ny)
    ds_out.attrs["binary_Nx"] = int(Nx)
    ds_out.attrs["note"] = (
        "NetCDF preserves original horizontal dimensions. "
        "Binary velocity files are cropped to (Nr, Ny, Nx)."
    )

    ds_out.to_netcdf(output_nc)

    ds.close()
    ds_out.close()

    print(f"Saved NetCDF to: {output_nc}")
    print(f"Saved u velocity to: {u_bin}")
    print(f"Saved v velocity to: {v_bin}")
    print(f"delR: {delR.tolist()}")
    print()
    print("NetCDF shapes:")
    print(f"  u full shape: {u_arr_full.shape} = ({depth_dim}, {u_y_dim}, {u_x_dim})")
    print(f"  v full shape: {v_arr_full.shape} = ({depth_dim}, {v_y_dim}, {v_x_dim})")
    print()
    print("MITgcm binary shapes:")
    print(f"  u binary shape written: {u_arr_bin.shape}")
    print(f"  v binary shape written: {v_arr_bin.shape}")

    if dtype == ">f4":
        print("Use readBinaryPrec = 32 in MITgcm.")
    elif dtype == ">f8":
        print("Use readBinaryPrec = 64 in MITgcm.")


def spinup_to_baro1_bin(
    input_nc,
    output_file,
    time_index=None,
    time_days=None,
    time_seconds=None,
    time_dim=None,
    z_index=0,
    dtype=">f4",
):
    """
    Read one vertical layer of UVEL and VVEL from an MITgcm NetCDF output file
    and write MITgcm-ready 1-layer binary files.

    Time selection priority:
        1. time_index   : direct integer index
        2. time_days    : physical model time in days, nearest available output
        3. time_seconds : physical model time in seconds, nearest available output
        4. None         : last available timestep

    Preferred use:
        spinup_to_baro1_bin(
            input_nc="spinup_state.nc",
            output_file="../data/processed/MITgcm_spinup_256x256_jan2020_1layer.nc",
            time_days=2.0,
        )

    writes:
        ../data/processed/MITgcm_spinup_256x256_jan2020_1layer_uvel.bin
        ../data/processed/MITgcm_spinup_256x256_jan2020_1layer_vvel.bin
    """

    input_nc = Path(input_nc)
    output_file = Path(output_file)

    ds = xr.open_dataset(input_nc)

    try:
        if "UVEL" not in ds:
            raise KeyError("UVEL not found in input dataset.")
        if "VVEL" not in ds:
            raise KeyError("VVEL not found in input dataset.")

        u = ds["UVEL"]
        v = ds["VVEL"]

        # ------------------------------------------------------------
        # Infer time dimension
        # ------------------------------------------------------------
        if time_dim is None:
            candidate_time_dims = ["T", "time", "Time", "iter"]
            time_dim = next(
                (d for d in candidate_time_dims if d in u.dims or d in v.dims),
                None,
            )

            if time_dim is None:
                for d in u.dims:
                    dl = d.lower()
                    if dl.startswith("t") or "time" in dl:
                        time_dim = d
                        break

        has_time = time_dim is not None and (time_dim in u.dims or time_dim in v.dims)

        # ------------------------------------------------------------
        # Select timestep
        # ------------------------------------------------------------
        selected_time_index = None
        selected_time_value = None

        if has_time:
            nt = ds.sizes[time_dim]

            n_time_selectors = sum(
                x is not None for x in [time_index, time_days, time_seconds]
            )
            if n_time_selectors > 1:
                raise ValueError(
                    "Choose only one of time_index, time_days, or time_seconds."
                )

            if time_index is not None:
                selected_time_index = int(time_index)

            elif time_days is not None or time_seconds is not None:
                target_seconds = (
                    float(time_seconds)
                    if time_seconds is not None
                    else float(time_days) * 86400.0
                )

                if time_dim not in ds.coords:
                    raise ValueError(
                        f"Cannot select by physical time because {time_dim!r} "
                        "is not available as a coordinate. Use time_index instead."
                    )

                time_values = np.asarray(ds[time_dim].values)

                # Convert datetime64 coordinates to seconds relative to first output.
                if np.issubdtype(time_values.dtype, np.datetime64):
                    time_seconds_available = (
                        time_values - time_values[0]
                    ) / np.timedelta64(1, "s")
                    time_seconds_available = time_seconds_available.astype(float)

                else:
                    time_seconds_available = time_values.astype(float)

                    # Heuristic:
                    # MITgcm MNC T is often in seconds.
                    # If values are small and user asks by days, they may already be in days.
                    if np.nanmax(np.abs(time_seconds_available)) < 1000.0 and time_days is not None:
                        time_seconds_available = time_seconds_available * 86400.0

                selected_time_index = int(
                    np.nanargmin(np.abs(time_seconds_available - target_seconds))
                )

                selected_time_value = float(time_seconds_available[selected_time_index])

            else:
                selected_time_index = nt - 1

            if selected_time_index < 0:
                selected_time_index = nt + selected_time_index

            if selected_time_index < 0 or selected_time_index >= nt:
                raise IndexError(
                    f"time_index={selected_time_index} out of range for "
                    f"{time_dim}, nt={nt}"
                )

            if time_dim in u.dims:
                u = u.isel({time_dim: selected_time_index})
            if time_dim in v.dims:
                v = v.isel({time_dim: selected_time_index})

        else:
            if any(x is not None for x in [time_index, time_days, time_seconds]):
                raise ValueError(
                    "A time selector was provided, but no time dimension was found."
                )

        # ------------------------------------------------------------
        # Select vertical layer
        # ------------------------------------------------------------
        z_u = [d for d in u.dims if d.startswith("Z") or d.lower().startswith("z")]
        z_v = [d for d in v.dims if d.startswith("Z") or d.lower().startswith("z")]

        if z_u:
            nz_u = u.sizes[z_u[0]]
            if z_index < 0 or z_index >= nz_u:
                raise IndexError(f"z_index={z_index} out of range for UVEL; nz={nz_u}")
            u = u.isel({z_u[0]: z_index})

        if z_v:
            nz_v = v.sizes[z_v[0]]
            if z_index < 0 or z_index >= nz_v:
                raise IndexError(f"z_index={z_index} out of range for VVEL; nz={nz_v}")
            v = v.isel({z_v[0]: z_index})

        # ------------------------------------------------------------
        # Crop staggered grids to tracer size
        # UVEL: Y x Xp1 -> Y x X
        # VVEL: Yp1 x X -> Y x X
        # ------------------------------------------------------------
        Nx = ds.attrs.get("Nx", None)
        Ny = ds.attrs.get("Ny", None)

        if Nx is None:
            Nx = ds.sizes["X"] if "X" in ds.sizes else min(u.shape[-1], v.shape[-1])
        if Ny is None:
            Ny = ds.sizes["Y"] if "Y" in ds.sizes else min(u.shape[-2], v.shape[-2])

        Nx = int(Nx)
        Ny = int(Ny)

        u_arr = np.nan_to_num(u.values, nan=0.0)[:Ny, :Nx]
        v_arr = np.nan_to_num(v.values, nan=0.0)[:Ny, :Nx]

        # Add vertical dimension back: (Nr, Ny, Nx), with Nr=1
        u_arr = u_arr[np.newaxis, :, :]
        v_arr = v_arr[np.newaxis, :, :]

        output_file.parent.mkdir(parents=True, exist_ok=True)

        stem = output_file.with_suffix("")
        u_bin = stem.parent / f"{stem.name}_uvel.bin"
        v_bin = stem.parent / f"{stem.name}_vvel.bin"

        np.ascontiguousarray(u_arr).astype(dtype).tofile(u_bin)
        np.ascontiguousarray(v_arr).astype(dtype).tofile(v_bin)

        print(f"Saved u velocity to: {u_bin}")
        print(f"Saved v velocity to: {v_bin}")
        print(f"u shape written: {u_arr.shape}")
        print(f"v shape written: {v_arr.shape}")

        if has_time:
            print(f"Selected time dimension: {time_dim}")
            print(f"Selected time index    : {selected_time_index}")
            if selected_time_value is not None:
                print(f"Selected model time    : {selected_time_value:.1f} s = {selected_time_value / 86400.0:.4f} days")

        print(f"Selected vertical index: {z_index}")

        if dtype == ">f4":
            print("Use readBinaryPrec = 32 in MITgcm.")
        elif dtype == ">f8":
            print("Use readBinaryPrec = 64 in MITgcm.")

        return {
            "u_bin": u_bin,
            "v_bin": v_bin,
            "u_shape": u_arr.shape,
            "v_shape": v_arr.shape,
            "time_dim": time_dim,
            "time_index": selected_time_index,
            "time_seconds": selected_time_value,
            "z_index": z_index,
        }

    finally:
        ds.close()


def calculate_EDS(
    filepath,
    target_box=None,
    max_wavelength_km=None,
    initialized_velocity=False,
    x_res=None,
    y_res=None,
    n_bins=8,
    remove_mean=True,
    temporal_window_days=7,
    temporal_skip_days=0,
    temporal_stride_days=1,
    min_modes_per_bin=3,
    rose_scale_bands_km=None,
    rose_n_angle_bins=12,
):
    ds = xr.open_dataset(filepath)

    # 1. Select box and infer spacing
    if initialized_velocity:
        if target_box is None:
            box_ds = ds.isel(depth=0)
        else:
            box_ds = ds.sel(
                x=slice(target_box[0], target_box[1]),
                y=slice(target_box[2], target_box[3]),
            ).isel(depth=0)

        x = box_ds.x.values
        y = box_ds.y.values

        if x_res is None:
            x_res = float(np.abs(np.mean(np.gradient(x))))
        if y_res is None:
            y_res = float(np.abs(np.mean(np.gradient(y))))

        dx = x_res
        dy = y_res
        nx = len(x)
        ny = len(y)

    else:
        if target_box is None:
            box_ds = ds.isel(depth=0)
        else:
            box_ds = ds.sel(
                longitude=slice(target_box[0], target_box[1]),
                latitude=slice(target_box[2], target_box[3]),
            ).isel(depth=0)

        R_earth = 6371000.0
        phi_lat = float(box_ds.latitude.mean())

        dlat_deg = np.abs(np.mean(np.gradient(box_ds.latitude.values)))
        dlon_deg = np.abs(np.mean(np.gradient(box_ds.longitude.values)))

        dy = np.deg2rad(dlat_deg) * R_earth
        dx = np.deg2rad(dlon_deg) * R_earth * np.cos(np.deg2rad(phi_lat))

        ny = len(box_ds.latitude)
        nx = len(box_ds.longitude)

    if "time" not in box_ds.dims:
        raise ValueError("Dataset must contain a time dimension.")

    Lx = nx * dx
    Ly = ny * dy

    # 2. Wavenumber grid
    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="xy")

    k_mag = np.sqrt(kx_grid**2 + ky_grid**2)
    theta_rad = np.arctan2(ky_grid, kx_grid)

    valid_k = k_mag > 0
    if not np.any(valid_k):
        raise ValueError("No valid nonzero wavenumbers found.")

    natural_lambda_max = min(Lx, Ly)
    if max_wavelength_km is None:
        lambda_max = natural_lambda_max
    else:
        lambda_max = min(max_wavelength_km * 1000.0, natural_lambda_max)

    k_min = 1.0 / lambda_max
    k_max = np.nanmax(k_mag[valid_k])

    if k_min >= k_max:
        raise ValueError("Selected max wavelength is too small relative to the domain/grid.")

    k_bins = np.logspace(np.log10(k_min), np.log10(k_max), num=n_bins + 1)
    k_centers = np.sqrt(k_bins[:-1] * k_bins[1:])
    dk = np.diff(k_bins)

    shell_masks = []
    shell_mode_count = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        shell = (k_mag >= k_bins[i]) & (k_mag < k_bins[i + 1])
        shell_masks.append(shell)
        shell_mode_count[i] = int(np.sum(shell))

    # 3. Rose setup
    rose_enabled = rose_scale_bands_km is not None and len(rose_scale_bands_km) > 0
    if rose_enabled:
        rose_scale_bands_km = [tuple(b) for b in rose_scale_bands_km]
        n_rose_bands = len(rose_scale_bands_km)

        rose_angle_edges_deg = np.linspace(-180.0, 180.0, rose_n_angle_bins + 1)
        rose_angle_centers_deg = 0.5 * (rose_angle_edges_deg[:-1] + rose_angle_edges_deg[1:])

        rose_band_masks = []
        rose_band_labels = []

        for lam_hi_km, lam_lo_km in rose_scale_bands_km:
            k_lo = 1.0 / (lam_hi_km * 1000.0)
            k_hi = 1.0 / (lam_lo_km * 1000.0)
            rose_band_masks.append((k_mag >= k_lo) & (k_mag < k_hi))
            rose_band_labels.append(f"{lam_hi_km:g}-{lam_lo_km:g} km")
    else:
        n_rose_bands = 0
        rose_angle_edges_deg = np.array([])
        rose_angle_centers_deg = np.array([])
        rose_band_masks = []
        rose_band_labels = []

    # 4. Snapshot spectra
    n_time = len(box_ds.time)

    spectra_shell = []
    spectra_density = []
    spectra_axis_complex = []
    rose_snapshots = []

    for t in range(n_time):
        u = np.nan_to_num(box_ds.uo.isel(time=t).values, nan=0.0)
        v = np.nan_to_num(box_ds.vo.isel(time=t).values, nan=0.0)

        if remove_mean:
            u = u - np.mean(u)
            v = v - np.mean(v)

        u_hat = np.fft.fft2(u)
        v_hat = np.fft.fft2(v)

        N = nx * ny
        ke_2d = 0.5 * (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2) / (N ** 2)

        shell_spectrum = np.full(n_bins, np.nan)
        density_spectrum = np.full(n_bins, np.nan)
        axis_complex = np.full(n_bins, np.nan + 1j * np.nan, dtype=complex)

        for i in range(n_bins):
            shell = shell_masks[i]
            if shell_mode_count[i] < min_modes_per_bin:
                continue

            shell_energy = ke_2d[shell]
            shell_theta = theta_rad[shell]
            shell_sum = np.nansum(shell_energy)

            if not np.isfinite(shell_sum) or shell_sum <= 0:
                continue

            # only change: normalize shell-integrated spectrum by area
            shell_spectrum[i] = shell_sum
            density_spectrum[i] = shell_sum / dk[i]
            axis_complex[i] = np.nansum(shell_energy * np.exp(2j * shell_theta)) / shell_sum

        spectra_shell.append(shell_spectrum)
        spectra_density.append(density_spectrum)
        spectra_axis_complex.append(axis_complex)

        if rose_enabled:
            rose_band_energy = np.full((n_rose_bands, rose_n_angle_bins), np.nan)
            for b, band_mask in enumerate(rose_band_masks):
                band_energy = ke_2d[band_mask]
                band_theta = np.rad2deg(theta_rad[band_mask])
                band_sum = np.nansum(band_energy)

                if not np.isfinite(band_sum) or band_sum <= 0:
                    continue

                vals = np.zeros(rose_n_angle_bins, dtype=float)
                for j in range(rose_n_angle_bins):
                    m = (band_theta >= rose_angle_edges_deg[j]) & (band_theta < rose_angle_edges_deg[j + 1])
                    vals[j] = np.nansum(band_energy[m]) if np.any(m) else 0.0

                rose_band_energy[b, :] = vals

            rose_snapshots.append(rose_band_energy)

    spectra_shell = np.asarray(spectra_shell)
    spectra_density = np.asarray(spectra_density)
    spectra_axis_complex = np.asarray(spectra_axis_complex)
    if rose_enabled:
        rose_snapshots = np.asarray(rose_snapshots)

    # 5. Temporal averaging
    if n_time < temporal_window_days:
        raise ValueError(f"Dataset has only {n_time} time steps, but temporal_window_days={temporal_window_days}.")

    sample_step = temporal_skip_days + 1
    window_starts = np.arange(0, n_time - temporal_window_days + 1, temporal_stride_days)

    window_shell_means = []
    window_density_means = []
    window_axis_means = []
    window_rose_means = []

    for start in window_starts:
        idx = np.arange(start, start + temporal_window_days, sample_step)

        window_shell_means.append(np.nanmean(spectra_shell[idx, :], axis=0))
        window_density_means.append(np.nanmean(spectra_density[idx, :], axis=0))
        window_axis_means.append(np.nanmean(spectra_axis_complex[idx, :], axis=0))

        if rose_enabled:
            window_rose_means.append(np.nanmean(rose_snapshots[idx, :, :], axis=0))

    window_shell_means = np.asarray(window_shell_means)
    window_density_means = np.asarray(window_density_means)
    window_axis_means = np.asarray(window_axis_means)

    mean_shell = np.nanmean(window_shell_means, axis=0)
    std_shell = np.nanstd(window_shell_means, axis=0)

    mean_density = np.nanmean(window_density_means, axis=0)
    std_density = np.nanstd(window_density_means, axis=0)

    mean_axis = np.nanmean(window_axis_means, axis=0)
    anisotropy_strength = np.abs(mean_axis)
    dominant_axis_deg = np.mod(0.5 * np.rad2deg(np.angle(mean_axis)), 180.0)

    if rose_enabled:
        window_rose_means = np.asarray(window_rose_means)
        mean_rose = np.nanmean(window_rose_means, axis=0)
        rose_row_sums = np.nansum(mean_rose, axis=1, keepdims=True)
        rose_normalized = np.divide(
            mean_rose,
            rose_row_sums,
            out=np.full_like(mean_rose, np.nan, dtype=float),
            where=rose_row_sums > 0,
        )
    else:
        mean_rose = np.empty((0, 0))
        rose_normalized = np.empty((0, 0))

    # 6. Nyquist cutoff
    f_nyquist = 1.0 / (2.0 * max(dx, dy))
    nyq_mask = k_centers <= f_nyquist

    data_vars = {
        "shell_integrated_spectrum": (("wavenumber",), mean_shell[nyq_mask]),
        "shell_integrated_spectrum_std": (("wavenumber",), std_shell[nyq_mask]),
        "spectral_density": (("wavenumber",), mean_density[nyq_mask]),
        "spectral_density_std": (("wavenumber",), std_density[nyq_mask]),
        "dominant_axis_deg": (("wavenumber",), dominant_axis_deg[nyq_mask]),
        "anisotropy_strength": (("wavenumber",), anisotropy_strength[nyq_mask]),
        "shell_mode_count": (("wavenumber",), shell_mode_count[nyq_mask]),
    }

    coords = {
        "wavenumber": k_centers[nyq_mask],
        "characteristic_length": (("wavenumber",), 1.0 / k_centers[nyq_mask]),
    }

    if rose_enabled:
        data_vars["rose_spectrum"] = (("scale_band", "rose_angle_deg"), mean_rose)
        data_vars["rose_spectrum_normalized"] = (("scale_band", "rose_angle_deg"), rose_normalized)
        coords["scale_band"] = np.array(rose_band_labels, dtype=object)
        coords["rose_angle_deg"] = rose_angle_centers_deg

    return xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "dx_m": dx,
            "dy_m": dy,
            "domain_Lx_km": Lx / 1000.0,
            "domain_Ly_km": Ly / 1000.0,
        },
    ).dropna(dim="wavenumber", how="all")





def plot_eds_overview(
    eds,
    title="Spectrum and anisotropy overview",
    target_box=None,
    ylim_a=None,
    ylim_b=None,
    xlim_a=None,
    xlim_b=None,
    save_path=None,
    save=False,
    show=True,
    close=False,
):
    """
    Overview figure with consistent layout:
    - A: shell-integrated spectrum versus characteristic length
    - B: spectral density versus wavenumber
    - C: angular energy distribution for the first scale band

    Parameters
    ----------
    ylim_a : tuple[float, float] or None
        Manual y-axis limits for subplot A. Example: (1e-8, 1e-2).
    ylim_b : tuple[float, float] or None
        Manual y-axis limits for subplot B. Example: (1e-4, 1e4).
    xlim_a : tuple[float, float] or None
        Optional x-axis limits for subplot A in km.
    xlim_b : tuple[float, float] or None
        Optional x-axis limits for subplot B in cycles m^-1.
    """
    _apply_consistent_style()

    if "rose_spectrum_normalized" not in eds:
        raise ValueError("Dataset does not contain rose_spectrum_normalized.")

    coord_label = ""
    if target_box is not None:
        lon_w, lon_e, lat_s, lat_n = target_box
        coord_label = (
            f"{lat_s:.2f}--{lat_n:.2f}$^\\circ$N, "
            f"{abs(lon_w):.2f}--{abs(lon_e):.2f}$^\\circ$W"
        )

    length_km = eds["characteristic_length"].values / 1000.0
    E_shell = eds["shell_integrated_spectrum"].values
    E_shell_std = eds["shell_integrated_spectrum_std"].values

    k = eds["wavenumber"].values
    E_k = eds["spectral_density"].values
    E_k_std = eds["spectral_density_std"].values

    rose = eds["rose_spectrum_normalized"].values
    rose_angles_deg = eds["rose_angle_deg"].values
    rose_labels = eds["scale_band"].values

    if rose.shape[0] < 1:
        raise ValueError("No rose scale band found in rose_spectrum_normalized.")

    rose_vals = np.nan_to_num(rose[0], nan=0.0)
    rose_label = str(rose_labels[0])

    # Manual layout is used instead of constrained_layout to avoid title/label overlap.
    fig = plt.figure(figsize=(11.0, 13.5), facecolor="white")
    _set_white_background(fig)

    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.0, 1.0, 1.25],
    )

    fig.subplots_adjust(
        left=0.13,
        right=0.97,
        bottom=0.055,
        top=0.90,
        hspace=0.72,
    )

    # ------------------------------------------------------------
    # A. Shell-integrated spectrum
    # ------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    _style_cartesian_axis(ax1, grid=True)

    valid1 = _valid_xy(length_km, E_shell) & (length_km > 0) & (E_shell > 0)

    ax1.loglog(
        length_km[valid1],
        E_shell[valid1],
        "-o",
        lw=getattr(ptheme, "THICK_LINE_WIDTH", 2.0),
        ms=getattr(ptheme, "MARKER_SIZE", 4),
        color=_color("spectrum_shell", "#003755"),
        label="Shell-integrated KE",
    )

    if np.any(np.isfinite(E_shell_std[valid1])):
        lo = np.maximum(E_shell[valid1] - E_shell_std[valid1], 1e-30)
        hi = E_shell[valid1] + E_shell_std[valid1]

        ax1.fill_between(
            length_km[valid1],
            lo,
            hi,
            color=_color("std_band_shell", "#003755"),
            alpha=getattr(ptheme, "BAND_ALPHA", 0.15),
            linewidth=0,
            label=r"$\pm 1$ std. dev.",
        )

    ax1.set_xlabel("Characteristic length [km]", labelpad=6)
    ax1.set_ylabel(r"Shell contribution to mean KE [m$^2$ s$^{-2}$]", labelpad=8)
    ax1.set_title("A. Shell-integrated spectrum", pad=12)
    ax1.legend(loc="upper right", frameon=True)

    if xlim_a is not None:
        ax1.set_xlim(xlim_a)
    if ylim_a is not None:
        ax1.set_ylim(ylim_a)

    # ------------------------------------------------------------
    # B. Spectral density
    # ------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    _style_cartesian_axis(ax2, grid=True)

    valid2 = _valid_xy(k, E_k) & (k > 0) & (E_k > 0)

    ax2.loglog(
        k[valid2],
        E_k[valid2],
        "-o",
        lw=getattr(ptheme, "THICK_LINE_WIDTH", 2.0),
        ms=getattr(ptheme, "MARKER_SIZE", 4),
        color=_color("spectrum_density", "#01CBE1"),
        label="Spectral density",
    )

    if np.any(np.isfinite(E_k_std[valid2])):
        lo = np.maximum(E_k[valid2] - E_k_std[valid2], 1e-30)
        hi = E_k[valid2] + E_k_std[valid2]

        ax2.fill_between(
            k[valid2],
            lo,
            hi,
            color=_color("std_band_density", "#01CBE1"),
            alpha=getattr(ptheme, "BAND_ALPHA", 0.15),
            linewidth=0,
            label=r"$\pm 1$ std. dev.",
        )

    if np.sum(valid2) >= 3:
        k_plot = k[valid2]
        E_plot = E_k[valid2]

        mid = len(k_plot) // 2
        k0 = k_plot[mid]
        E0 = E_plot[mid]

        k_ref = np.array([k_plot[0], k_plot[-1]])
        y_53 = E0 * (k_ref / k0) ** (-5 / 3)
        y_3 = E0 * (k_ref / k0) ** (-3)

        ax2.loglog(
            k_ref,
            y_53,
            "--",
            color=_color("reference", "0.25"),
            lw=getattr(ptheme, "LINE_WIDTH", 1.2),
            alpha=getattr(ptheme, "REFERENCE_ALPHA", 0.8),
            label=r"$k^{-5/3}$",
        )
        ax2.loglog(
            k_ref,
            y_3,
            "--",
            color=_color("highlight", "#D61418"),
            lw=getattr(ptheme, "LINE_WIDTH", 1.2),
            alpha=getattr(ptheme, "REFERENCE_ALPHA", 0.8),
            label=r"$k^{-3}$",
        )

    ax2.set_xlabel(r"Wavenumber $k$ [cycles m$^{-1}$]", labelpad=6)
    ax2.set_ylabel(r"Spectral density [(m$^2$ s$^{-2}$)/(cycles m$^{-1}$)]", labelpad=8)
    ax2.set_title("B. Spectral density", pad=12)
    ax2.legend(loc="upper right", frameon=True)

    if xlim_b is not None:
        ax2.set_xlim(xlim_b)
    if ylim_b is not None:
        ax2.set_ylim(ylim_b)

    # ------------------------------------------------------------
    # C. Rose plot
    # ------------------------------------------------------------
    axr = fig.add_subplot(gs[2, 0], projection="polar")
    axr.set_facecolor("white")
    axr.set_anchor("C")

    angle_rad = np.deg2rad(rose_angles_deg)
    dtheta = 2 * np.pi / len(rose_angles_deg)

    axr.bar(
        angle_rad,
        rose_vals,
        width=dtheta,
        align="center",
        color=_color("spectrum_shell", "#003755"),
        edgecolor="white",
        linewidth=getattr(ptheme, "THIN_LINE_WIDTH", 0.8),
        alpha=0.90,
    )

    axr.set_theta_zero_location("E")
    axr.set_theta_direction(1)

    vmax = np.nanmax(rose_vals)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    axr.set_ylim(0, vmax * 1.1)

    rticks = np.linspace(0, vmax, 5)[1:]
    axr.set_rticks(rticks)
    axr.set_yticklabels([f"{r:.2f}" for r in rticks])
    axr.set_rlabel_position(135)

    axr.grid(
        True,
        linewidth=getattr(ptheme, "MAP_GRIDLINE_LINEWIDTH", 0.6),
        linestyle=getattr(ptheme, "MAP_GRIDLINE_LINESTYLE", "--"),
        alpha=getattr(ptheme, "GRID_ALPHA", 0.35),
    )

    axr.set_title(
        f"C. Angular energy distribution integrated over {rose_label} length scales",
        pad=28,
    )

    layer_label = ""
    if "layer_index" in eds.attrs:
        layer_label = f"Layer {eds.attrs['layer_index']}"

    title_parts = [title]
    if coord_label:
        title_parts.append(coord_label)
    if layer_label:
        title_parts.append(layer_label)

    fig.suptitle(
        "\n".join(title_parts),
        fontsize=getattr(ptheme, "TITLE_SIZE", 16),
        y=0.965,
    )

    fig.align_ylabels([ax1, ax2])

    _save_and_show(fig, save_path=save_path, save=save, show=show, close=close)

    return fig

def calculate_EDS_init(
    filepath,
    target_box=None,
    max_wavelength_km=None,
    x_res=None,
    y_res=None,
    n_bins=8,
    remove_mean=True,
    min_modes_per_bin=3,
    rose_scale_bands_km=None,
    rose_n_angle_bins=12,
    snapshot_index=None,
    layer_index=0,
):
    ds = xr.open_dataset(filepath)
    print(ds)

    # Step 1: Automatically determine the correct dimension for depth
    depth_dim = None
    for dim in ds.dims:
        if "Z" in dim or "depth" in dim.lower():
            depth_dim = dim
            break

    if depth_dim is None:
        raise ValueError("No depth-like dimension found in the dataset.")

    # Check what depth dimension is used
    print(f"Depth dimension: {depth_dim}")

    # Validate requested vertical layer
    n_layers = ds.sizes[depth_dim]
    if layer_index < 0 or layer_index >= n_layers:
        raise IndexError(
            f"layer_index={layer_index} is outside the available range 0..{n_layers - 1}"
        )

    print(f"Selected layer index: {layer_index}")

    # Select requested vertical layer
    if target_box is None:
        box_ds = ds.isel({depth_dim: layer_index})
    else:
        x_coord = 'X' if 'X' in ds.coords else 'x'
        y_coord = 'Y' if 'Y' in ds.coords else 'y'
        box_ds = ds.sel(
            **{
                x_coord: slice(target_box[0], target_box[1]),
                y_coord: slice(target_box[2], target_box[3]),
            }
        ).isel({depth_dim: layer_index})

    box_ds['UVEL'] = box_ds['UVEL'].interp(Xp1=box_ds['X'])
    box_ds['VVEL'] = box_ds['VVEL'].interp(Yp1=box_ds['Y'])

    print(f"Selected box shape UVEL: {box_ds['UVEL'].shape}")
    print(f"Selected box shape VVEL: {box_ds['VVEL'].shape}")

    x_coord = 'X'
    y_coord = 'Y'
    x = box_ds[x_coord].values
    y = box_ds[y_coord].values

    if x_res is None:
        x_res = float(np.abs(np.mean(np.gradient(x))))
    if y_res is None:
        y_res = float(np.abs(np.mean(np.gradient(y))))

    dx = x_res
    dy = y_res
    nx = len(x)
    ny = len(y)

    Lx = nx * dx
    Ly = ny * dy

    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="xy")

    k_mag = np.sqrt(kx_grid**2 + ky_grid**2)
    theta_rad = np.arctan2(ky_grid, kx_grid)

    valid_k = k_mag > 0
    if not np.any(valid_k):
        raise ValueError("No valid nonzero wavenumbers found.")

    natural_lambda_max = min(Lx, Ly)
    if max_wavelength_km is None:
        lambda_max = natural_lambda_max
    else:
        lambda_max = min(max_wavelength_km * 1000.0, natural_lambda_max)

    k_min = 1.0 / lambda_max
    k_max = np.nanmax(k_mag[valid_k])

    if k_min >= k_max:
        raise ValueError("Selected max wavelength is too small relative to the domain/grid.")

    k_bins = np.logspace(np.log10(k_min), np.log10(k_max), num=n_bins + 1)
    k_centers = np.sqrt(k_bins[:-1] * k_bins[1:])
    dk = np.diff(k_bins)

    shell_masks = []
    shell_mode_count = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        shell = (k_mag >= k_bins[i]) & (k_mag < k_bins[i + 1])
        shell_masks.append(shell)
        shell_mode_count[i] = int(np.sum(shell))

    rose_enabled = rose_scale_bands_km is not None and len(rose_scale_bands_km) > 0
    if rose_enabled:
        rose_scale_bands_km = [tuple(b) for b in rose_scale_bands_km]
        n_rose_bands = len(rose_scale_bands_km)

        rose_angle_edges_deg = np.linspace(-180.0, 180.0, rose_n_angle_bins + 1)
        rose_angle_centers_deg = 0.5 * (rose_angle_edges_deg[:-1] + rose_angle_edges_deg[1:])

        rose_band_masks = []
        rose_band_labels = []

        for lam_hi_km, lam_lo_km in rose_scale_bands_km:
            k_lo = 1.0 / (lam_hi_km * 1000.0)
            k_hi = 1.0 / (lam_lo_km * 1000.0)
            rose_band_masks.append((k_mag >= k_lo) & (k_mag < k_hi))
            rose_band_labels.append(f"{lam_hi_km:g}-{lam_lo_km:g} km")
    else:
        n_rose_bands = 0
        rose_angle_edges_deg = np.array([])
        rose_angle_centers_deg = np.array([])
        rose_band_masks = []
        rose_band_labels = []

    time_coord = 'T' if 'T' in box_ds.dims else ('time' if 'time' in box_ds.dims else None)
    if time_coord is None:
        raise ValueError("No time dimension found in the dataset.")

    n_time = len(box_ds[time_coord])
    if snapshot_index is not None:
        if snapshot_index < 0 or snapshot_index >= n_time:
            raise IndexError(
                f"snapshot_index={snapshot_index} is outside the available time range 0..{n_time - 1}"
            )
        selected_time_indices = [snapshot_index]
    else:
        selected_time_indices = np.arange(n_time)

    spectra_shell = []
    spectra_density = []
    spectra_axis_complex = []
    rose_snapshots = []

    for t in selected_time_indices:
        u = np.nan_to_num(box_ds["UVEL"].isel(**{time_coord: t}).values, nan=0.0)
        v = np.nan_to_num(box_ds["VVEL"].isel(**{time_coord: t}).values, nan=0.0)

        if remove_mean:
            u = u - np.mean(u)
            v = v - np.mean(v)

        u_hat = np.fft.fft2(u)
        v_hat = np.fft.fft2(v)

        N = nx * ny
        ke_2d = 0.5 * (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2) / (N ** 2)

        shell_spectrum = np.full(n_bins, np.nan)
        density_spectrum = np.full(n_bins, np.nan)
        axis_complex = np.full(n_bins, np.nan + 1j * np.nan, dtype=complex)

        for i in range(n_bins):
            shell = shell_masks[i]
            if shell_mode_count[i] < min_modes_per_bin:
                continue

            shell_energy = ke_2d[shell]
            shell_theta = theta_rad[shell]
            shell_sum = np.nansum(shell_energy)

            if not np.isfinite(shell_sum) or shell_sum <= 0:
                continue

            shell_spectrum[i] = shell_sum
            density_spectrum[i] = shell_sum / dk[i]
            axis_complex[i] = np.nansum(shell_energy * np.exp(2j * shell_theta)) / shell_sum

        spectra_shell.append(shell_spectrum)
        spectra_density.append(density_spectrum)
        spectra_axis_complex.append(axis_complex)

        if rose_enabled:
            rose_band_energy = np.full((n_rose_bands, rose_n_angle_bins), np.nan)
            for b, band_mask in enumerate(rose_band_masks):
                band_energy = ke_2d[band_mask]
                band_theta = np.rad2deg(theta_rad[band_mask])
                band_sum = np.nansum(band_energy)

                if not np.isfinite(band_sum) or band_sum <= 0:
                    continue

                vals = np.zeros(rose_n_angle_bins, dtype=float)
                for j in range(rose_n_angle_bins):
                    m = (band_theta >= rose_angle_edges_deg[j]) & (
                        band_theta < rose_angle_edges_deg[j + 1]
                    )
                    vals[j] = np.nansum(band_energy[m]) if np.any(m) else 0.0

                rose_band_energy[b, :] = vals

            rose_snapshots.append(rose_band_energy)

    spectra_shell = np.asarray(spectra_shell)
    spectra_density = np.asarray(spectra_density)
    spectra_axis_complex = np.asarray(spectra_axis_complex)

    if rose_enabled:
        rose_snapshots = np.asarray(rose_snapshots)
    else:
        rose_snapshots = np.empty((0, 0, 0))

    mean_shell = np.nanmean(spectra_shell, axis=0)
    std_shell = np.nanstd(spectra_shell, axis=0)

    mean_density = np.nanmean(spectra_density, axis=0)
    std_density = np.nanstd(spectra_density, axis=0)

    mean_axis = np.nanmean(spectra_axis_complex, axis=0)
    anisotropy_strength = np.abs(mean_axis)
    dominant_axis_deg = np.mod(0.5 * np.rad2deg(np.angle(mean_axis)), 180.0)

    if rose_enabled:
        mean_rose = np.nanmean(rose_snapshots, axis=0)
        rose_row_sums = np.nansum(mean_rose, axis=1, keepdims=True)
        rose_normalized = np.divide(
            mean_rose,
            rose_row_sums,
            out=np.full_like(mean_rose, np.nan, dtype=float),
            where=rose_row_sums > 0,
        )
    else:
        mean_rose = np.empty((0, 0))
        rose_normalized = np.empty((0, 0))

    if snapshot_index is None:
        snapshot_time_index = None
        snapshot_time_value = "all timesteps"
    else:
        snapshot_time_index = int(snapshot_index)
        snapshot_time_value = box_ds[time_coord].isel(**{time_coord: snapshot_index}).item()

    data_vars = {
        "shell_integrated_spectrum": (("wavenumber",), mean_shell),
        "shell_integrated_spectrum_std": (("wavenumber",), std_shell),
        "spectral_density": (("wavenumber",), mean_density),
        "spectral_density_std": (("wavenumber",), std_density),
        "dominant_axis_deg": (("wavenumber",), dominant_axis_deg),
        "anisotropy_strength": (("wavenumber",), anisotropy_strength),
    }

    coords = {
        "wavenumber": k_centers,
        "characteristic_length": (("wavenumber",), 1.0 / k_centers),
    }

    if rose_enabled:
        data_vars["rose_spectrum"] = (("scale_band", "rose_angle_deg"), mean_rose)
        data_vars["rose_spectrum_normalized"] = (("scale_band", "rose_angle_deg"), rose_normalized)
        coords["scale_band"] = np.array(rose_band_labels, dtype=object)
        coords["rose_angle_deg"] = rose_angle_centers_deg

    return xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "dx_m": dx,
            "dy_m": dy,
            "domain_Lx_km": Lx / 1000.0,
            "domain_Ly_km": Ly / 1000.0,
            "snapshot_time_index": snapshot_time_index,
            "snapshot_time_value": snapshot_time_value,
            "layer_index": int(layer_index),
            "depth_dim": depth_dim,
        },
    ).dropna(dim="wavenumber", how="all")




def plot_EDS_seasonal(
    data_groups,
    target_box,
    initialized_velocity=False,
    max_wavelength_km=None,
    n_bins=8,
    remove_mean=True,
    temporal_window_days=31,
    temporal_skip_days=0,
    temporal_stride_days=1,
    min_modes_per_bin=3,
    spectrum_var="shell_integrated_spectrum",
    base_colors=None,
    title="Seasonal energy spectrum comparison",
    xlim_km=None,
    ylim=None,
    save_path=None,
    save=False,
    show=True,
    close=False,
):
    """
    Consistent seasonal/yearly EDS comparison plot.

    Parameters
    ----------
    ylim : tuple[float, float] or None
        Optional y-axis limits for the seasonal spectrum plot.
    """
    _apply_consistent_style()

    if base_colors is None:
        base_colors = {
            "Winter": "winter",
            "Summer": "summer",
            "Spring": "spring",
            "Autumn": "autumn",
        }

    ylabel_map = {
        "shell_integrated_spectrum": r"Shell contribution to mean KE [m$^2$ s$^{-2}$]",
        "spectral_density": r"Spectral density [(m$^2$ s$^{-2}$)/(cycles m$^{-1}$)]",
    }

    if spectrum_var not in ylabel_map:
        raise ValueError("spectrum_var must be 'shell_integrated_spectrum' or 'spectral_density'")

    fig, ax = plt.subplots(figsize=(10.5, 6.2), facecolor="white")
    _set_white_background(fig, ax)
    _style_cartesian_axis(ax, grid=True)

    for season, years in data_groups.items():
        cmap_key = base_colors.get(season, "default")
        cmap = _cmap(cmap_key)

        if len(years) == 1:
            shades = np.array([0.70])
        else:
            shades = np.linspace(0.40, 0.90, len(years))

        for idx, (year_label, path) in enumerate(years.items()):
            eds = calculate_EDS(
                filepath=path,
                target_box=target_box,
                initialized_velocity=initialized_velocity,
                max_wavelength_km=max_wavelength_km,
                n_bins=n_bins,
                remove_mean=remove_mean,
                temporal_window_days=temporal_window_days,
                temporal_skip_days=temporal_skip_days,
                temporal_stride_days=temporal_stride_days,
                min_modes_per_bin=min_modes_per_bin,
                rose_scale_bands_km=None,
            )

            length_km = eds["characteristic_length"].values / 1000.0
            E = eds[spectrum_var].values

            valid = _valid_xy(length_km, E) & (length_km > 0) & (E > 0)

            ax.loglog(
                length_km[valid],
                E[valid],
                "-o",
                label=f"{season} {year_label}",
                color=cmap(shades[idx]),
                lw=getattr(ptheme, "THICK_LINE_WIDTH", 2.0),
                ms=getattr(ptheme, "MARKER_SIZE", 4),
                alpha=0.95,
            )

    ax.set_xlabel("Characteristic length [km]", labelpad=6)
    ax.set_ylabel(ylabel_map[spectrum_var], labelpad=8)
    ax.set_title(title, pad=12)

    if xlim_km is not None:
        ax.set_xlim(xlim_km)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend(
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        frameon=True,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.12, right=0.76, bottom=0.14, top=0.88)

    _save_and_show(fig, save_path=save_path, save=save, show=show, close=close)

    return fig, ax

