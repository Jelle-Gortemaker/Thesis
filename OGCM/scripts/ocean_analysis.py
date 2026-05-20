import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import sys
from pathlib import Path

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

def plot_ocean_field(data, u=None, v=None, title="", cmap='viridis', label="", target_box=None, **kwargs):
    """Plotting wrapper with explicit ax.text argument naming."""
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)

    # Use pop to avoid passing these to the data.plot() function
    vector_scale = kwargs.pop('vector_scale', 10)
    skip = kwargs.pop('skip', 5)

    im = data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, 
                   cbar_kwargs={'label': label}, **kwargs)

    if u is not None and v is not None:
        ax.quiver(u.longitude[::skip], u.latitude[::skip], 
                  u[::skip, ::skip], v[::skip, ::skip],
                  color='black', alpha=0.4, scale=vector_scale, transform=ccrs.PlateCarree())
    
    if target_box is not None:
        # target_box is [lon_min, lon_max, lat_min, lat_max]
        lon_min, lon_max, lat_min, lat_max = target_box
        
        # Calculate width and height in degrees for the rectangle patch
        width = lon_max - lon_min
        height = lat_max - lat_min
        
        rect = patches.Rectangle((lon_min, lat_min), width, height,
                                 linewidth=3, edgecolor='red', facecolor='none', 
                                 transform=ccrs.PlateCarree(), zorder=10)
        ax.add_patch(rect)
        
        ax.text(x=lon_min, y=lat_max + 0.1, s="Target Box", 
                color='red', fontweight='bold', transform=ccrs.PlateCarree())

    plt.title(title)
    plt.show()


def plot_eddy_intensity(intensity_data, u=None, v=None, title="", target_box=None):
    """Specialized plot for cyclonic vs anticyclonic cores."""
    plot_ocean_field(
        intensity_data, 
        u=u, v=v, 
        title=title, 
        target_box=target_box,
        cmap='RdBu_r', 
        label="Relative Vorticity [s^-1]",
        center=0,       
        robust=True,
        vector_scale=15
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


def spinup_to_baro1_bin(input_nc, output_file, dtype=">f4"):
    """
    Read top-layer UVEL and VVEL from an MITgcm NetCDF output file and write
    MITgcm-ready binary files.

    output_file is used as naming template.

    Example:
        output_file = "../data/processed/MITgcm_spinup_256x256_jan2020_1layer.nc"

    writes:
        ../data/processed/MITgcm_spinup_256x256_jan2020_1layer_uvel.bin
        ../data/processed/MITgcm_spinup_256x256_jan2020_1layer_vvel.bin
    """

    input_nc = Path(input_nc)
    output_file = Path(output_file)

    ds = xr.open_dataset(input_nc)

    # Select last time step if time dimension exists
    u = ds["UVEL"]
    v = ds["VVEL"]

    if "T" in u.dims:
        u = u.isel(T=-1)
    if "T" in v.dims:
        v = v.isel(T=-1)

    # Select top vertical layer
    # In your file UVEL dims: (T, Zmd000002, Y, Xp1)
    # VVEL dims: (T, Zmd000002, Yp1, X)
    z_u = [d for d in u.dims if d.startswith("Z")]
    z_v = [d for d in v.dims if d.startswith("Z")]

    if z_u:
        u = u.isel({z_u[0]: 0})
    if z_v:
        v = v.isel({z_v[0]: 0})

    # Crop staggered grids to tracer size:
    # UVEL: Y x Xp1 -> Y x X
    # VVEL: Yp1 x X -> Y x X
    Nx = ds.attrs.get("Nx", None)
    Ny = ds.attrs.get("Ny", None)

    if Nx is None:
        Nx = ds.sizes["X"] if "X" in ds.sizes else min(u.shape[-1], v.shape[-1])
    if Ny is None:
        Ny = ds.sizes["Y"] if "Y" in ds.sizes else min(u.shape[-2], v.shape[-2])

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

    ds.close()

    print(f"Saved u velocity to: {u_bin}")
    print(f"Saved v velocity to: {v_bin}")
    print(f"u shape written: {u_arr.shape}")
    print(f"v shape written: {v_arr.shape}")

    if dtype == ">f4":
        print("Use readBinaryPrec = 32 in MITgcm.")
    elif dtype == ">f8":
        print("Use readBinaryPrec = 64 in MITgcm.")


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
):
    """
    Overview figure with:
    - top: shell-integrated spectrum vs characteristic length
    - middle: spectral density vs wavenumber
    - bottom: single rose plot (first / only scale band)
    """

    if "rose_spectrum_normalized" not in eds:
        raise ValueError("Dataset does not contain rose_spectrum_normalized.")
    
    coord_label = ""
    if target_box is not None:
        lon_w, lon_e, lat_s, lat_n = target_box
        coord_label = (
            f"{lat_s:.2f}–{lat_n:.2f}°N, "
            f"{abs(lon_w):.2f}–{abs(lon_e):.2f}°W"
        )

    # -----------------------------
    # Extract data
    # -----------------------------
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

    # Use only the first rose band
    rose_vals = rose[0]
    rose_label = str(rose_labels[0])

    # -----------------------------
    # Styling
    # -----------------------------
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
    })

    fig = plt.figure(figsize=(8.5, 12.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.05, 1.2], hspace=0.18)

    # -----------------------------
    # A. Shell-integrated spectrum
    # -----------------------------
    ax1 = fig.add_subplot(gs[0, 0])

    valid1 = np.isfinite(length_km) & np.isfinite(E_shell) & (length_km > 0) & (E_shell > 0)

    ax1.loglog(
        length_km[valid1],
        E_shell[valid1],
        "-o",
        lw=2.0,
        ms=5,
        color="tab:blue",
        label="Shell-integrated KE",
    )

    if np.any(np.isfinite(E_shell_std[valid1])):
        lo = np.maximum(E_shell[valid1] - E_shell_std[valid1], 1e-30)
        hi = E_shell[valid1] + E_shell_std[valid1]
        ax1.fill_between(
            length_km[valid1],
            lo,
            hi,
            color="tab:blue",
            alpha=0.15,
            linewidth=0,
        )

    ax1.set_xlabel("Characteristic length (km)")
    ax1.set_ylabel(r"Shell contribution to mean KE (m$^2$ s$^{-2}$)")
    ax1.set_title("A. Shell-integrated spectrum", pad=14)
    ax1.grid(True, which="both", alpha=0.22)
    ax1.legend(
    handles=[
        ax1.lines[0],
        patches.Patch(facecolor="tab:blue", alpha=0.15, edgecolor="none"),
    ],
    labels=[
        "Shell-integrated KE",
        r"$\pm 1$ std. dev.",
    ],
    loc="upper right",
    frameon=True,
    )

    # -----------------------------
    # B. Spectral density
    # -----------------------------
    ax2 = fig.add_subplot(gs[1, 0])

    valid2 = np.isfinite(k) & np.isfinite(E_k) & (k > 0) & (E_k > 0)

    ax2.loglog(
        k[valid2],
        E_k[valid2],
        "-o",
        lw=2.0,
        ms=5,
        color="tab:orange",
        label="Spectral density",
    )

    if np.any(np.isfinite(E_k_std[valid2])):
        lo = np.maximum(E_k[valid2] - E_k_std[valid2], 1e-30)
        hi = E_k[valid2] + E_k_std[valid2]
        ax2.fill_between(
            k[valid2],
            lo,
            hi,
            color="tab:orange",
            alpha=0.15,
            linewidth=0,
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

        ax2.loglog(k_ref, y_53, "--", color="0.45", lw=1.8, alpha=0.9, label=r"$k^{-5/3}$")
        ax2.loglog(k_ref, y_3, "--", color="royalblue", lw=1.8, alpha=0.9, label=r"$k^{-3}$")

    ax2.set_xlabel(r"Wavenumber $k$ (cycles m$^{-1}$)")
    ax2.set_ylabel(r"Spectral density ((m$^2$ s$^{-2}$)/(cycles m$^{-1}$))")
    ax2.set_title("B. Spectral density", pad=8)
    ax2.grid(True, which="both", alpha=0.22)
    ax2.legend(
    handles=[
        ax2.lines[0],
        patches.Patch(facecolor="tab:orange", alpha=0.15, edgecolor="none", label=r"$\pm 1$ std. dev."),
        ax2.lines[1],
        ax2.lines[2],
    ],
    loc="upper right",
    frameon=True,
    )
    
    legend_handles = [
    ax2.lines[0],
    patches.Patch(facecolor="tab:orange", alpha=0.15, edgecolor="none"),
    ]
    legend_labels = [
        "Spectral density",
        r"$\pm 1$ std. dev.",
    ]

    if len(ax2.lines) > 1:
        legend_handles.extend(ax2.lines[1:])
        legend_labels.extend([line.get_label() for line in ax2.lines[1:]])

    ax2.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper right",
        frameon=True,
    )


    # -----------------------------
    # C. Single rose plot
    # -----------------------------
    axr = fig.add_subplot(gs[2, 0], projection="polar")
    axr.set_anchor("C")

    angle_rad = np.deg2rad(rose_angles_deg)
    dtheta = 2 * np.pi / len(rose_angles_deg)
    vals = np.nan_to_num(rose_vals, nan=0.0)

    axr.bar(
        angle_rad,
        vals,
        width=dtheta,
        align="center",
        color="tab:blue",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.9,
    )

    axr.set_theta_zero_location("E")
    axr.set_theta_direction(1)

    vmax = np.nanmax(vals)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    axr.set_ylim(0, vmax * 1.1)

    # sparse radial labels
    rticks = np.linspace(0, vmax, 5)[1:]
    axr.set_rticks(rticks)
    axr.set_yticklabels([f"{r:.2f}" for r in rticks])
    axr.set_rlabel_position(135)

    axr.grid(alpha=0.35)
    axr.set_title(f"C. Angular energy distribution integrated over ({rose_label}) length scales", va="bottom", pad=14)

    layer_label = ""
    if "layer_index" in eds.attrs:
        layer_label = f"Layer {eds.attrs['layer_index']}"
    title_parts = [title]

    if coord_label != "":
        title_parts.append(coord_label)

    if layer_label != "":
        title_parts.append(layer_label)

    fig.suptitle("\n".join(title_parts), fontsize=15, y=1.04)
    plt.show()


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
    spectrum_var="shell_integrated_spectrum",  # or "spectral_density"
    base_colors=None,
    title="Seasonal energy spectrum comparison",
    xlim_km=None,
):
    """
    Plot seasonal / yearly comparison of EDS using the current calculate_EDS method.

    Parameters
    ----------
    data_groups : dict
        Example:
        {
            "Winter": {"2020": "...", "2021": "..."},
            "Summer": {"2020": "...", "2021": "..."},
        }
    target_box : list
        [lon_min, lon_max, lat_min, lat_max] for lon/lat data,
        or [x_min, x_max, y_min, y_max] for initialized Cartesian data.
    spectrum_var : str
        Either:
        - "shell_integrated_spectrum"
        - "spectral_density"
    """

    if base_colors is None:
        base_colors = {
            "Winter": "Blues",
            "Summer": "Reds",
            "Spring": "Greens",
            "Autumn": "Oranges",
        }

    ylabel_map = {
        "shell_integrated_spectrum": r"Shell contribution to mean KE (m$^2$ s$^{-2}$)",
        "spectral_density": r"Spectral density ((m$^2$ s$^{-2}$)/(m$^{-1}$))",
    }

    if spectrum_var not in ylabel_map:
        raise ValueError("spectrum_var must be 'shell_integrated_spectrum' or 'spectral_density'")

    plt.figure(figsize=(10, 7))

    for season, years in data_groups.items():
        cmap = plt.get_cmap(base_colors.get(season, "viridis"))
        shades = np.linspace(0.4, 0.9, len(years))

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
                rose_scale_bands_km=None,  # not needed for this plot
            )

            length_km = eds["characteristic_length"].values / 1000.0
            E = eds[spectrum_var].values

            valid = np.isfinite(length_km) & np.isfinite(E) & (length_km > 0) & (E > 0)

            plt.loglog(
                length_km[valid],
                E[valid],
                "-o",
                label=f"{season} {year_label}",
                color=cmap(shades[idx]),
                lw=2,
                ms=5,
                alpha=0.95,
            )

    plt.xlabel("Characteristic length (km)")
    plt.ylabel(ylabel_map[spectrum_var])
    plt.title(title, fontsize=14)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(True, which="both", alpha=0.2)

    if xlim_km is not None:
        plt.xlim(xlim_km)

    plt.tight_layout()
    plt.show()