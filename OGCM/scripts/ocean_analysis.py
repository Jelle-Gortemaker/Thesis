import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import sys

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


def calculate_coriolis_beta_plane(ds_slice):
    """
    Calculates the Coriolis parameter f using the beta-plane approximation.
    f = f0 + beta * y
    """
    omega_earth = 7.2921e-5  # Earth's angular velocity [rad/s]
    R = 6371000             # Earth's radius [m]
    
    # Identify lat dimension name
    lat_dim = 'latitude' if 'latitude' in ds_slice.dims else 'lat'
    
    # 1. Define reference latitude (center of the domain)
    phi_0_deg = float(ds_slice[lat_dim].mean())
    phi_0_rad = np.deg2rad(phi_0_deg)
    
    # 2. Calculate constants
    f0 = 2 * omega_earth * np.sin(phi_0_rad)
    beta = (2 * omega_earth * np.cos(phi_0_rad)) / R
    
    # 3. Calculate y (meridional distance from the center in meters)
    y = (ds_slice[lat_dim] - phi_0_deg) * (np.pi * R / 180)
    
    f = f0 + beta * y
    return f.assign_attrs(units='s^-1', long_name='Coriolis parameter (beta-plane)')

def get_eddy_intensity(results, alpha=0.2):
    """
    Returns the vorticity field only where OW passes the threshold.
    This allows us to see:
    1. Strength (magnitude of vorticity)
    2. Rotation sign (Cyclonic > 0, Anticyclonic < 0 in NH)
    """
    ow_param = results.w
    vorticity = results.vorticity
    
    # Identify dims for standard deviation
    dims_to_std = [d for d in ['latitude', 'longitude', 'lat', 'lon'] if d in ow_param.coords]
    
    sigma_W = ow_param.std(dim=dims_to_std)
    ow_threshold = -alpha * sigma_W
    
    # Create the binary mask
    mask = xr.where(ow_param < ow_threshold, 1, 0)
    
    # Proxy for Intensity: Relative Vorticity masked by OW
    # Positive values = Cyclonic, Negative = Anticyclonic (Northern Hemisphere)
    signed_intensity = vorticity.where(mask == 1)
    
    # Optional: Normalize by Coriolis to get Rossby Number cores
    ro_intensity = (vorticity / results.f).where(mask == 1)
    
    return signed_intensity, ro_intensity, mask

def calculate_okubo_weiss(ds_slice):
    """Calculates OW parameter with broadcasting fix for latitude-dependent dx."""
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

    sn, ss, omega = (du_dx - dv_dy), (dv_dx + du_dy), (dv_dx - du_dy)
    w = sn**2 + ss**2 - omega**2
    f = calculate_coriolis_beta_plane(ds_slice)
    
    return xr.Dataset({
        'w': ((lat_dim, lon_dim), w),
        'vorticity': ((lat_dim, lon_dim), omega),
        'f': ((lat_dim), f.values)
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


# def calculate_EDS(
#     filepath,
#     target_box=None,
#     max_wavelength_km=400.0,
#     initialized_velocity=False,
#     x_res=None,
#     y_res=None,
#     n_bins=40,
#     remove_mean=True,
# ):
#     """
#     Calculate an isotropic energy-density spectrum from horizontal velocity fields.

#     Method:
#     - Compute FFT of u and v separately
#     - Form 2D kinetic-energy power spectral density
#     - Radially bin in |k|
#     - Average over time snapshots

#     Parameters
#     ----------
#     filepath : str
#         Path to NetCDF file.
#     target_box : list or None
#         If initialized_velocity=False:
#             [lon_min, lon_max, lat_min, lat_max]
#         If initialized_velocity=True:
#             [x_min, x_max, y_min, y_max] in meters
#         If None, full horizontal domain is used.
#     max_wavelength_km : float
#         Largest wavelength to include in binning.
#     initialized_velocity : bool
#         True for initialized MITgcm velocity files on Cartesian x/y grids.
#         False for lon/lat datasets.
#     x_res, y_res : float or None
#         Horizontal grid spacing in meters for initialized_velocity=True.
#         If None, inferred from x and y coordinates.
#     n_bins : int
#         Number of logarithmic radial wavenumber bins.
#     remove_mean : bool
#         If True, subtract spatial mean from u and v before FFT.

#     Returns
#     -------
#     xr.DataArray
#         Isotropic energy-density spectrum as function of characteristic length [m].
#     """

#     ds = xr.open_dataset(filepath)

#     # -------------------------
#     # 1. Select horizontal box
#     # -------------------------
#     if initialized_velocity:
#         # Cartesian MITgcm-style data
#         if target_box is None:
#             box_ds = ds.isel(depth=0)
#         else:
#             box_ds = ds.sel(
#                 x=slice(target_box[0], target_box[1]),
#                 y=slice(target_box[2], target_box[3])
#             ).isel(depth=0)

#         x = box_ds.x.values
#         y = box_ds.y.values

#         if x_res is None:
#             x_res = float(np.abs(np.mean(np.gradient(x))))
#         if y_res is None:
#             y_res = float(np.abs(np.mean(np.gradient(y))))

#         dx = x_res
#         dy = y_res
#         nx = len(x)
#         ny = len(y)

#     else:
#         # Geographic lon/lat data
#         if target_box is None:
#             box_ds = ds.isel(depth=0)
#         else:
#             box_ds = ds.sel(
#                 longitude=slice(target_box[0], target_box[1]),
#                 latitude=slice(target_box[2], target_box[3])
#             ).isel(depth=0)

#         R_earth = 6371000.0
#         phi_lat = float(box_ds.latitude.mean())

#         dlat_deg = np.abs(np.mean(np.gradient(box_ds.latitude.values)))
#         dlon_deg = np.abs(np.mean(np.gradient(box_ds.longitude.values)))

#         dy = np.deg2rad(dlat_deg) * R_earth
#         dx = np.deg2rad(dlon_deg) * R_earth * np.cos(np.deg2rad(phi_lat))

#         ny = len(box_ds.latitude)
#         nx = len(box_ds.longitude)

#     # -------------------------
#     # 2. Wavenumber grid
#     # -------------------------
#     kx = np.fft.fftfreq(nx, d=dx)
#     ky = np.fft.fftfreq(ny, d=dy)
#     kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="xy")
#     k_mag = np.sqrt(kx_grid**2 + ky_grid**2)

#     valid_k = k_mag > 0
#     k_min = 1.0 / (max_wavelength_km * 1000.0)
#     k_max = np.nanmax(k_mag[valid_k])

#     k_bins = np.logspace(np.log10(k_min), np.log10(k_max), num=n_bins + 1)
#     k_centers = np.sqrt(k_bins[:-1] * k_bins[1:])

#     all_daily_spectra = []

#     # -------------------------
#     # 3. Loop over time
#     # -------------------------
#     for t in range(len(box_ds.time)):
#         u = box_ds.uo.isel(time=t).values
#         v = box_ds.vo.isel(time=t).values

#         if remove_mean:
#             u = u - np.nanmean(u)
#             v = v - np.nanmean(v)

#         # Replace NaNs if needed
#         u = np.nan_to_num(u, nan=0.0)
#         v = np.nan_to_num(v, nan=0.0)

#         # FFT of velocity components
#         u_hat = np.fft.fft2(u)
#         v_hat = np.fft.fft2(v)

#         # 2D KE power spectral density
#         # Normalization chosen to be consistent across snapshots/grids
#         psd_2d = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2) / (nx * ny)

#         # Radial binning
#         E_spectrum = np.full(len(k_bins) - 1, np.nan)
#         N_modes = np.zeros(len(k_bins) - 1)

#         for i in range(len(k_bins) - 1):
#             mask_bin = (k_mag >= k_bins[i]) & (k_mag < k_bins[i + 1])
#             N_modes[i] = np.sum(mask_bin)
#             if N_modes[i] > 0:
#                 E_spectrum[i] = np.nansum(psd_2d[mask_bin])

#         all_daily_spectra.append(E_spectrum)

#     mean_eds = np.nanmean(all_daily_spectra, axis=0)

#     # -------------------------
#     # 4. Nyquist cutoff
#     # -------------------------
#     f_nyquist = 1.0 / (2.0 * max(dx, dy))
#     nyq_mask = k_centers <= f_nyquist

#     return xr.DataArray(
#         mean_eds[nyq_mask],
#         coords=[("characteristic_length", 1.0 / k_centers[nyq_mask])],
#         name="energy_density_spectrum",
#         attrs={
#             "method": "velocity_fft_then_ke_psd",
#             "initialized_velocity": initialized_velocity,
#             "dx_m": dx,
#             "dy_m": dy,
#             "nyquist_km": (1.0 / f_nyquist) / 1000.0,
#         },
#     ).dropna(dim="characteristic_length")


def calculate_EDS(
    filepath,
    target_box=None,
    max_wavelength_km=400.0,
    initialized_velocity=False,
    x_res=None,
    y_res=None,
    n_bins=40,
    remove_mean=True,
    apply_hann_window=True,
    direction_half_width_deg=15.0,
):
    """
    Calculate an isotropic horizontal kinetic-energy spectrum from 2D velocity fields,
    plus a simple isotropy check based on zonal vs meridional directional spectra.

    Method
    ------
    1. Compute FFT of u and v separately
    2. Form 2D horizontal kinetic-energy spectrum
    3. Radially bin in |k| to get isotropic E(k)
    4. Compute directional spectra near zonal and meridional axes
    5. Average over time snapshots

    Notes
    -----
    - The returned isotropic spectrum is formed from the velocity field itself,
      not from a precomputed scalar KE field.
    - A Hann window is applied by default to reduce spectral leakage.
    - The isotropy check is intentionally simple:
          R(k) = E_zonal(k) / E_meridional(k)
      If R(k) ~ 1 over a range of scales, isotropic averaging is more defensible there.

    Parameters
    ----------
    filepath : str
        Path to NetCDF file.
    target_box : list or None
        If initialized_velocity=False:
            [lon_min, lon_max, lat_min, lat_max]
        If initialized_velocity=True:
            [x_min, x_max, y_min, y_max] in meters
        If None, full horizontal domain is used.
    max_wavelength_km : float
        Largest wavelength to include in binning.
    initialized_velocity : bool
        True for initialized MITgcm-style Cartesian x/y grids.
        False for lon/lat datasets.
    x_res, y_res : float or None
        Horizontal grid spacing in meters for initialized_velocity=True.
        If None, inferred from x and y coordinates.
    n_bins : int
        Number of logarithmic radial wavenumber bins.
    remove_mean : bool
        If True, subtract spatial mean from u and v before FFT.
    apply_hann_window : bool
        If True, apply a separable 2D Hann window before FFT.
    direction_half_width_deg : float
        Half-width of angular sectors used for zonal/meridional spectra.

    Returns
    -------
    xr.Dataset
        Dataset containing:
        - energy_density_spectrum(characteristic_length): isotropic shell-integrated KE spectrum
        - zonal_spectrum(characteristic_length): directional spectrum around kx-axis
        - meridional_spectrum(characteristic_length): directional spectrum around ky-axis
        - isotropy_ratio(characteristic_length): zonal / meridional
    """

    ds = xr.open_dataset(filepath)

    # ------------------------------------------------------------------
    # 1. Select horizontal box and infer grid spacing
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 2. Build wavenumber grid
    #    Units: cycles per meter, consistent with np.fft.fftfreq
    # ------------------------------------------------------------------
    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="xy")

    k_mag = np.sqrt(kx_grid**2 + ky_grid**2)
    theta = np.rad2deg(np.arctan2(ky_grid, kx_grid))  # angle in degrees, [-180, 180]

    valid_k = k_mag > 0
    if not np.any(valid_k):
        raise ValueError("No valid nonzero wavenumbers found.")

    k_min = 1.0 / (max_wavelength_km * 1000.0)
    k_max = np.nanmax(k_mag[valid_k])

    if k_min >= k_max:
        raise ValueError(
            "max_wavelength_km is too small for the selected domain/grid. "
            "Choose a larger max_wavelength_km or a larger spatial box."
        )

    k_bins = np.logspace(np.log10(k_min), np.log10(k_max), num=n_bins + 1)
    k_centers = np.sqrt(k_bins[:-1] * k_bins[1:])

    # ------------------------------------------------------------------
    # 3. Optional 2D Hann window
    # ------------------------------------------------------------------
    if apply_hann_window:
        wx = np.hanning(nx)
        wy = np.hanning(ny)
        window_2d = np.outer(wy, wx)

        # Normalize so mean(window^2) = 1, keeping spectral levels comparable
        window_norm = np.sqrt(np.mean(window_2d**2))
        if window_norm == 0:
            raise ValueError("Invalid window normalization.")
        window_2d = window_2d / window_norm
    else:
        window_2d = np.ones((ny, nx), dtype=float)

    # ------------------------------------------------------------------
    # 4. Direction masks for simple isotropy check
    #    Zonal: around 0° and 180°
    #    Meridional: around 90° and -90°
    # ------------------------------------------------------------------
    hw = float(direction_half_width_deg)

    def angle_close(angle_deg, center_deg, half_width_deg):
        diff = (angle_deg - center_deg + 180.0) % 360.0 - 180.0
        return np.abs(diff) <= half_width_deg

    zonal_mask = angle_close(theta, 0.0, hw) | angle_close(theta, 180.0, hw)
    meridional_mask = angle_close(theta, 90.0, hw) | angle_close(theta, -90.0, hw)

    all_iso_spectra = []
    all_zonal_spectra = []
    all_merid_spectra = []

    # ------------------------------------------------------------------
    # 5. Loop over time
    # ------------------------------------------------------------------
    n_time = len(box_ds.time)

    for t in range(n_time):
        u = box_ds.uo.isel(time=t).values
        v = box_ds.vo.isel(time=t).values

        # Replace NaNs first to avoid mean/window issues
        u = np.nan_to_num(u, nan=0.0)
        v = np.nan_to_num(v, nan=0.0)

        if remove_mean:
            u = u - np.mean(u)
            v = v - np.mean(v)

        # Apply window
        u = u * window_2d
        v = v * window_2d

        # FFTs
        u_hat = np.fft.fft2(u)
        v_hat = np.fft.fft2(v)

        # 2D horizontal KE spectrum
        # Shell-integrated form for relative spectral energy comparisons
        ke_2d = 0.5 * (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2) / (nx * ny)

        E_iso = np.full(n_bins, np.nan)
        E_zonal = np.full(n_bins, np.nan)
        E_merid = np.full(n_bins, np.nan)

        for i in range(n_bins):
            shell = (k_mag >= k_bins[i]) & (k_mag < k_bins[i + 1])

            shell_iso = shell
            shell_zonal = shell & zonal_mask
            shell_merid = shell & meridional_mask

            if np.any(shell_iso):
                E_iso[i] = np.nansum(ke_2d[shell_iso])

            if np.any(shell_zonal):
                E_zonal[i] = np.nansum(ke_2d[shell_zonal])

            if np.any(shell_merid):
                E_merid[i] = np.nansum(ke_2d[shell_merid])

        all_iso_spectra.append(E_iso)
        all_zonal_spectra.append(E_zonal)
        all_merid_spectra.append(E_merid)

    mean_iso = np.nanmean(np.asarray(all_iso_spectra), axis=0)
    mean_zonal = np.nanmean(np.asarray(all_zonal_spectra), axis=0)
    mean_merid = np.nanmean(np.asarray(all_merid_spectra), axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        isotropy_ratio = mean_zonal / mean_merid

    # ------------------------------------------------------------------
    # 6. Nyquist cutoff
    # ------------------------------------------------------------------
    f_nyquist = 1.0 / (2.0 * max(dx, dy))
    nyq_mask = k_centers <= f_nyquist

    characteristic_length = 1.0 / k_centers[nyq_mask]

    out = xr.Dataset(
        data_vars={
            "energy_density_spectrum": (
                ("characteristic_length",),
                mean_iso[nyq_mask],
            ),
            "zonal_spectrum": (
                ("characteristic_length",),
                mean_zonal[nyq_mask],
            ),
            "meridional_spectrum": (
                ("characteristic_length",),
                mean_merid[nyq_mask],
            ),
            "isotropy_ratio": (
                ("characteristic_length",),
                isotropy_ratio[nyq_mask],
            ),
        },
        coords={
            "characteristic_length": characteristic_length,
        },
        attrs={
            "method": "velocity_fft_to_2d_ke_then_shell_binning",
            "initialized_velocity": initialized_velocity,
            "dx_m": dx,
            "dy_m": dy,
            "nyquist_km": (1.0 / f_nyquist) / 1000.0,
            "window": "hann" if apply_hann_window else "none",
            "direction_half_width_deg": direction_half_width_deg,
            "notes": (
                "energy_density_spectrum is the isotropic shell-integrated horizontal KE spectrum. "
                "isotropy_ratio = zonal_spectrum / meridional_spectrum; values near 1 suggest "
                "approximate isotropy over that scale range."
            ),
        },
    )

    return out.dropna(dim="characteristic_length", how="all")