import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import sys

def process_glorys_data(filepath, averaging_period, depth_idx=0, time_idx=0):
    """Loads, averages, and extracts the surface layer from GLORYS data."""
    ds = xr.open_dataset(filepath)
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

def preprocess_netcdf(surface_ds, les_box, active=True, subtract_mean=False):
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
    base_name = "GPGP_oct2020"
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


def calculate_EDS(filepath, target_box, max_wavelength_km=800.0):
    """
    Validated EDS: Returns the pre-multiplied spectrum E(l) in units of m²/s².
    This allows direct comparison with energy-scale reference slopes.
    """
    ds = xr.open_dataset(filepath)
    box_ds = ds.sel(longitude=slice(target_box[0], target_box[1]),
                    latitude=slice(target_box[2], target_box[3])).isel(depth=0)

    # 1. Grid Metrics
    R_earth = 6371000.0  
    phi_lat = float(box_ds.latitude.mean())
    dy = np.deg2rad(np.abs(np.mean(np.gradient(box_ds.latitude.values)))) * R_earth
    dx = np.deg2rad(np.abs(np.mean(np.gradient(box_ds.longitude.values)))) * R_earth * np.cos(np.deg2rad(phi_lat))

    n_lat, n_lon = len(box_ds.latitude), len(box_ds.longitude)
    nt = n_lat * n_lon # Total pixels

    # 2. Spectral Bins
    f_nyquist = 1.0 / (2.0 * max(dx, dy))
    k_min = 1.0 / (max_wavelength_km * 1000.0)
    k_bins = np.logspace(np.log10(k_min), np.log10(f_nyquist), num=41)
    k_centers = (k_bins[:-1] * k_bins[1:])**0.5
    dk = np.diff(k_bins) 

    # 3. Wavenumber Grid
    freq_x = np.fft.fftshift(np.fft.fftfreq(n_lon, d=dx))
    freq_y = np.fft.fftshift(np.fft.fftfreq(n_lat, d=dy))
    KX, KY = np.meshgrid(freq_x, freq_y)
    K_radial = np.sqrt(KX**2 + KY**2) 
    indices = np.digitize(K_radial, k_bins)
    
    all_daily_spectra = []
    for t in range(len(box_ds.time)):
        u = box_ds.uo.isel(time=t).values
        v = box_ds.vo.isel(time=t).values
        
        # FFT Normalization: Standard Parseval normalization (1/nt)
        u_fft = np.fft.fftshift(np.fft.fft2(u - np.nanmean(u))) / nt
        v_fft = np.fft.fftshift(np.fft.fft2(v - np.nanmean(v))) / nt
        
        # Power Spectral Density (Energy per mode)
        psd_2d = 0.5 * (np.abs(u_fft)**2 + np.abs(v_fft)**2)
        
        # 4. Integration to 1D Density E(k)
        daily_eds = []
        for i in range(1, len(k_bins)):
            mask = (indices == i)
            if np.any(mask):
                # Calculate E(k) density (Sum energy in bin / bin width)
                Ek_density = np.sum(psd_2d[mask]) / dk[i-1]
                
                # CONVERSION TO E(l): Multiply by k to get energy magnitude (m²/s²)
                # This aligns the data with standard reference slope levels
                daily_eds.append(Ek_density * k_centers[i-1])
            else:
                daily_eds.append(np.nan)
        all_daily_spectra.append(daily_eds)

    return xr.DataArray(np.nanmean(all_daily_spectra, axis=0), 
                        coords=[("characteristic_length", 1.0 / k_centers)], 
                        name="energy_density").dropna(dim="characteristic_length")



def calculate_EDS_alternative(filepath, target_box, max_wavelength_km=400.0):
    """
    Alternative EDS calculation following the scalar-KE FFT logic.
    Compatible with the user's data slicing and averaging workflow.
    """
    ds = xr.open_dataset(filepath)
    
    # Selection using index for depth and manual slice for box
    box_ds = ds.sel(
        longitude=slice(target_box[0], target_box[1]),
        latitude=slice(target_box[2], target_box[3])
    ).isel(depth=0)

    # 1. Determine Physical Grid Metrics (Thesis Eq. 5) [cite: 280]
    R_earth = 6371000.0  
    phi_lat = float(box_ds.latitude.mean())
    dlat_deg = np.abs(np.mean(np.gradient(box_ds.latitude.values)))
    dlon_deg = np.abs(np.mean(np.gradient(box_ds.longitude.values)))
    
    dy = np.deg2rad(dlat_deg) * R_earth
    dx = np.deg2rad(dlon_deg) * R_earth * np.cos(np.deg2rad(phi_lat))

    nx, ny = len(box_ds.latitude), len(box_ds.longitude)

    # 2. Wavenumber Axes and Grid
    kx = np.fft.fftfreq(nx, d=dx)  
    ky = np.fft.fftfreq(ny, d=dy) 
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
    k_mag = np.sqrt(kx_grid**2 + ky_grid**2)

    # 3. Bin Definition (Alternative Logic)
    valid_k = k_mag > 0
    k_min = 1.0 / (max_wavelength_km * 1000.0)
    # Using 40 bins as per alternative logic
    k_bins = np.logspace(np.log10(k_min), np.log10(k_mag.max()), num=41)
    k_centers = (k_bins[:-1] * k_bins[1:])**0.5
    
    all_daily_spectra = []

    # 4. Processing Loop: Snapshot-by-snapshot
    for t in range(len(box_ds.time)):
        u = box_ds.uo.isel(time=t).values
        v = box_ds.vo.isel(time=t).values

        # Alternative Step: Compute Kinetic Energy field in physical space first
        # Normalized by grid size as per alternative script logic
        E_l = 0.5 * (np.abs(u)**2 + np.abs(v)**2) / (nx * ny)

        # Alternative Step: FFT of the scalar KE field
        E_k = np.abs(np.fft.fft2(E_l))
        
        # Binning logic
        E_spectrum = np.zeros(len(k_bins)-1)
        N_modes = np.zeros(len(k_bins)-1)
        
        for k in range(len(k_bins)-1):
            mask_bin = (k_mag >= k_bins[k]) & (k_mag < k_bins[k+1])
            E_spectrum[k] = np.sum(E_k[mask_bin])
            N_modes[k] = np.sum(mask_bin)

        # Average by modes in each bin
        valid_bins = N_modes > 0
        E_spectrum[valid_bins] /= N_modes[valid_bins]
        all_daily_spectra.append(E_spectrum)

    # 5. Averaging snapshots over the month [cite: 207]
    mean_eds = np.nanmean(all_daily_spectra, axis=0)
    
    # 6. Apply Nyquist Cutoff (Thesis Eq. 7) [cite: 292]
    f_nyquist = 1.0 / (2.0 * max(dx, dy))
    nyq_mask = (k_centers <= f_nyquist)

    return xr.DataArray(
        mean_eds[nyq_mask],
        coords=[("characteristic_length", 1.0 / k_centers[nyq_mask])],
        name="energy_density",
        attrs={"nyquist_km": (1.0/f_nyquist)/1000.0}
    ).dropna(dim="characteristic_length")