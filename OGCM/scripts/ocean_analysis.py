import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def process_glorys_data(filepath, averaging_period, depth_idx=0, time_idx=0):
    """Loads, averages, and extracts the surface layer from GLORYS data."""
    ds = xr.open_dataset(filepath)
    ds_resampled = ds.resample(time=averaging_period).mean()
    surface_ds = ds_resampled.isel(depth=depth_idx, time=time_idx)
    return surface_ds

def get_subdomain(center_lat, center_lon):
    """
    Calculates the lat/lon bounds for an exact 200x200 km box.
    100km in each direction from center.
    """
    R = 6371.0  # Earth radius in km
    
    # Latitude offset (constant: 1 deg lat approx 111km)
    lat_offset = (100.0 / R) * (180.0 / np.pi)
    
    # Longitude offset (shrinks as you move away from equator)
    lon_offset = (100.0 / (R * np.cos(np.deg2rad(center_lat)))) * (180.0 / np.pi)
    
    return [center_lon - lon_offset, center_lon + lon_offset, 
            center_lat - lat_offset, center_lat + lat_offset]

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
        # Drawing the 200km box
        width = target_box[1] - target_box[0]
        height = target_box[3] - target_box[2]
        rect = patches.Rectangle((target_box[0], target_box[2]), width, height,
                                 linewidth=3, edgecolor='red', facecolor='none', 
                                 transform=ccrs.PlateCarree(), zorder=10)
        ax.add_patch(rect)
        
        # FIX: Explicitly naming x, y, and s to avoid positional argument mismatch
        ax.text(x=target_box[0], 
                y=target_box[3] + 0.15, 
                s="LES Study Area (200km)", 
                color='red', fontweight='bold', 
                transform=ccrs.PlateCarree())

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