"""
Common Parcels kernels for the flat, periodic MITgcm particle simulations.
"""


def periodic_xy(particle, fieldset, time):
    """
    Periodic wrapping in a flat Cartesian x/y domain.

    Requires fieldset constants:
        x_edge_min, x_edge_max, y_edge_min, y_edge_max, Lx, Ly
    """
    if particle.lon < fieldset.x_edge_min:
        particle.lon += fieldset.Lx
    elif particle.lon >= fieldset.x_edge_max:
        particle.lon -= fieldset.Lx

    if particle.lat < fieldset.y_edge_min:
        particle.lat += fieldset.Ly
    elif particle.lat >= fieldset.y_edge_max:
        particle.lat -= fieldset.Ly
