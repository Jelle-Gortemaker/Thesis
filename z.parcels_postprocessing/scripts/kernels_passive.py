"""
Passive tracer advection kernels.

This file intentionally contains only passive tracer dynamics. The slow-manifold
Maxey-Riley inertial-particle kernel lives in kernels_mr_sm.py.
"""


def advection_passive_rk4(particle, fieldset, time):
    """
    Passive RK4 advection on a flat Cartesian grid.

    Required fields:
        U(time, y, x), V(time, y, x) in m/s

    Notes
    -----
    Parcels uses particle.lon and particle.lat as the two horizontal coordinates.
    In this project these are actually flat-grid x and y coordinates in metres.
    """
    u1 = fieldset.U[time, particle.depth, particle.lat, particle.lon]
    v1 = fieldset.V[time, particle.depth, particle.lat, particle.lon]

    lon1 = particle.lon + 0.5 * particle.dt * u1
    lat1 = particle.lat + 0.5 * particle.dt * v1

    u2 = fieldset.U[time + 0.5 * particle.dt, particle.depth, lat1, lon1]
    v2 = fieldset.V[time + 0.5 * particle.dt, particle.depth, lat1, lon1]

    lon2 = particle.lon + 0.5 * particle.dt * u2
    lat2 = particle.lat + 0.5 * particle.dt * v2

    u3 = fieldset.U[time + 0.5 * particle.dt, particle.depth, lat2, lon2]
    v3 = fieldset.V[time + 0.5 * particle.dt, particle.depth, lat2, lon2]

    lon3 = particle.lon + particle.dt * u3
    lat3 = particle.lat + particle.dt * v3

    u4 = fieldset.U[time + particle.dt, particle.depth, lat3, lon3]
    v4 = fieldset.V[time + particle.dt, particle.depth, lat3, lon3]

    particle.up = (u1 + 2.0 * u2 + 2.0 * u3 + u4) / 6.0
    particle.vp = (v1 + 2.0 * v2 + 2.0 * v3 + v4) / 6.0

    particle.lon += particle.dt * particle.up
    particle.lat += particle.dt * particle.vp

    # Diagnostics are zero for passive tracers.
    particle.uslip = 0.0
    particle.vslip = 0.0
    particle.Rep = 0.0
    particle.C_Rep_current = 1.0
