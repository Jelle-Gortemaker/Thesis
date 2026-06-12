"""
Slow-manifold Maxey-Riley kernels for flat MITgcm/Parcels simulations.

This file intentionally contains only the MR-SM inertial-particle dynamics.
Passive tracer advection lives in kernels_passive.py.
"""

import math


def advection_mr_sm_rk4(particle, fieldset, time):
    """
    Simplified tau-only inertial advection without explicit Coriolis.

    Effective particle velocity:

        u_p = U + tau_p * DU/Dt
        v_p = V + tau_p * DV/Dt

    where:

        DU/Dt = dUdt + U*dUdx + V*dUdy
        DV/Dt = dVdt + U*dVdx + V*dVdy

    tau_p is prescribed directly by the user.

    Diameter, buoyancy, viscosity, drag correction, Rep, and explicit Coriolis
    are not used in this simplified model.
    """

    tau = particle.tau_p

    # ============================================================
    # RK4 stage 1
    # ============================================================
    uf1 = fieldset.U[time, particle.depth, particle.lat, particle.lon]
    vf1 = fieldset.V[time, particle.depth, particle.lat, particle.lon]

    dudt1 = fieldset.dUdt[time, particle.depth, particle.lat, particle.lon]
    dvdt1 = fieldset.dVdt[time, particle.depth, particle.lat, particle.lon]
    dudx1 = fieldset.dUdx[time, particle.depth, particle.lat, particle.lon]
    dudy1 = fieldset.dUdy[time, particle.depth, particle.lat, particle.lon]
    dvdx1 = fieldset.dVdx[time, particle.depth, particle.lat, particle.lon]
    dvdy1 = fieldset.dVdy[time, particle.depth, particle.lat, particle.lon]

    DuDt1 = dudt1 + uf1 * dudx1 + vf1 * dudy1
    DvDt1 = dvdt1 + uf1 * dvdx1 + vf1 * dvdy1

    u1 = uf1 + tau * DuDt1
    v1 = vf1 + tau * DvDt1

    lon1 = particle.lon + 0.5 * particle.dt * u1
    lat1 = particle.lat + 0.5 * particle.dt * v1
    time1 = time + 0.5 * particle.dt

    # ============================================================
    # RK4 stage 2
    # ============================================================
    uf2 = fieldset.U[time1, particle.depth, lat1, lon1]
    vf2 = fieldset.V[time1, particle.depth, lat1, lon1]

    dudt2 = fieldset.dUdt[time1, particle.depth, lat1, lon1]
    dvdt2 = fieldset.dVdt[time1, particle.depth, lat1, lon1]
    dudx2 = fieldset.dUdx[time1, particle.depth, lat1, lon1]
    dudy2 = fieldset.dUdy[time1, particle.depth, lat1, lon1]
    dvdx2 = fieldset.dVdx[time1, particle.depth, lat1, lon1]
    dvdy2 = fieldset.dVdy[time1, particle.depth, lat1, lon1]

    DuDt2 = dudt2 + uf2 * dudx2 + vf2 * dudy2
    DvDt2 = dvdt2 + uf2 * dvdx2 + vf2 * dvdy2

    u2 = uf2 + tau * DuDt2
    v2 = vf2 + tau * DvDt2

    lon2 = particle.lon + 0.5 * particle.dt * u2
    lat2 = particle.lat + 0.5 * particle.dt * v2
    time2 = time + 0.5 * particle.dt

    # ============================================================
    # RK4 stage 3
    # ============================================================
    uf3 = fieldset.U[time2, particle.depth, lat2, lon2]
    vf3 = fieldset.V[time2, particle.depth, lat2, lon2]

    dudt3 = fieldset.dUdt[time2, particle.depth, lat2, lon2]
    dvdt3 = fieldset.dVdt[time2, particle.depth, lat2, lon2]
    dudx3 = fieldset.dUdx[time2, particle.depth, lat2, lon2]
    dudy3 = fieldset.dUdy[time2, particle.depth, lat2, lon2]
    dvdx3 = fieldset.dVdx[time2, particle.depth, lat2, lon2]
    dvdy3 = fieldset.dVdy[time2, particle.depth, lat2, lon2]

    DuDt3 = dudt3 + uf3 * dudx3 + vf3 * dudy3
    DvDt3 = dvdt3 + uf3 * dvdx3 + vf3 * dvdy3

    u3 = uf3 + tau * DuDt3
    v3 = vf3 + tau * DvDt3

    lon3 = particle.lon + particle.dt * u3
    lat3 = particle.lat + particle.dt * v3
    time3 = time + particle.dt

    # ============================================================
    # RK4 stage 4
    # ============================================================
    uf4 = fieldset.U[time3, particle.depth, lat3, lon3]
    vf4 = fieldset.V[time3, particle.depth, lat3, lon3]

    dudt4 = fieldset.dUdt[time3, particle.depth, lat3, lon3]
    dvdt4 = fieldset.dVdt[time3, particle.depth, lat3, lon3]
    dudx4 = fieldset.dUdx[time3, particle.depth, lat3, lon3]
    dudy4 = fieldset.dUdy[time3, particle.depth, lat3, lon3]
    dvdx4 = fieldset.dVdx[time3, particle.depth, lat3, lon3]
    dvdy4 = fieldset.dVdy[time3, particle.depth, lat3, lon3]

    DuDt4 = dudt4 + uf4 * dudx4 + vf4 * dudy4
    DvDt4 = dvdt4 + uf4 * dvdx4 + vf4 * dvdy4

    u4 = uf4 + tau * DuDt4
    v4 = vf4 + tau * DvDt4

    # ============================================================
    # Final RK4 update and diagnostics
    # ============================================================
    particle.up = (u1 + 2.0 * u2 + 2.0 * u3 + u4) / 6.0
    particle.vp = (v1 + 2.0 * v2 + 2.0 * v3 + v4) / 6.0

    particle.lon += particle.dt * particle.up
    particle.lat += particle.dt * particle.vp

    particle.uslip = particle.up - uf4
    particle.vslip = particle.vp - vf4
    particle.Rep = 0.0
    particle.C_Rep_current = 1.0
