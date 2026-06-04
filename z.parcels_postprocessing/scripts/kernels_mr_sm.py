"""
Slow-manifold Maxey-Riley kernels for flat MITgcm/Parcels simulations.

This file intentionally contains only the MR-SM inertial-particle dynamics.
Passive tracer advection lives in kernels_passive.py.
"""

import math


def advection_mr_sm_rk4(particle, fieldset, time):
    """
    Slow-manifold Maxey-Riley advection with RK4 position integration.

    Required fields
    ---------------
    U, V : m/s
    dUdt, dVdt : m/s2
    dUdx, dUdy, dVdx, dVdy : 1/s

    Required constants
    ------------------
    f0 : Coriolis parameter [1/s]
    nu : kinematic viscosity [m2/s]
    mr_drag_mode : 0 none, 1 constant, 2 flexible
    Rep_max : cap on particle Reynolds number for numerical robustness

    Particle variables
    ------------------
    B : density ratio rho_particle / rho_fluid
    diameter : particle diameter [m]
    tau_p : Stokes relaxation time tau_s [s]
    C_Rep : constant drag correction when mr_drag_mode == 1

    Model
    -----
    u_p = U + tau_s/C(Rep) * beta * (DU/Dt - f0 V)
    v_p = V + tau_s/C(Rep) * beta * (DV/Dt + f0 U)
    beta = 2(1-B)/(1+2B)
    """
    beta = 2.0 * (1.0 - particle.B) / (1.0 + 2.0 * particle.B)

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

    uslip_est1 = particle.up - uf1
    vslip_est1 = particle.vp - vf1
    Rep1 = 0.0
    C1 = particle.C_Rep

    if fieldset.mr_drag_mode < 0.5:
        C1 = 1.0
    elif fieldset.mr_drag_mode > 1.5:
        if particle.diameter > 0.0 and fieldset.nu > 0.0:
            Rep1 = math.sqrt(uslip_est1 * uslip_est1 + vslip_est1 * vslip_est1) * particle.diameter / fieldset.nu
            if Rep1 > fieldset.Rep_max:
                Rep1 = fieldset.Rep_max
            C1 = 1.0 + Rep1 / (4.0 * (1.0 + math.sqrt(Rep1))) + Rep1 / 60.0
        else:
            C1 = 1.0
    else:
        if particle.diameter > 0.0 and fieldset.nu > 0.0:
            Rep1 = math.sqrt(uslip_est1 * uslip_est1 + vslip_est1 * vslip_est1) * particle.diameter / fieldset.nu
            if Rep1 > fieldset.Rep_max:
                Rep1 = fieldset.Rep_max

    if C1 <= 0.0:
        C1 = 1.0

    tau_eff1 = particle.tau_p / C1
    u1 = uf1 + tau_eff1 * beta * (DuDt1 - fieldset.f0 * vf1)
    v1 = vf1 + tau_eff1 * beta * (DvDt1 + fieldset.f0 * uf1)

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

    uslip_est2 = u1 - uf2
    vslip_est2 = v1 - vf2
    Rep2 = 0.0
    C2 = particle.C_Rep

    if fieldset.mr_drag_mode < 0.5:
        C2 = 1.0
    elif fieldset.mr_drag_mode > 1.5:
        if particle.diameter > 0.0 and fieldset.nu > 0.0:
            Rep2 = math.sqrt(uslip_est2 * uslip_est2 + vslip_est2 * vslip_est2) * particle.diameter / fieldset.nu
            if Rep2 > fieldset.Rep_max:
                Rep2 = fieldset.Rep_max
            C2 = 1.0 + Rep2 / (4.0 * (1.0 + math.sqrt(Rep2))) + Rep2 / 60.0
        else:
            C2 = 1.0
    else:
        if particle.diameter > 0.0 and fieldset.nu > 0.0:
            Rep2 = math.sqrt(uslip_est2 * uslip_est2 + vslip_est2 * vslip_est2) * particle.diameter / fieldset.nu
            if Rep2 > fieldset.Rep_max:
                Rep2 = fieldset.Rep_max

    if C2 <= 0.0:
        C2 = 1.0

    tau_eff2 = particle.tau_p / C2
    u2 = uf2 + tau_eff2 * beta * (DuDt2 - fieldset.f0 * vf2)
    v2 = vf2 + tau_eff2 * beta * (DvDt2 + fieldset.f0 * uf2)

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

    uslip_est3 = u2 - uf3
    vslip_est3 = v2 - vf3
    Rep3 = 0.0
    C3 = particle.C_Rep

    if fieldset.mr_drag_mode < 0.5:
        C3 = 1.0
    elif fieldset.mr_drag_mode > 1.5:
        if particle.diameter > 0.0 and fieldset.nu > 0.0:
            Rep3 = math.sqrt(uslip_est3 * uslip_est3 + vslip_est3 * vslip_est3) * particle.diameter / fieldset.nu
            if Rep3 > fieldset.Rep_max:
                Rep3 = fieldset.Rep_max
            C3 = 1.0 + Rep3 / (4.0 * (1.0 + math.sqrt(Rep3))) + Rep3 / 60.0
        else:
            C3 = 1.0
    else:
        if particle.diameter > 0.0 and fieldset.nu > 0.0:
            Rep3 = math.sqrt(uslip_est3 * uslip_est3 + vslip_est3 * vslip_est3) * particle.diameter / fieldset.nu
            if Rep3 > fieldset.Rep_max:
                Rep3 = fieldset.Rep_max

    if C3 <= 0.0:
        C3 = 1.0

    tau_eff3 = particle.tau_p / C3
    u3 = uf3 + tau_eff3 * beta * (DuDt3 - fieldset.f0 * vf3)
    v3 = vf3 + tau_eff3 * beta * (DvDt3 + fieldset.f0 * uf3)

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

    uslip_est4 = u3 - uf4
    vslip_est4 = v3 - vf4
    Rep4 = 0.0
    C4 = particle.C_Rep

    if fieldset.mr_drag_mode < 0.5:
        C4 = 1.0
    elif fieldset.mr_drag_mode > 1.5:
        if particle.diameter > 0.0 and fieldset.nu > 0.0:
            Rep4 = math.sqrt(uslip_est4 * uslip_est4 + vslip_est4 * vslip_est4) * particle.diameter / fieldset.nu
            if Rep4 > fieldset.Rep_max:
                Rep4 = fieldset.Rep_max
            C4 = 1.0 + Rep4 / (4.0 * (1.0 + math.sqrt(Rep4))) + Rep4 / 60.0
        else:
            C4 = 1.0
    else:
        if particle.diameter > 0.0 and fieldset.nu > 0.0:
            Rep4 = math.sqrt(uslip_est4 * uslip_est4 + vslip_est4 * vslip_est4) * particle.diameter / fieldset.nu
            if Rep4 > fieldset.Rep_max:
                Rep4 = fieldset.Rep_max

    if C4 <= 0.0:
        C4 = 1.0

    tau_eff4 = particle.tau_p / C4
    u4 = uf4 + tau_eff4 * beta * (DuDt4 - fieldset.f0 * vf4)
    v4 = vf4 + tau_eff4 * beta * (DvDt4 + fieldset.f0 * uf4)

    # ============================================================
    # Final RK4 update and diagnostics
    # ============================================================
    particle.up = (u1 + 2.0 * u2 + 2.0 * u3 + u4) / 6.0
    particle.vp = (v1 + 2.0 * v2 + 2.0 * v3 + v4) / 6.0

    particle.lon += particle.dt * particle.up
    particle.lat += particle.dt * particle.vp

    particle.uslip = u4 - uf4
    particle.vslip = v4 - vf4
    particle.Rep = Rep4
    particle.C_Rep_current = C4
