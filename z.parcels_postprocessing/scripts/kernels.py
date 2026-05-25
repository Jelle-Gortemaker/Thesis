def advection_passive_or_inertial(particle, fieldset, time):
    """
    Passive:
        RK4 advection with the fluid velocity.

    Inertial:
        velocity relaxation model:
            dxp/dt = vp
            dvp/dt = (u(xp,t) - vp) / tau_p

    particle_class_id:
        0 = passive
        1 = inertial
    """

    # ========================================================
    # Passive particles: RK4
    # ========================================================
    if particle.particle_class_id < 1:
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

    # ========================================================
    # Inertial particles: relaxation model
    # ========================================================
    else:
        u = fieldset.U[time, particle.depth, particle.lat, particle.lon]
        v = fieldset.V[time, particle.depth, particle.lat, particle.lon]

        if particle.tau_p > 0.0:
            relax = particle.dt / particle.tau_p

            # Stable implicit relaxation step.
            particle.up = (particle.up + relax * u) / (1.0 + relax)
            particle.vp = (particle.vp + relax * v) / (1.0 + relax)
        else:
            particle.up = u
            particle.vp = v

        particle.lon += particle.dt * particle.up
        particle.lat += particle.dt * particle.vp


def periodic_xy(particle, fieldset, time):
    if particle.lon < fieldset.x_edge_min:
        particle.lon += fieldset.Lx
    elif particle.lon >= fieldset.x_edge_max:
        particle.lon -= fieldset.Lx

    if particle.lat < fieldset.y_edge_min:
        particle.lat += fieldset.Ly
    elif particle.lat >= fieldset.y_edge_max:
        particle.lat -= fieldset.Ly