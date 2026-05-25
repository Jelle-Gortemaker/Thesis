def age_particle(particle, fieldset, time):
    particle.age += abs(particle.dt)


def periodic_xy(particle, fieldset, time):
    """Wrap particles periodically in the flat MITgcm x-y domain."""
    if particle.lon < fieldset.x_edge_min:
        particle.lon += fieldset.Lx
    elif particle.lon >= fieldset.x_edge_max:
        particle.lon -= fieldset.Lx

    if particle.lat < fieldset.y_edge_min:
        particle.lat += fieldset.Ly
    elif particle.lat >= fieldset.y_edge_max:
        particle.lat -= fieldset.Ly