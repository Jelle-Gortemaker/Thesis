"""Parcels kernel for accumulating fluid-divergence exposure.

The particle position represents the end of the current time
step, and ``time + particle.dt`` is sampled. Together with the divergence
stored from the previous endpoint, this gives a trapezoidal 300-s integral.

The accumulated quantities describe the *fluid* divergence sampled along a
particle trajectory. For inertial particles this is not the divergence of the
effective particle-velocity field.
"""


def accumulate_fluid_divergence_exposure(particle, fieldset, time):
    """Accumulate signed divergence, convergence and residence time."""
    sample_time = time + particle.dt
    current_divergence = fieldset.div[
        sample_time,
        particle.depth,
        particle.lat,
        particle.lon,
    ]

    dt_seconds = particle.dt
    if dt_seconds < 0.0:
        dt_seconds = -dt_seconds

    previous_divergence = particle.previous_divergence

    previous_convergence = 0.0
    current_convergence = 0.0
    previous_positive_divergence = 0.0
    current_positive_divergence = 0.0
    previous_convergent_flag = 0.0
    current_convergent_flag = 0.0

    if previous_divergence < 0.0:
        previous_convergence = -previous_divergence
        previous_convergent_flag = 1.0
    else:
        previous_positive_divergence = previous_divergence

    if current_divergence < 0.0:
        current_convergence = -current_divergence
        current_convergent_flag = 1.0
    else:
        current_positive_divergence = current_divergence

    particle.cumulative_signed_divergence += (
        0.5 * (previous_divergence + current_divergence) * dt_seconds
    )
    particle.cumulative_convergence_exposure += (
        0.5 * (previous_convergence + current_convergence) * dt_seconds
    )
    particle.cumulative_divergence_exposure += (
        0.5
        * (previous_positive_divergence + current_positive_divergence)
        * dt_seconds
    )
    particle.convergent_residence_time_seconds += (
        0.5
        * (previous_convergent_flag + current_convergent_flag)
        * dt_seconds
    )
    particle.valid_exposure_time_seconds += dt_seconds

    particle.divergence_instantaneous = current_divergence
    particle.previous_divergence = current_divergence
