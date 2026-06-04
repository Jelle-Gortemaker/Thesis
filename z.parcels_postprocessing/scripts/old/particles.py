from parcels import JITParticle, Variable
import numpy as np


class FloatingParticle(JITParticle):
    """
    Particle class for passive and inertial particles.

    Written to output:
        release_id
        particle_class_id
        tau_p
        stokes_number

    Internal only:
        up, vp

    particle_class_id:
        0 = passive
        1 = inertial
    """

    release_id = Variable("release_id", dtype=np.int32, initial=0, to_write="once")
    particle_class_id = Variable("particle_class_id", dtype=np.int32, initial=0, to_write="once")

    tau_p = Variable("tau_p", dtype=np.float32, initial=0.0, to_write="once")
    stokes_number = Variable("stokes_number", dtype=np.float32, initial=0.0, to_write="once")

    # Internal inertial particle velocity. Do not write to output.
    up = Variable("up", dtype=np.float32, initial=0.0, to_write=False)
    vp = Variable("vp", dtype=np.float32, initial=0.0, to_write=False)