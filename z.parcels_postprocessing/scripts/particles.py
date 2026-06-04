from parcels import JITParticle, Variable
import numpy as np


class FloatingParticle(JITParticle):
    """
    Shared particle class for passive tracers and slow-manifold Maxey-Riley particles.

    Parcels cannot write string labels as particle variables, so the human-readable
    class labels are stored in the sidecar metadata JSON written by run.py. The
    numeric particle characteristics are written into the trajectory output so they
    can always be recovered later.

    particle_class_id:
        0 = passive tracer
        1 = slow-manifold Maxey-Riley particle

    drag_mode_id:
        0 = no drag correction, C(Rep)=1
        1 = constant C_Rep
        2 = flexible C(Rep) from instantaneous slip estimate
    """

    release_id = Variable("release_id", dtype=np.int32, initial=0, to_write="once")
    particle_class_id = Variable("particle_class_id", dtype=np.int32, initial=0, to_write="once")
    drag_mode_id = Variable("drag_mode_id", dtype=np.int32, initial=0, to_write="once")

    # Physical / model parameters written once for later filtering.
    B = Variable("B", dtype=np.float32, initial=1.0, to_write="once")
    diameter = Variable("diameter", dtype=np.float32, initial=0.0, to_write="once")
    tau_p = Variable("tau_p", dtype=np.float32, initial=0.0, to_write="once")
    tau_eff_nominal = Variable("tau_eff_nominal", dtype=np.float32, initial=0.0, to_write="once")
    C_Rep = Variable("C_Rep", dtype=np.float32, initial=1.0, to_write="once")
    stokes_number = Variable("stokes_number", dtype=np.float32, initial=0.0, to_write="once")

    # Diagnostics written every output step.
    Rep = Variable("Rep", dtype=np.float32, initial=0.0, to_write=True)
    uslip = Variable("uslip", dtype=np.float32, initial=0.0, to_write=True)
    vslip = Variable("vslip", dtype=np.float32, initial=0.0, to_write=True)
    C_Rep_current = Variable("C_Rep_current", dtype=np.float32, initial=1.0, to_write=True)

    # Effective particle velocity [m/s], useful for diagnostics.
    up = Variable("up", dtype=np.float32, initial=0.0, to_write=True)
    vp = Variable("vp", dtype=np.float32, initial=0.0, to_write=True)
