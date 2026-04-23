from parcels import JITParticle, Variable


class SurfaceParticle(JITParticle):
    age = Variable("age", dtype=float, initial=0.0)


class DepthParticle(JITParticle):
    age = Variable("age", dtype=float, initial=0.0)