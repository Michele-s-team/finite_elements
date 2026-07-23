from fenics import *
import mesh.load as lmsh

class PeriodicBoundary(SubDomain):

    def inside(self, x, on_boundary):
        return bool(
            near(x[0], 0.0) and on_boundary
        )

    def map(self, x, y):

        L = lmsh.parameters['L']

        y[0] = x[0] - L