from fenics import *
import importlib

import mesh.load as lmsh
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)


# This enforces periodic boundary conditions which map the l vertex into the r vertex
class PeriodicBoundary(SubDomain):
    # Identify the "target domain": the left vertex
    def inside(self, x, on_boundary):
        return near(x[0], rmsh.parameters['x_l']) and on_boundary

    # Map the other boundaries to the "target domain"
    def map(self, x, y):
        if near(x[0], rmsh.parameters['x_r']):
            # right vertex → left vertex
            y[0] = rmsh.parameters['x_l']
        else:
            # Required: set unmapped points to identity
            y[0] = x[0]


periodic_boundary = PeriodicBoundary()

Q = VectorFunctionSpace(lmsh.mesh, 'P', rpam.parameters['function_space_degree'], dim=2, constrained_domain=periodic_boundary)
V = TensorFunctionSpace(lmsh.mesh, 'P', rpam.parameters['function_space_degree'], shape=(2, 1), constrained_domain=periodic_boundary)


# Define variational problem
u_n = Function(Q)
u_n_1 = Function(Q)
nu_u_n = TestFunction(Q)
v = Function(Q)
#y_s_notes
ys = Function(Q)
J_u_n = TrialFunction(Q)