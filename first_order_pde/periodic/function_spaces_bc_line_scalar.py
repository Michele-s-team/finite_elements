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

Q = FunctionSpace(lmsh.mesh, 'P', rpam.parameters['function_space_degree'], constrained_domain=periodic_boundary)
V = VectorFunctionSpace(lmsh.mesh, 'P', rpam.parameters['function_space_degree'], constrained_domain=periodic_boundary)


# Define variational problem
u = Function(Q)
nu_u = TestFunction(Q)
f = Function(Q)
u_exact = Function(Q)
J_u = TrialFunction(Q)

# Define gradient of u for post-processing (pp) variational problem
grad_u = Function(V)
nu_grad_u = TestFunction(V)
J_grad_u = TrialFunction(V)
grad_u_exact = Function(V)
