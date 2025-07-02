from fenics import *
import importlib

import load_mesh as lmsh
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)


'''
This enforces periodic boundary conditions between the left and right edge of the rectangle
'''
class PeriodicBoundary(SubDomain):
    # Identify the "target domain": the origin corner (bottom-left)
    def inside(self, x, on_boundary):
        return near(x[0], 0) and on_boundary

    # Map the other boundaries to the "target domain"
    def map(self, x, y):

        if near(x[0], rmsh.parameters["L"]):
            # Right edge → left edge
            y[0] = x[0] - rmsh.parameters["L"]
            y[1] = x[1]
        else:
            # Required: set unmapped points to identity
            y[0] = x[0]
            y[1] = x[1]

function_space_degree = 1

periodic_boundary = PeriodicBoundary()

# function space for u
R = FunctionSpace(lmsh.mesh, 'P', function_space_degree, constrained_domain=periodic_boundary)
U = VectorFunctionSpace(lmsh.mesh, 'P', function_space_degree, constrained_domain=periodic_boundary)

# Define variational problem
u = Function(U)
g = Function(U)
nu_u = TestFunction(U)
J_u = TrialFunction(U)

u_l = Function(U)
rho = Function(R)

