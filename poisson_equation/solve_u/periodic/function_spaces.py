import dolfin
from fenics import *
import importlib

import load_2d_mesh as lmsh
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

function_space_degree = 4

'''
This enforces periodic boundary conditions between the left and right edge of the rectangle

'''


class PeriodicBoundary(SubDomain):
    # Identify the "target domain": the origin corner (bottom-left)
    def inside(self, x, on_boundary):
        return (near(x[0], 0) or near(x[1], 0)) and on_boundary

    # Map the other boundaries to the "target domain"
    def map(self, x, y):
        if near(x[0], rmsh.L) and near(x[1], rmsh.h):
            # Top-right corner → bottom-left corner
            y[0] = x[0] - rmsh.L
            y[1] = x[1] - rmsh.h
        elif near(x[0], rmsh.L):
            # Right edge → left edge
            y[0] = x[0] - rmsh.L
            y[1] = x[1]
        elif near(x[1], rmsh.h):
            # Top edge → bottom edge
            y[0] = x[0]
            y[1] = x[1] - rmsh.h
        else:
            # Required: set unmapped points to identity
            y[0] = x[0]
            y[1] = x[1]


periodic_boundary = PeriodicBoundary()

Q = FunctionSpace(lmsh.mesh, 'P', function_space_degree, constrained_domain=periodic_boundary)
V = VectorFunctionSpace(lmsh.mesh, 'P', function_space_degree, constrained_domain=periodic_boundary)
T = TensorFunctionSpace(lmsh.mesh, 'P', function_space_degree, shape=(2, 2), constrained_domain=periodic_boundary)

# Define variational problem
u = Function(Q)
nu_u = TestFunction(Q)
f = Function(Q)
grad_u = Function(V)
J_u = TrialFunction(Q)
u_exact = Function(Q)

# Define post-processing (pp) variational problem
# hess_u is a tensor which is the Hessian matrix of u: hess_u[i, j] = \partial_i \partial_j u
hess_u = Function(T)
nu_hess_u = TestFunction(T)
hess_u_exact = Function(T)
J_hess_u = TrialFunction(T)
