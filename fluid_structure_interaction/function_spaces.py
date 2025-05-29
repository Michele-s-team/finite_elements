import dolfin
from fenics import *

import load_mesh as lmsh

function_space_degree = 1

U = VectorFunctionSpace(lmsh.mesh, 'P', function_space_degree)

# Define variational problem
u = Function(U)
nu_u = TestFunction(U)
J_u = TrialFunction(U)
u_exact = Function(U)
f = Function(U)

# Define post-processing (pp) variational problem
# hess_u is a tensor which is the Hessian matrix of u: hess_u[i, j] = \partial_i \partial_j u
# hess_u = Function(T)
# nu_hess_u = TestFunction(T)
# hess_u_exact = Function(T)
# J_hess_u = TrialFunction(T)
