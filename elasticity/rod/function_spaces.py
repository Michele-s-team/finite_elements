import dolfin
from fenics import *

import load_mesh as lmsh

function_space_degree = 1

# function space for u
R = FunctionSpace(lmsh.mesh, 'P', function_space_degree)
U = VectorFunctionSpace(lmsh.mesh, 'P', function_space_degree)

# Define variational problem
u = Function(U)
g = Function(U)
nu_u = TestFunction(U)
J_u = TrialFunction(U)

u_l = Function(U)
rho = Function(R)

