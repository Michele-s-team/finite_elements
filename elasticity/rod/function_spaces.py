import dolfin
from fenics import *

import load_mesh as lmsh

function_space_degree = 1

# function space for u
U = VectorFunctionSpace(lmsh.mesh, 'P', function_space_degree)

# Define variational problem
u = Function(U)
nu_u = TestFunction(U)
J_u = TrialFunction(U)

u_l = Function(U)
