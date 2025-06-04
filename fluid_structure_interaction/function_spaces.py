import dolfin
from fenics import *

import load_mesh as lmsh

function_space_degree = 1

# function space for u
U = VectorFunctionSpace(lmsh.mesh, 'P', function_space_degree)
# function space for du/dt
U_dot = VectorFunctionSpace(lmsh.mesh, 'P', function_space_degree)
T = TensorFunctionSpace(lmsh.mesh, 'P', function_space_degree, shape=(lmsh.mesh.topology().dim(), lmsh.mesh.topology().dim()))

# Define variational problem
u = Function(U)
u_dot = Function(U_dot)
nu_u = TestFunction(U)
nu_u_dot = TestFunction(U_dot)
J_u = TrialFunction(U)
J_u_dot = TrialFunction(U_dot)
u_in = Function(U)
u_out = Function(U)
u_dot_in = Function(U_dot)
u_dot_out = Function(U_dot)
# f = Function(U)

t = Function(T)
