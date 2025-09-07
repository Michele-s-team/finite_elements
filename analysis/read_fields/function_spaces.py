from fenics import *

import mesh.load as lmsh

function_space_degree = 2

Q_u = FunctionSpace(lmsh.mesh, 'P', function_space_degree)
Q_v = VectorFunctionSpace(lmsh.mesh, 'P', function_space_degree, dim=3)
Q_t = TensorFunctionSpace(lmsh.mesh, 'P', function_space_degree, shape=(lmsh.mesh.topology().dim(), lmsh.mesh.topology().dim()))

# Define variational problem
u = Function(Q_u)
v = Function(Q_v)
t = Function(Q_t)

u_read = Function(Q_u)
