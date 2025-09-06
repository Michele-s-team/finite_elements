from fenics import *

import mesh.load as lmsh

function_space_degree = 2

Q = FunctionSpace(lmsh.mesh, 'P', function_space_degree)
V = VectorFunctionSpace(lmsh.mesh, 'P', function_space_degree)
T = TensorFunctionSpace(lmsh.mesh, 'P', function_space_degree, shape=(lmsh.mesh.topology().dim(), lmsh.mesh.topology().dim()))

# Define variational problem
u = Function(Q)
v = Function(V)
t = Function(T)
