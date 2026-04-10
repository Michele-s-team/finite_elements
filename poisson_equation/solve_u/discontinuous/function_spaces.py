from fenics import *

import mesh.load as lmsh
import parameters.read.solution as rpam 

Q = FunctionSpace(lmsh.mesh, 'DG', rpam.parameters['function_space_degree'])
# V = VectorFunctionSpace(lmsh.mesh, 'DG', rpam.parameters['function_space_degree'])
# T = TensorFunctionSpace(lmsh.mesh, 'DG', rpam.parameters['function_space_degree'], shape=(lmsh.mesh.topology().dim(), lmsh.mesh.topology().dim()))

# Define variational problem
u = Function(Q)
nu_u = TestFunction(Q)
f = Function(Q)

# grad_u = Function(V)
J_u = TrialFunction(Q)
u_exact = Function(Q)

# Define post-processing (pp) variational problem
# hess_u is a tensor which is the Hessian matrix of u: hess_u[i, j] = \partial_i \partial_j u
# hess_u = Function(T)
# nu_hess_u = TestFunction(T)
# hess_u_exact = Function(T)
# J_hess_u = TrialFunction(T)
