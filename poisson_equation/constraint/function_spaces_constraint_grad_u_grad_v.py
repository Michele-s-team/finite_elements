from fenics import *

import read_parameters_solve as rpam
import load_mesh as lmsh

# define elements and function spaces
P_u = FiniteElement('P', triangle, rpam.parameters['function_space_degree'])
P_v = FiniteElement('P', triangle, rpam.parameters['function_space_degree'])
element = MixedElement([P_u, P_v])
Q = FunctionSpace(lmsh.mesh, element)

Q_u = Q.sub(0).collapse()
Q_v = Q.sub(1).collapse()

# function space used to store the gradient of v
Q_grad_v = VectorFunctionSpace(lmsh.mesh, 'P', rpam.parameters['function_space_degree'])


# Define functions
psi = Function(Q)
J_psi = TrialFunction(Q)
nu_u, nu_v = TestFunctions(Q)

u_exact = Function(Q_u)
v_exact = Function(Q_v)
laplacian_u_exact = Function(Q_u)

f = Function(Q_u)
g = Function(Q_grad_v)
J_uv = TrialFunction(Q)
u, v = split(psi)

grad_v = Function(Q_grad_v)

assigner = FunctionAssigner( Q, [Q_u, Q_v] )
