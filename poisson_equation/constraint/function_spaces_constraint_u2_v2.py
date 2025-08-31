from fenics import *

import parameters.read.solution as rpam
import mesh.load as lmsh

# define elements and function spaces
P_u = FiniteElement('P', triangle, rpam.parameters['function_space_degree'])
P_v = FiniteElement('P', triangle, rpam.parameters['function_space_degree'])
element = MixedElement([P_u, P_v])
Q = FunctionSpace(lmsh.mesh, element)

Q_u = Q.sub(0).collapse()
Q_v = Q.sub(1).collapse()

# Define functions
psi = Function(Q)
J_psi = TrialFunction(Q)
nu_u, nu_v = TestFunctions(Q)

u_exact = Function(Q_u)
v_exact = Function(Q_v)
laplacian_u_exact = Function(Q_u)

u_0 = Function(Q_u)
v_0 = Function(Q_v)


f = Function(Q_u)
g = Function(Q_u)
J_uv = TrialFunction(Q)
u, v = split(psi)

assigner = FunctionAssigner(Q, [Q_u, Q_v])
