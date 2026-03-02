from fenics import *

import mesh.utils as msh
import parameters.read.solution as rpam
import mesh.load as lmsh

# define elements and function spaces
P_u = FiniteElement('P', msh.element_geometry(lmsh.mesh), rpam.parameters['function_space_degree'])
P_v = VectorElement('P', msh.element_geometry(lmsh.mesh), rpam.parameters['function_space_degree'])
element = MixedElement([P_u, P_v])
Q = FunctionSpace(lmsh.mesh, element)

Q_u = Q.sub(0).collapse()
Q_v = Q.sub(1).collapse()

# Define functions
psi = Function(Q)
nu_u, nu_v = TestFunctions(Q)

u_exact = Function(Q_u)
hess_u_u_exact = Function(Q_u)
v_exact = Function(Q_v)
laplacian_u_exact = Function(Q_u)

f = Function(Q_u)
J_uv = TrialFunction(Q)
u, v = split(psi)
