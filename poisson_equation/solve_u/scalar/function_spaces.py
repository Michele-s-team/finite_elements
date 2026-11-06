from fenics import *

import mesh.load as lmsh
import mesh.utils as msh
import parameters.read.solution as rpam


P_u = FiniteElement('P', msh.element_geometry(lmsh.mesh), rpam.parameters['function_space_degree'])
P_s = FiniteElement('Real', msh.element_geometry(lmsh.mesh), 0)
element = MixedElement([P_u, P_s])
Q = FunctionSpace(lmsh.mesh, element)

Q_u = Q.sub(0).collapse()
Q_s = Q.sub(1).collapse()


# Define functions
psi = Function(Q)
u, s = split(psi)

nu_u, nu_s = TestFunctions(Q)


f = Function(Q_u)
J = TrialFunction(Q)

u_exact = Function(Q_u)
s_exact = Function(Q_s)