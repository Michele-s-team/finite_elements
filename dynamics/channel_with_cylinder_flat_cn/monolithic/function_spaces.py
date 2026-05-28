import dolfin
from fenics import *

import mesh.load as lmsh

# Define function spaces
P_v_n = VectorElement( 'P', triangle, 2 )
P_sigma_n = FiniteElement('P', triangle, 1)

element = MixedElement( [P_v_n, P_sigma_n] )

Q = FunctionSpace(lmsh.mesh, element)

Q_v_n = Q.sub(0).collapse()
Q_sigma_n = Q.sub(1).collapse()

J_psi = TrialFunction(Q)
psi = Function(Q)

v_n, sigma_n = split(psi)

nu_v_n, nu_sigma_n = TestFunctions( Q )

# Define functions for solutions at previous and current time steps
v_n_1 = Function(Q_v_n)


v_l = Function(Q_v_n)
v_tb_circle = Function(Q_v_n)
sigma_r = Function(Q_sigma_n)
