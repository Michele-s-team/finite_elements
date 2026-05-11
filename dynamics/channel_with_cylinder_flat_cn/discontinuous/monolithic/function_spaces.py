'''
The fields are

- v_n[alpha] = {v^n_alpha}_notes
- sigma_n = \sigma^n_notes
'''

from fenics import *

import mesh.load as lmsh

# 1. function spaces
D_v_n = VectorElement('DG', triangle, 2)
D_sigma_n = FiniteElement('DG', triangle, 1)

element = MixedElement([D_v_n, D_sigma_n])

Q = FunctionSpace(lmsh.mesh, element)

Q_v_n = Q.sub(0).collapse()
Q_sigma_n = Q.sub(1).collapse()


# 2. fields
psi = Function(Q)
v_n, sigma_n = split(psi)

v_n_1 = Function(Q_v_n)

v_l = Function(Q_v_n)
v_tb_circle = Function(Q_v_n)

nu_v_n, nu_sigma_n = TestFunctions(Q)

J_psi = TrialFunction(Q)

