from fenics import *
import dolfin
import ufl as ufl

import geometry as geo

i, j, k, l = ufl.indices(4)

'''
Deformation-gradient tensor
Input values:
- 'u': displacement vector field
Return values:
- F[i][j] = F_{ij}_{Notes fluid-structure interaction}
'''


def F(u):
    return as_tensor(ufl.Identity(len(u))[i, j] + u[i].dx(j), (i, j))


'''
Green–Lagrange strain tensor
Input values:
- 'u': displacement vector field
Return values:
- E[i][j] = E_{ij}_{Notes fluid-structure interaction}
'''


def E(u):
    return as_tensor(1.0 / 2.0 * (F(u)[k, i] * F(u)[k, j] - ufl.Identity(len(u))[i, j]), (i, j))


'''
second Piola-Kirkhoff stress tensor
Input values:
- 'u': displacement vector field
- 'K', 'mu': bulk modulus and modulus of hydrostatic compression
Return values:
- S[i][j] = S_{ij}_{Notes fluid-structure interaction}
'''


def S(u, K, mu):
    I = ufl.Identity(len(u))
    return as_tensor(K * E(u)[k, k] * I[i, j] + 2 * mu * (E(u)[i, j] - E(u)[k, k] / len(u) * I[i, j]), (i, j))


'''
fictitious bulk modulus which depends on the deformation-gradient tensor
Input values:
- 'u': displacement vector field
- 'exponent': a power exponent for the determinant of F
Return values:
- 1/det(F(u))^exponent
'''


def K(u, exponent):
    # return 1 / ((ufl.det(F(u))) ** exponent)
    return 1

'''
fictitious  modulus of hydrostatic compression, which depends on the deformation-gradient tensor
Input values:
- 'u': displacement vector field
- 'exponent': a power exponent for the determinant of F
Return values:
- 1/det(F(u))^exponent
'''


def mu(u, exponent):
    # return 1 / ((ufl.det(F(u))) ** exponent)
    return 1

'''
time derivative of F
Input values:
- 'u_dot': {du^t/dt}_notes
Return values:
- dF_{ij}^t/dt_notes
'''
def F_dot(u_dot):
    return as_tensor(u_dot[i].dx(j), (i, j))


'''
time derivative of E
Input values:
- 'u': {u^t}_notes
- 'u_dot': {du^t/dt}_notes
Return values:
- dE_{ij}^t/dt_notes
'''
def E_dot(u, u_dot):
    return as_tensor(1.0/2.0 * (F_dot(u_dot)[k, i] * F(u)[k, j] + F_dot(u_dot)[k, j] * F(u)[k, i]), (i, j))