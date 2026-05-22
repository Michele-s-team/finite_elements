from fenics import *
import dolfin
import ufl as ufl

import differential_geometry.manifold.geometry as geo

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
    return ((ufl.det(F(u))) ** (-exponent))



'''
time derivative of the coefficient K
Input values:
- 'u': displacement vector field
- 'u_dot': displacement vector field
- 'exponent': the exponent in 'K(u, exponent)'
Return values:
- dKdt
'''

def K_dot(u, u_dot, exponent):
    return (-exponent * ((ufl.det(F(u))) ** (-exponent)) * ( G(u)[i, j] * F_dot(u_dot)[j, i] ))


'''
fictitious  modulus of hydrostatic compression, which depends on the deformation-gradient tensor
Input values:
- 'u': displacement vector field
- 'exponent': a power exponent for the determinant of F
Return values:
- 1/det(F(u))^exponent
'''


def mu(u, exponent):
    return ((ufl.det(F(u))) ** (-exponent))


'''
time derivative of the coefficient mu
Input values:
- 'u': displacement vector field
- 'u_dot': displacement vector field
- 'exponent': the exponent in 'mu(u, exponent)'
Return values:
- dmudt
'''

def mu_dot(u, u_dot, exponent):
    return (-exponent * ((ufl.det(F(u))) ** (-exponent)) * ( G(u)[i, j] * F_dot(u_dot)[j, i] ))



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
    return as_tensor(1.0 / 2.0 * (F_dot(u_dot)[k, i] * F(u)[k, j] + F_dot(u_dot)[k, j] * F(u)[k, i]), (i, j))


'''
time derivative of S
Input values:
- 'u': {u^t}_notes
- 'u_dot': {du^t/dt}_notes
Return values:
- dS_{ij}^t/dt_notes
'''


def S_dot(u, u_dot, K, K_dot, mu, mu_dot):
    I = ufl.Identity(len(u))
    return as_tensor( \
        K_dot * E(u)[k, k] * I[i, j] \
        + K * F(u)[l, k] * F_dot(u_dot)[l, k] * I[i, j] \
        + 2 * mu_dot * (E(u)[i, j] - E(u)[k, k] / len(u) * I[i, j])\
        + 2 * mu * (E_dot(u, u_dot)[i, j] - (F(u)[l, k] * F_dot(u_dot)[l, k]) / len(u) * I[i, j]), \
        (i, j))


'''
tensor {G^t_{ij}}_notes
Input values: 
- 'u': displacement vector field
Return values:
- G[i,j] = {G^t_{ij}}_notes
'''


def G(u):
    return ufl.inv(F(u))


'''
tensor {\varsigma_{ij}}_notes
Input values:
- 'var_sigma': \varsigma_notes (transformed surface tension)
- 'var_v': {\rm v}_notes (transformed velocity)
- 'u': displacement vector field
- 'eta': viscosity
Return values:
- var_sigma_tensor[i, j] = {\varsigma_{ij}}_notes
'''


def var_sigma_tensor(var_sigma, var_v, u, eta):
    I = ufl.Identity(len(u))
    return as_tensor(var_sigma * I[i, j] + eta * (G(u)[k, j] * (var_v[i]).dx(k) + G(u)[k, i] * (var_v[j]).dx(k)), (i, j))


'''
determinant of F
Input values:
- 'u': displacement vector field
Return values:
- det(F_{ij}(u))
'''


def detF(u):
    return ufl.det(F(u))

'''
tensor P in 'Equations of motion for an elastic body
Input values:
- 'u': displacement vector field
- 'K', 'mu': bulk modulus and modulus of hydrostatic compression

Return values:
- P[i, j] = P_{ij}
'''
def P(u, K, mu):
    return as_tensor(F(u)[i, j] * S(u, K, mu)[j, k], (i, k))

'''
tensor C in 'Notes "Kanensky legcture notes"' 

Input values:
- 'u': displacement vector field

Return values:
- C[i, j] = C_{ij}
'''
def C(u):
    return as_tensor(2 * E(u)[i, k] + ufl.Identity(len(u))[i, k], (i, k))


'''
tensor N in 'Notes "Kanensky legcture notes"' 
corresponding to the stress tensor of a neo-Hookean elastic model stable under compresssion

Input values:
- 'u': displacement vector field
- 'K', 'mu': bulk modulus and modulus of hydrostatic compression

Return values:
- N[i, j] = N_{ij}
'''
def N(u, K, mu):
    return as_tensor(mu * (detF(u)**(-2.0 / len(u))) * (-1.0/len(u) * C(u)[k, k] * G(u)[i, j] + F(u)[j, i]) + K/2.0 * (detF(u)**2-1.0) * G(u)[i, j], (j, i))


'''
functional psi in 'Notes "Kanensky legcture notes"', which is related to `N` by \delta \psi = N_{ji} \partial_i \delta u_j
Input values: 
    - 'u': displacement vector field
    - 'K', 'mu': bulk modulus and modulus of hydrostatic compression
Return values; 
    -  psi_{Notes "Kanensky legcture notes"}
'''

def psi(u, K, mu):

    return (1.0/2.0 * (mu * ( detF(u)**(-2.0/len(u))) * C(u)[i, i] - len(u) ) + K * (1.0/2.0 * (detF(u)**2 - 1.0) - log(detF(u))))