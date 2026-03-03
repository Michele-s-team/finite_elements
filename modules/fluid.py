'''
This module contains methods related to fluid mechanics
'''

from fenics import *
import ufl as ufl

alpha, beta = ufl.indices(2)


'''
force per unit length exerted by the fluid on a line element
Input values: 
    - 'sigma': stress tensor of the fluid
    - 'n': vector normal to the line element 
    
Return values:
    - the force per unit length {dF/dl}^alpha (a vector)
'''

def dFdl(sigma, n):
    return as_tensor(- sigma[alpha, beta] * n[beta], (alpha))

'''
stress tensor of a fluid living on a flat manifold with dimension d
Input values: 
    - 'v': the fluid velocity (a d-dimensional vector)
    - 's': the fluid negative pressure (or tension), a scalar
    - 'eta': the fluid viscosity

Return values:  
    - sigma[i][j] = sig \delta_{ij} + eta (\partial_j v_i + \partial_i v_j)
'''
def sigma(v, s, eta):
    return(as_tensor(s * ufl.Identity(len(v))[alpha, beta] + eta * (v[alpha].dx(beta) + v[beta].dx(alpha)),(alpha, beta)))