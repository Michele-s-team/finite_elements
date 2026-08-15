'''
This module contains methods related to fluid mechanics
'''

from fenics import *
import ufl as ufl

import physics.elasticity as ela

alpha, beta, gamma = ufl.indices(3)


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
viscous part of the stress tensor (i.e. stress tensor without the pressure part) of a fluid living on a flat manifold with dimension d
Input values: 
    - 'v': the fluid velocity (a d-dimensional vector)
    - 'eta': the fluid viscosity

Return values:  
    - sigma_no_pressure[alpha][beta] = eta (\partial_beta v_alpha + \partial_alpha v_beta)
'''
def sigma_no_pressure(v, eta):
    return(as_tensor(eta * (v[alpha].dx(beta) + v[beta].dx(alpha)),(alpha, beta)))


'''
stress tensor of a fluid living on a flat manifold with dimension d
Input values: 
    - 'v': the fluid velocity (a d-dimensional vector)
    - 's': the fluid negative pressure (or tension), a scalar
    - 'eta': the fluid viscosity

Return values:  
    - sigma[alpha][beta] = s \delta_{alpha beta} + eta (\partial_beta v_alpha + \partial_alpha v_beta)
'''
def sigma(v, s, eta):
    return(as_tensor(s * ufl.Identity(len(v))[alpha, beta] + sigma_no_pressure(v, eta)[alpha, beta],(alpha, beta)))




'''
stress tensor of a fluid living on a flat domain which is deformed according to the ALE (arbitrary Lagrangian Eulerian) method. Note that the coordinates given as input of this method are the reference-configuration coordinates
Input values: 
    - 'v': the fluid velocity (a d-dimensional vector) 'pulled back' onto reference-configuration coordinates
    - 's': the fluid negative pressure (or tension), a scalar, 'pulled back' onto reference-configuration coordinates
    - 'u': the ALE deformation field (a d-dimensional vector), which depends on reference-configuration coordinates
    - 'eta': the fluid viscosity

Return values:  
    - varsigma[alpha][beta] = s \delta_{alpha beta} + eta (G(u)_{gamma beta} \partial_gamma v_alpha + G(u)_{gamma alpha} \partial_gamma v_beta)
'''
def sigma_ale(v, s, u, eta):
        return(as_tensor(s * ufl.Identity(len(v))[alpha, beta] + eta * ( ela.G(u)[gamma, beta] * (v[alpha].dx(gamma)) + ela.G(u)[gamma, alpha] * v[beta].dx(gamma) ),(alpha, beta)))


def sigma_ale_no_pressure(v, s, u, eta):
        return(as_tensor(eta * ( ela.G(u)[gamma, beta] * (v[alpha].dx(gamma)) + ela.G(u)[gamma, alpha] * v[beta].dx(gamma) ),(alpha, beta)))