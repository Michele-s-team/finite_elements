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
