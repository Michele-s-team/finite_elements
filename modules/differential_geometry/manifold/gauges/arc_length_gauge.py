'''
this module contains the differential-geometry definitions
for a one-dimensional manifold parameterized with a coordinate x^1, in the generalized arc-length gauge e_1^alpha e_1^alpha = nu^2, where nu is a function which depends on the manifold coordinate, see "Lagrangian approach"

all methods specific to one dimension and to the arc-length gauge are defined here, while methods indepentend on the dimension and on the gauge are defined in geometry.py
'''

from fenics import *
import ufl as ufl

import differential_geometry.manifold.geometry as geo

epsilon = ufl.PermutationSymbol(2)

# definition of scalar, vectorial and tensorial quantities
# latin indexes run on 2d curvilinear coordinates
i, j, k, l = ufl.indices(4)

'''
vector tangent to the curvilinear coordinate x on the manifold 
e(psi) = {e_1}_{Lagrangian approach}

Input values: 
- 'psi': the angle psi_here = psi_{Lagrangian approach}
- 'nu': the function specifying the gauge, defined by e_1^alpha e_1^alpha = nu^2
Return values:
- the vector e(psi)[i, j]
'''


def e(psi, nu):
    return as_tensor([[nu*cos(psi), -nu*sin(psi)]])


'''
normal vector to the manifold
Input values: 
- 'psi': the angle psi_here = psi_{Lagrangian approach}
- 'nu': the function specifying the gauge, defined by e_1^alpha e_1^alpha = nu^2
Return values: 
- the normal vector n[i], a vector with two components
'''


def normal(psi, nu):
    v = as_tensor(-epsilon[i, j] * e(psi, nu)[0, j], (i))
    return as_tensor(v[i] / geo.ufl_norm(v), (i))

'''
gaussian curvature: K = K_{al-izzi2020shear}
Input values: 
- 'psi': the angle psi_here = psi_{Lagrangian approach}
- 'nu': the function specifying the gauge, defined by e_1^alpha e_1^alpha = nu^2
Return values: 
- the Gaussian curvature (in this case it is identially zero because the manifold is one dimensional) 
'''
def K(psi, nu):
    return 0
