'''
this module contains the differential-geometry definitions
for a one-dimensional manifold parameterized with a coordinate x^1 which is the arc-length, see "Lagrangian approach"

all methods specific to one dimension and to the arc-length gauge are defined here, while methods indepentend on the dimension and on the gauge are defined in geometry.py
'''

from fenics import *
import ufl as ufl
import numpy as np

import geometry as geo

epsilon = ufl.PermutationSymbol(2)

# definition of scalar, vectorial and tensorial quantities
# latin indexes run on 2d curvilinear coordinates
i, j, k, l = ufl.indices(4)

'''
vector tangent to the curvilinear coordinate x on the manifold 
e(psi) = {e_1}_{Lagrangian approach}

Input values: 
- 'psi': the angle psi_here = psi_{Lagrangian approach}
Return values:
- the vector e(psi)[i, j]
'''


def e(psi):
    return as_tensor([[cos(psi), -sin(psi)]])


'''
normal vector to the manifold
Input values: 
- 'psi': the angle psi_here = psi_{Lagrangian approach}
Return values: 
- the normal vector n[i], a vector with two components
'''


def normal(psi):
    v = as_tensor(-epsilon[i, j] * e(psi)[0, j], (i))
    return as_tensor(v[i] / ufl_norm(v), (i))


# gaussian curvature: K = K_{al-izzi2020shear}
def K(psi):
    return 0
