'''
this module contains the differential-geometry definitions
for a two-dimensional manifold parameterized with a coordinate x^1 = x, x^2 = y in the Monge gauge

all methods specific to one dimension and to the Monge gauge are defined here, while methods independent of the dimension and
of the gauge are defined in geometry.py
'''

from fenics import *
import ufl as ufl

import differential_geometry.manifold.geometry as geo

i, j, k, l = ufl.indices(4)

# three-dimensional vector of the differential manifold, which is equal to \vec{X}_{\Gamma}(x_1, x_2) on page 8 if al-izzi2020shear
def X(z, nu=None):
    x = ufl.SpatialCoordinate(mesh)
    return as_tensor([x[0], x[1], z])


# the vectors tangent to the curvilinear coordinates on the manifold : e(z)[i] = e_i_{al-izzi2020shear}
def e(omega, nu=None):
    return as_tensor([[1, 0, omega[0]], [0, 1, omega[1]]])


# MAKE SURE THAT THIS NORMAL IS DIRECTED OUTWARDS
# normal(z) = \hat{n}_{al-izzi2020shear}
def normal(omega, nu=None):
    return as_tensor(cross(e(omega)[0], e(omega)[1]) / geo.ufl_norm(cross(e(omega)[0], e(omega)[1])))


# MAKE SURE THAT THIS NORMAL IS DIRECTED OUTWARDS


# gaussian curvature: K = K_{al-izzi2020shear}
def K(omega, nu=None):
    return (ufl.det(as_tensor(geo.b(omega)[i, k] * geo.g_c(omega)[k, j], (i, j))))
