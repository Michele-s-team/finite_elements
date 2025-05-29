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
    return as_tensor(1.0 / 2.0 * (F(u)[k, i] * F(u)[k, j] - ufl.Identity(len(u))[i, j] ), (i, j))
