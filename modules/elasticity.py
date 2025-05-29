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
