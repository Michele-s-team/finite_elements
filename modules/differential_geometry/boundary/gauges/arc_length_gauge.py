from fenics import *
import ufl as ufl

import differential_geometry.manifold.geometry as geo
import load_mesh as lmsh

i, j, k, l, alpha = ufl.indices(4)


# square root of the determinant of the pull-back of the metric on \partial \Omega_in(out), parametrized with l , given by  x^1 = 0 (L) and x^2 = l, as coordinate for \partial \Omega_in (out)
def sqrt_deth_lr(psi):
    return sqrt(1)


'''
vector in the tangent bundle of Omega which normal to \partial Omega and points outside \Omega
Input values:
- 'psi': the angle psi_here = psi_{Lagrangian approach}
Return values: 
- Nt^i_{'Eq. (A9) for a one-dimensional manifold' in notes deserno2004notes}  on partial Omega
'''
def Nt_lr(psi):

    # load the coordinate 'x' on the mesh,
    x = ufl.SpatialCoordinate(lmsh.mesh)
    # define the middle point of the coordinate 'x'
    x_middle = (lmsh.parameters['x_l'] + lmsh.parameters['x_r']) / 2.0

    '''
    vector in the surrounding two-dimensional Euclidean space
    if x[0] lies in the left (right) half of the interval x_l, x_r, N_2d = e[0] (-e[0]), 
    '''
    N2d = as_tensor(conditional(lt(x[0], x_middle), -1.0, 1.0) * geo.e(psi)[0, alpha], (alpha))

    # return the projection of N_2d on the tangent bundle of the manifold
    return as_tensor(geo.g_c(psi)[i, j] * N2d[alpha] * geo.e(psi)[j, alpha], (i))


# n^i_notes on \partial \Omega_in and out
def n_lr(omega):
    return as_tensor((Nt_lr(omega))[k] / sqrt(geo.g(omega)[i, j] * (Nt_lr(omega))[i] * (Nt_lr(omega))[j]), (k))
