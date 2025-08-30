from fenics import *
import ufl as ufl

import differential_geometry.manifold.geometry as geo
import load_mesh as lmsh
import mesh as mesh_module

i, j, k, l, alpha = ufl.indices(4)


# square root of the determinant of the pull-back of the metric on \partial \Omega_in(out), parametrized with l , given by  x^1 = 0 (L) and x^2 = l, as coordinate for \partial \Omega_in (out)
def sqrt_deth_lr(psi):
    return sqrt(1)


# Nt^i_notes on \partial \Omega_in and out
def Nt_lr(psi):
    # x = ufl.SpatialCoordinate(lmsh.mesh)
    # L = (mesh_module.extremal_coordinates(lmsh.mesh))[1]

    # vector in the surrounding two-dimensional Euclidean space
    N2d = as_tensor(geo.e(psi)[0, alpha], (alpha))

    return as_tensor(geo.g_c(psi)[i, j] * N2d[alpha] * geo.e(psi)[j, alpha], (i))


# n^i_notes on \partial \Omega_in and out
def n_lr(omega):
    return as_tensor((Nt_lr(omega))[k] / sqrt(geo.g(omega)[i, j] * (Nt_lr(omega))[i] * (Nt_lr(omega))[j]), (k))
