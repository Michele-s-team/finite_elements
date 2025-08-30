from fenics import *
import ufl as ufl

import differential_geometry.manifold.geometry as geo
import differential_geometry.boundary.geometry as bgeo
import load_mesh as lmsh
import mesh.utils as mesh_module

i, j, k, l = ufl.indices(4)


# Nt^i_notes on \partisal \Omega_O
def Nt_circle(omega):
    N3d = as_tensor([bgeo.facet_normal[0], bgeo.facet_normal[1], 0.0])
    return as_tensor(geo.g_c(omega)[i, j] * N3d[k] * geo.e(omega)[j, k], (i))


# N_n_notes on \partial \Omega_O
def Nn_circle(omega):
    N3d = as_tensor([bgeo.facet_normal[0], bgeo.facet_normal[1], 0.0])
    return (N3d[i] * (normal(omega))[i])


# vector used to define the pull-back of the metric, h, on a circle with radius r centered at c ( it is independent of r), see 'notes reall2013general'
def dydtheta(c):
    x = ufl.SpatialCoordinate(lmsh.mesh)
    return as_tensor([-(x[1] - c[1]), x[0] - c[0]])


# square root of the determinant of the pull-back of the metric, h, on a circle with radius r centered at c ( it is independent of r). This pull-back is done by parameterizing the circle, \partial \Omega_O witht the polar angle \theta as a variable
def sqrt_deth_circle(omega, c):
    return (sqrt((dydtheta(c))[i] * (dydtheta(c))[j] * geo.g(omega)[i, j]))


# square root of the determinant of the pull-back of the metric on \partial \Omega_in(out), parametrized with l , given by  x^1 = 0 (L) and x^2 = l, as coordinate for \partial \Omega_in (out)
def sqrt_deth_lr(omega):
    return sqrt(geo.g(omega)[1, 1])


# square root of the determinant of the pull-back of the metric on \partial \Omega_W (top or bottom), parametrized with l , given by  x^1 = l and x^2 = 0 (h), as coordinate for \partial \Omega_W
def sqrt_deth_tb(omega):
    return sqrt(geo.g(omega)[0, 0])


# Nt^i_notes on \partial \Omega_in and out
def Nt_lr(omega):
    x = ufl.SpatialCoordinate(lmsh.mesh)
    L = (mesh_module.extremal_coordinates(lmsh.mesh))[0][1]

    N3d = as_tensor([conditional(lt(x[0], L / 2.0), -1.0, 1.0), 0.0, 0.0])
    return as_tensor(geo.g_c(omega)[i, j] * N3d[k] * geo.e(omega)[j, k], (i))


# N_n_notes on \partial \Omega_in and out
def Nn_lr(omega):
    x = ufl.SpatialCoordinate(lmsh.mesh)
    L = (mesh_module.extremal_coordinates(lmsh.mesh))[0][1]

    N3d = as_tensor([conditional(lt(x[0], L / 2.0), -1.0, 1.0), 0.0, 0.0])
    return (N3d[i] * (normal(omega))[i])


# Nt^i_notes on \partisal \Omega_W
def Nt_tb(omega):
    x = ufl.SpatialCoordinate(lmsh.mesh)
    h = (mesh_module.extremal_coordinates(lmsh.mesh))[1][1]

    N3d = as_tensor([0.0, conditional(lt(x[1], h / 2.0), -1.0, 1.0), 0.0])
    return as_tensor(geo.g_c(omega)[i, j] * N3d[k] * geo.e(omega)[j, k], (i))


# N_n_notes on \partial \Omega_top and bottom
def Nn_tb(omega):
    x = ufl.SpatialCoordinate(lmsh.mesh)
    h = (mesh_module.extremal_coordinates(lmsh.mesh))[1][1]

    N3d = as_tensor([0.0, conditional(lt(x[1], h / 2.0), -1.0, 1.0), 0.0])
    return (N3d[i] * (normal(omega))[i])


# n^i_notes on \partial \Omega_in and out
def n_lr(omega):
    return as_tensor((Nt_lr(omega))[k] / sqrt(geo.g(omega)[i, j] * (Nt_lr(omega))[i] * (Nt_lr(omega))[j]), (k))


def n_tb(omega):
    return as_tensor((Nt_tb(omega))[k] / sqrt(geo.g(omega)[i, j] * (Nt_tb(omega))[i] * (Nt_tb(omega))[j]), (k))


def n_circle(omega):
    return as_tensor((Nt_circle(omega))[k] / sqrt(geo.g(omega)[i, j] * (Nt_circle(omega))[i] * (Nt_circle(omega))[j]), (k))
