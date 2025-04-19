from fenics import *
from mshr import *
import numpy as np
import scipy.integrate as integrate

import boundary_geometry as bgeo
import calculus as cal
import geometry as geo
import mesh as msh

import read_mesh_ring as rmsh
from calculus import surface_integral_ring

# CHANGE PARAMETERS HERE
c_test = [0.3, 0.76]
r_test = 0.345
# CHANGE PARAMETERS HERE

# a function space used solely to define function_test_integrals_fenics
Q_test = FunctionSpace(bgeo.mesh, 'P', 2)

# function_test_integrals_fenics is a function of two variables, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
def function_test_integrals(x):
    return (np.cos(geo.my_norm(np.subtract(x, c_test)) - r_test) ** 2.0)

# function_test_integrals_fenics is the same as function_test_integrals, but in fenics format
function_test_integrals_fenics = Function(Q_test)

# analytical expression for a  scalar function used to test the ds
class FunctionTestIntegrals(UserExpression):
    def eval(self, values, x):
        values[0] = function_test_integrals(x)

    def value_shape(self):
        return (1,)


function_test_integrals_fenics.interpolate(FunctionTestIntegrals(element=Q_test.ufl_element()))

# integral_exact_dx = integrate.dblquad(
#     lambda r, theta: r * test_function(rmsh.c_r[1] + r * np.sin(theta), rmsh.c_r[0] + r * np.cos(theta)), 0, 2 * np.pi,
#     lambda r: rmsh.r, lambda r: rmsh.R)[0]

integral_exact_dx = surface_integral_ring(function_test_integrals, rmsh.r, rmsh.R, rmsh.c_r)

# integral_exact_ds_r = (integrate.quad(
#     lambda theta: rmsh.r * test_function(rmsh.c_r[1] + rmsh.r * np.sin(theta), rmsh.c_r[0] + rmsh.r * np.cos(theta)), 0,
#     2 * np.pi))[0]
# integral_exact_ds_R = (integrate.quad(
#     lambda theta: rmsh.R * test_function(rmsh.c_r[1] + rmsh.R * np.sin(theta), rmsh.c_r[0] + rmsh.R * np.cos(theta)), 0,
#     2 * np.pi))[0]

# integral_exact_ds = integral_exact_ds_r + integral_exact_ds_R

msh.test_mesh_integral(integral_exact_dx, function_test_integrals_fenics, rmsh.dx, '\int f dx')

# msh.test_mesh_integral(integral_exact_ds_r, f_test_ds, rmsh.ds_r, '\int f ds_r')
# msh.test_mesh_integral(integral_exact_ds_R, f_test_ds, rmsh.ds_R, '\int f ds_R')

# msh.test_mesh_integral(integral_exact_ds, f_test_ds, rmsh.ds, '\int f ds')
