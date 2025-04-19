from fenics import *
from mshr import *
import numpy as np

import boundary_geometry as bgeo
import calculus as cal
import geometry as geo
import mesh as msh

import read_mesh_square_no_circle as rmsh

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

integral_exact_dx = cal.surface_integral_rectangle(function_test_integrals, [0, 0], [rmsh.L, rmsh.h])

integral_exact_ds_l = cal.curve_integral_line(function_test_integrals, [0, 0], [0, rmsh.h])
integral_exact_ds_r = cal.curve_integral_line(function_test_integrals, [rmsh.L, 0], [rmsh.L, rmsh.h])
integral_exact_ds_t = cal.curve_integral_line(function_test_integrals, [0, rmsh.h], [rmsh.L, rmsh.h])
integral_exact_ds_b = cal.curve_integral_line(function_test_integrals, [0, 0], [rmsh.L, 0])

integral_exact_ds_lr = integral_exact_ds_l + integral_exact_ds_r
integral_exact_ds_tb = integral_exact_ds_t + integral_exact_ds_b

integral_exact_ds = integral_exact_ds_lr + integral_exact_ds_tb


msh.test_mesh_integral(integral_exact_dx, function_test_integrals_fenics, rmsh.dx, '\int f dx')

msh.test_mesh_integral(integral_exact_ds_l, function_test_integrals_fenics, rmsh.ds_l, '\int f ds_l')
msh.test_mesh_integral(integral_exact_ds_r, function_test_integrals_fenics, rmsh.ds_r, '\int f ds_r')
msh.test_mesh_integral(integral_exact_ds_t, function_test_integrals_fenics, rmsh.ds_t, '\int f ds_t')
msh.test_mesh_integral(integral_exact_ds_b, function_test_integrals_fenics, rmsh.ds_b, '\int f ds_b')

msh.test_mesh_integral(integral_exact_ds_lr, function_test_integrals_fenics, rmsh.ds_lr, '\int f ds_lr')
msh.test_mesh_integral(integral_exact_ds_tb, function_test_integrals_fenics, rmsh.ds_tb, '\int f ds_tb')

msh.test_mesh_integral(integral_exact_ds, function_test_integrals_fenics, rmsh.ds, '\int f ds')
