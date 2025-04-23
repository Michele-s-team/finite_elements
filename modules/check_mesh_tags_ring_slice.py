from fenics import *
from mshr import *
import numpy as np

import boundary_geometry as bgeo
import calculus as cal
import geometry as geo
import mesh as msh

# the module read_mesh_square which is being called will be in the local folder, e.g., in steady-state-no-flow
import read_mesh_ring_slice as rmsh

print(f'Module {__file__} called {rmsh.__file__}', flush=True)

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

integral_exact_dx = cal.surface_integral_ring_slice(function_test_integrals, rmsh.r, rmsh.R, rmsh.theta_min, rmsh.theta_max, rmsh.c_r)

integral_exact_ds_arc_r = cal.curve_integral_circle_arc(function_test_integrals, rmsh.r, rmsh.theta_min, rmsh.theta_max, rmsh.c_r)
integral_exact_ds_arc_R = cal.curve_integral_circle_arc(function_test_integrals, rmsh.R, rmsh.theta_min, rmsh.theta_max, rmsh.c_R)

integral_exact_ds_t = cal.curve_integral_line(function_test_integrals_fenics, rmsh.r_lt, rmsh.r_rt)
integral_exact_ds_b = cal.curve_integral_line(function_test_integrals_fenics, rmsh.r_lb, rmsh.r_rb)
integral_exact_ds_line_tb = integral_exact_ds_t + integral_exact_ds_b

integral_exact_ds_arc_rR = integral_exact_ds_arc_r + integral_exact_ds_arc_R



msh.test_mesh_integral(integral_exact_dx, function_test_integrals_fenics, rmsh.dx, '\int f dx')

msh.test_mesh_integral(integral_exact_ds_arc_r, function_test_integrals_fenics, rmsh.ds_arc_r, '\int f ds_arc_r')
msh.test_mesh_integral(integral_exact_ds_arc_R, function_test_integrals_fenics, rmsh.ds_arc_R, '\int f ds_arc_R')
msh.test_mesh_integral(integral_exact_ds_arc_rR, function_test_integrals_fenics, rmsh.ds_arc_rR, '\int f ds_arc_rR')

msh.test_mesh_integral(integral_exact_ds_line_tb, function_test_integrals_fenics, rmsh.ds_line_tb, '\int f ds_line_tb')

