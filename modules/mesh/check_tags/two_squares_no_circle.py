import colorama as col
from fenics import *
import importlib
import numpy as np

import calculus as cal
import differential_geometry.manifold.geometry as geo
import mesh.load as lmsh
import input_output as io
import mesh.utils as msh

rmsh = importlib.import_module('mesh.read.two_squares_no_circle')

print(f'Module {__file__} called {rmsh.__file__}', flush=True)

# CHANGE PARAMETERS HERE
c_test = [0.3, 0.76]
r_test = 0.345
# CHANGE PARAMETERS HERE


# a function space used solely to define function_test_integrals_fenics
Q_test = FunctionSpace(lmsh.mesh, 'P', 2)


# function_test_integrals_fenics is a function of two variables, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
def function_test_integrals(x):
    return (np.cos(geo.my_norm(np.subtract(x, c_test)) - r_test) ** 2.0)
    # return 1


# function_test_integrals_fenics is the same as function_test_integrals, but in fenics format
function_test_integrals_fenics = Function(Q_test)


# analytical expression for a  scalar function used to test the ds
class FunctionTestIntegrals(UserExpression):
    def eval(self, values, x):
        values[0] = function_test_integrals(x)

    def value_shape(self):
        return (1,)


function_test_integrals_fenics.interpolate(FunctionTestIntegrals(element=Q_test.ufl_element()))

integral_exact_dx_l = cal.surface_integral_rectangle(function_test_integrals, [0, 0], [rmsh.parameters["L_m"], rmsh.parameters["h"]])
integral_exact_dx_r = cal.surface_integral_rectangle(function_test_integrals, [rmsh.parameters["L_m"], 0], [rmsh.parameters["L"], rmsh.parameters["h"]])

integral_exact_dx = cal.surface_integral_rectangle(function_test_integrals, [0, 0], [rmsh.parameters["L"], rmsh.parameters["h"]])



integral_exact_ds_l = cal.curve_integral_line(function_test_integrals, [0, 0], [0, rmsh.parameters["h"]])
integral_exact_ds_r = cal.curve_integral_line(function_test_integrals, [rmsh.parameters["L"], 0], [rmsh.parameters["L"], rmsh.parameters["h"]])
integral_exact_ds_lb = cal.curve_integral_line(function_test_integrals, [0, 0], [rmsh.parameters["L_m"], 0])
integral_exact_ds_rb = cal.curve_integral_line(function_test_integrals, [rmsh.parameters["L_m"], 0], [rmsh.parameters["L"], 0])
integral_exact_ds_mid = cal.curve_integral_line(function_test_integrals, [rmsh.parameters["L_m"], 0], [rmsh.parameters["L_m"], rmsh.parameters["h"]])
integral_exact_ds_lt = cal.curve_integral_line(function_test_integrals, [0, rmsh.parameters["h"]], [rmsh.parameters["L_m"], rmsh.parameters["h"]])
integral_exact_ds_rt = cal.curve_integral_line(function_test_integrals, [rmsh.parameters["L_m"], rmsh.parameters["h"]], [rmsh.parameters["L"], rmsh.parameters["h"]])

integral_exact_ds_b = integral_exact_ds_lb + integral_exact_ds_rb
integral_exact_ds_t = integral_exact_ds_lt + integral_exact_ds_rt


integral_exact_ds = integral_exact_ds_l + integral_exact_ds_r + integral_exact_ds_t + integral_exact_ds_b

test_mesh_integral_errors = []


test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_dx_l, function_test_integrals_fenics, rmsh.dx_l, '\int f dx_l'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_dx_r, function_test_integrals_fenics, rmsh.dx_r, '\int f dx_r'))

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_dx, function_test_integrals_fenics, rmsh.dx, '\int f dx'))


test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_l, function_test_integrals_fenics, rmsh.ds_l, '\int f ds_l'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_r, function_test_integrals_fenics, rmsh.ds_r, '\int f ds_r'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_lb, function_test_integrals_fenics, rmsh.ds_lb, '\int f ds_lb'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_rb, function_test_integrals_fenics, rmsh.ds_rb, '\int f ds_rb'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_mid, function_test_integrals_fenics, rmsh.ds_m, '\int f ds_mid'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_lt, function_test_integrals_fenics, rmsh.ds_lt, '\int f ds_lt'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_rt, function_test_integrals_fenics, rmsh.ds_rt, '\int f ds_rt'))

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_t, function_test_integrals_fenics, rmsh.ds_t, '\int f ds_t'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_b, function_test_integrals_fenics, rmsh.ds_b, '\int f ds_b'))

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds, function_test_integrals_fenics, rmsh.ds, '\int f ds'))

# print to file the residuals of the tests of the mesh integrals
li.print_to_csv_file(test_mesh_integral_errors, 'check/test_mesh_integrals.csv')

print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{max(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')

