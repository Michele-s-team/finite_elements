import colorama as col
from fenics import *
import numpy as np

import calculus as cal
import load_mesh as lmsh
import geometry as geo
import mesh as msh

import input_output as io
import read_mesh_square_square as rmsh

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

integral_exact_dx_in = cal.surface_integral_rectangle(function_test_integrals, rmsh.parameters["p"][:2], np.add(rmsh.parameters["p"][:2], [rmsh.parameters["L_in"], rmsh.parameters["h_in"]]))
integral_exact_dx_out = cal.surface_integral_rectangle(function_test_integrals, [0, 0], [rmsh.parameters["L"], rmsh.parameters["h"]]) - integral_exact_dx_in

# exact line intergrals on out boundaries
integral_exact_ds_out_l = cal.curve_integral_line(function_test_integrals, [0, 0], [0, rmsh.parameters["h"]])
integral_exact_ds_out_r = cal.curve_integral_line(function_test_integrals, [rmsh.parameters["L"], 0], [rmsh.parameters["L"], rmsh.parameters["h"]])
integral_exact_ds_out_t = cal.curve_integral_line(function_test_integrals, [0, rmsh.parameters["h"]], [rmsh.parameters["L"], rmsh.parameters["h"]])
integral_exact_ds_out_b = cal.curve_integral_line(function_test_integrals, [0, 0], [rmsh.parameters["L"], 0])

# exact line intergrals on in boundaries
integral_exact_ds_in_l = cal.curve_integral_line(function_test_integrals, rmsh.parameters["p"][:2], np.add(rmsh.parameters["p"][:2], [0, rmsh.parameters["h_in"]]))
integral_exact_ds_in_r = cal.curve_integral_line(function_test_integrals, np.add(rmsh.parameters["p"][:2], [rmsh.parameters["L_in"], 0]), np.add(rmsh.parameters["p"][:2], [rmsh.parameters["L_in"], rmsh.parameters["h_in"]]))
integral_exact_ds_in_t = cal.curve_integral_line(function_test_integrals, np.add(rmsh.parameters["p"][:2], [0, rmsh.parameters["h_in"]]), np.add(rmsh.parameters["p"][:2], [rmsh.parameters["L_in"], rmsh.parameters["h_in"]]))
integral_exact_ds_in_b = cal.curve_integral_line(function_test_integrals, rmsh.parameters["p"][:2], np.add(rmsh.parameters["p"][:2], [rmsh.parameters["L_in"], 0]))

integral_exact_ds_out_lr = integral_exact_ds_out_l + integral_exact_ds_out_r
integral_exact_ds_out_tb = integral_exact_ds_out_t + integral_exact_ds_out_b

integral_exact_ds_out = integral_exact_ds_out_lr + integral_exact_ds_out_tb


integral_exact_ds_in_lr = integral_exact_ds_in_l + integral_exact_ds_in_r
integral_exact_ds_in_tb = integral_exact_ds_in_t + integral_exact_ds_in_b

integral_exact_ds_in = integral_exact_ds_in_lr + integral_exact_ds_in_tb

integral_exact_ds = integral_exact_ds_in + integral_exact_ds_out

test_mesh_integral_errors = []

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_dx_out, function_test_integrals_fenics, rmsh.dx_out, '\int_out f dx'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_dx_in, function_test_integrals_fenics, rmsh.dx_in, '\int_in f dx'))

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_out_l, function_test_integrals_fenics, rmsh.ds_out_l, '\int f ds_out_l'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_out_r, function_test_integrals_fenics, rmsh.ds_out_r, '\int f ds_out_r'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_out_t, function_test_integrals_fenics, rmsh.ds_out_t, '\int f ds_out_t'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_out_b, function_test_integrals_fenics, rmsh.ds_out_b, '\int f ds_out_b'))

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_out_lr, function_test_integrals_fenics, rmsh.ds_out_lr, '\int f ds_out_lr'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_out_tb, function_test_integrals_fenics, rmsh.ds_out_tb, '\int f ds_out_tb'))

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_out, function_test_integrals_fenics, rmsh.ds_out, '\int f ds_out'))


test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_in_l, function_test_integrals_fenics, rmsh.ds_in_l, '\int f ds_in_l'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_in_r, function_test_integrals_fenics, rmsh.ds_in_r, '\int f ds_in_r'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_in_t, function_test_integrals_fenics, rmsh.ds_in_t, '\int f ds_in_t'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_in_b, function_test_integrals_fenics, rmsh.ds_in_b, '\int f ds_in_b'))

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_in_lr, function_test_integrals_fenics, rmsh.ds_in_lr, '\int f ds_in_lr'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_in_tb, function_test_integrals_fenics, rmsh.ds_in_tb, '\int f ds_in_tb'))

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_in, function_test_integrals_fenics, rmsh.ds_in, '\int f ds_in'))

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds, function_test_integrals_fenics, rmsh.ds, '\int f ds'))

#
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_dx_out, function_test_integrals_fenics, rmsh.dx_submesh_out, '\int_submesh_out f dx'))

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_out_l, function_test_integrals_fenics, rmsh.ds_submesh_out_l, '\int f ds_submesh_out_l'))


test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_in_l, function_test_integrals_fenics, rmsh.ds_submesh_in_l, '\int f ds_submesh_in_l'))



print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{max(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')
