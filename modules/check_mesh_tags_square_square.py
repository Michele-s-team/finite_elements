import colorama as col
from fenics import *
import numpy as np

import calculus as cal
import load_mesh as lmsh
import differential_geometry.manifold.geometry as geo
import mesh.mesh as msh

import input_output as io
import mesh.read.square_square as rmsh

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

integral_exact = [''] * len(lmsh.sub_meshes)

# exact surface integrals
integral_exact[0] = dict([])
integral_exact[1] = dict([])

integral_exact[0]['dx'] = cal.surface_integral_rectangle(function_test_integrals, rmsh.parameters["p"][:2], np.add(rmsh.parameters["p"][:2], [rmsh.parameters["L_in"], rmsh.parameters["h_in"]]))
integral_exact[1]['dx'] = cal.surface_integral_rectangle(function_test_integrals, [0, 0], [rmsh.parameters["L"], rmsh.parameters["h"]]) - integral_exact[0]['dx']

integral_exact[0]['l'] = cal.curve_integral_line(function_test_integrals, rmsh.parameters["p"][:2], np.add(rmsh.parameters["p"][:2], [0, rmsh.parameters["h_in"]]))
integral_exact[1]['out_l'] = cal.curve_integral_line(function_test_integrals, [0, 0], [0, rmsh.parameters["h"]])
integral_exact[1]['in_l'] = integral_exact[0]['l']

integral_exact[0]['r'] = cal.curve_integral_line(function_test_integrals, np.add(rmsh.parameters["p"][:2], [rmsh.parameters["L_in"], 0]), np.add(rmsh.parameters["p"][:2], [rmsh.parameters["L_in"], rmsh.parameters["h_in"]]))
integral_exact[1]['out_r'] = cal.curve_integral_line(function_test_integrals, [rmsh.parameters["L"], 0], [rmsh.parameters["L"], rmsh.parameters["h"]])
integral_exact[1]['in_r'] = integral_exact[0]['r']

integral_exact[0]['t'] = cal.curve_integral_line(function_test_integrals, np.add(rmsh.parameters["p"][:2], [0, rmsh.parameters["h_in"]]), np.add(rmsh.parameters["p"][:2], [rmsh.parameters["L_in"], rmsh.parameters["h_in"]]))
integral_exact[1]['out_t'] = cal.curve_integral_line(function_test_integrals, [0, rmsh.parameters["h"]], [rmsh.parameters["L"], rmsh.parameters["h"]])
integral_exact[1]['in_t'] = integral_exact[0]['t']

integral_exact[0]['b'] = cal.curve_integral_line(function_test_integrals, rmsh.parameters["p"][:2], np.add(rmsh.parameters["p"][:2], [rmsh.parameters["L_in"], 0]))
integral_exact[1]['out_b'] = cal.curve_integral_line(function_test_integrals, [0, 0], [rmsh.parameters["L"], 0])
integral_exact[1]['in_b'] = integral_exact[0]['b']

integral_exact[0]['lr'] = integral_exact[0]['l'] + integral_exact[0]['r']
integral_exact[0]['tb'] = integral_exact[0]['t'] + integral_exact[0]['b']

integral_exact[0]['lrtb'] = integral_exact[0]['lr'] + integral_exact[0]['tb']

integral_exact[1]['in_lr'] = integral_exact[1]['in_l'] + integral_exact[1]['in_r']
integral_exact[1]['in_tb'] = integral_exact[1]['in_t'] + integral_exact[1]['in_b']

integral_exact[1]['in_lrtb'] = integral_exact[1]['in_lr'] + integral_exact[1]['in_tb']

integral_exact[1]['out_lr'] = integral_exact[1]['out_l'] + integral_exact[1]['out_r']
integral_exact[1]['out_tb'] = integral_exact[1]['out_t'] + integral_exact[1]['out_b']

integral_exact[1]['out_lrtb'] = integral_exact[1]['out_lr'] + integral_exact[1]['out_tb']


test_mesh_integral_errors = []

# 2. check mesh integral in the sub_meshes
print(f'Check integrals on the sub_meshes: ')

# for i in range(len(lmsh.sub_meshes)):

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[0]['dx'], function_test_integrals_fenics, rmsh.dx_sub_mesh[0], f'\int_sub_mesh_{0} f dx'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[1]['dx'], function_test_integrals_fenics, rmsh.dx_sub_mesh[1], f'\int_sub_mesh_{1} f dx'))

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[0]['l'], function_test_integrals_fenics, rmsh.ds_sub_mesh[0]['l'], f'\int f ds_sub_mesh_{0}_l'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[1]['out_l'], function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['out_l'], f'\int f ds_sub_mesh_{1}_out_l'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[1]['in_l'], function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['in_l'], f'\int f ds_sub_mesh_{1}_in_l'))

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[0]['r'], function_test_integrals_fenics, rmsh.ds_sub_mesh[0]['r'], f'\int f ds_sub_mesh_{0}_r'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[1]['out_r'], function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['out_r'], f'\int f ds_sub_mesh_{1}_out_r'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[1]['in_r'], function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['in_r'], f'\int f ds_sub_mesh_{1}_in_r'))

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[0]['t'], function_test_integrals_fenics, rmsh.ds_sub_mesh[0]['t'], f'\int f ds_sub_mesh_{0}_t'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[1]['out_t'], function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['out_t'], f'\int f ds_sub_mesh_{1}_out_t'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[1]['in_t'], function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['in_t'], f'\int f ds_sub_mesh_{1}_in_t'))

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[0]['b'], function_test_integrals_fenics, rmsh.ds_sub_mesh[0]['b'], f'\int f ds_sub_mesh_{0}_b'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[1]['out_b'], function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['out_b'], f'\int f ds_sub_mesh_{1}_out_b'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[1]['in_b'], function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['in_b'], f'\int f ds_sub_mesh_{1}_in_b'))

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[0]['lr'], function_test_integrals_fenics, rmsh.ds_sub_mesh[0]['lr'], f'\int f ds_sub_mesh_{0}_lr'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[0]['tb'], function_test_integrals_fenics, rmsh.ds_sub_mesh[0]['tb'], f'\int f ds_sub_mesh_{0}_tb'))

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[0]['lrtb'], function_test_integrals_fenics, rmsh.ds_sub_mesh[0]['lrtb'], f'\int f ds_sub_mesh_{0}_lrtb'))

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[1]['in_lr'], function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['in_lr'], f'\int f ds_sub_mesh_{1}_in_lr'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[1]['in_tb'], function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['in_tb'], f'\int f ds_sub_mesh_{1}_in_tb'))

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[1]['in_lrtb'], function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['in_lrtb'], f'\int f ds_sub_mesh_{1}_in_lrtb'))

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[1]['out_lr'], function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['out_lr'], f'\int f ds_sub_mesh_{1}_out_lr'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[1]['out_tb'], function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['out_tb'], f'\int f ds_sub_mesh_{1}_out_tb'))

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[1]['out_lrtb'], function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['out_lrtb'], f'\int f ds_sub_mesh_{1}_out_lrtb'))

print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{max(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')
