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

# exact sutface integrals
integral_exact_dx = []
integral_exact_dx.append(cal.surface_integral_rectangle(function_test_integrals, rmsh.parameters["p"][:2], np.add(rmsh.parameters["p"][:2], [rmsh.parameters["L_in"], rmsh.parameters["h_in"]])))
integral_exact_dx.append(cal.surface_integral_rectangle(function_test_integrals, [0, 0], [rmsh.parameters["L"], rmsh.parameters["h"]]) - integral_exact_dx[0])

integral_exact_ds_l, integral_exact_ds_r, integral_exact_ds_t, integral_exact_ds_b = [], [], [], []

integral_exact_ds_l.append(cal.curve_integral_line(function_test_integrals, rmsh.parameters["p"][:2], np.add(rmsh.parameters["p"][:2], [0, rmsh.parameters["h_in"]])))
integral_exact_ds_l.append(cal.curve_integral_line(function_test_integrals, [0, 0], [0, rmsh.parameters["h"]]))

integral_exact_ds_r.append(cal.curve_integral_line(function_test_integrals, np.add(rmsh.parameters["p"][:2], [rmsh.parameters["L_in"], 0]), np.add(rmsh.parameters["p"][:2], [rmsh.parameters["L_in"], rmsh.parameters["h_in"]])))
integral_exact_ds_r.append(cal.curve_integral_line(function_test_integrals, [rmsh.parameters["L"], 0], [rmsh.parameters["L"], rmsh.parameters["h"]]))

integral_exact_ds_t.append(cal.curve_integral_line(function_test_integrals, np.add(rmsh.parameters["p"][:2], [0, rmsh.parameters["h_in"]]), np.add(rmsh.parameters["p"][:2], [rmsh.parameters["L_in"], rmsh.parameters["h_in"]])))
integral_exact_ds_t.append(cal.curve_integral_line(function_test_integrals, [0, rmsh.parameters["h"]], [rmsh.parameters["L"], rmsh.parameters["h"]]))

integral_exact_ds_b.append(cal.curve_integral_line(function_test_integrals, rmsh.parameters["p"][:2], np.add(rmsh.parameters["p"][:2], [rmsh.parameters["L_in"], 0])))
integral_exact_ds_b.append(cal.curve_integral_line(function_test_integrals, [0, 0], [rmsh.parameters["L"], 0]))

# exact line integrals on boundaries
integral_exact_ds_lr, integral_exact_ds_tb, integral_exact_ds_lrtb = [], [], []
for i in range(len(lmsh.sub_meshes)):
    integral_exact_ds_lr.append(integral_exact_ds_l[i] + integral_exact_ds_r[i])
    integral_exact_ds_tb.append(integral_exact_ds_t[i] + integral_exact_ds_b[i])
    integral_exact_ds_lrtb.append(integral_exact_ds_lr[i] + integral_exact_ds_tb[i])

integral_exact_ds = integral_exact_ds_lrtb[0] + integral_exact_ds_lrtb[1]

test_mesh_integral_errors = []

'''
# 1. check integrals in the parent mesh
print(f'Check integrals on the parent mesh: ')
for i in range(len(lmsh.sub_meshes)):

    print(f'* fraction {i} of the parent mesh:')
    test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_dx[i], function_test_integrals_fenics, rmsh.dx_parent_mesh[i], f'\int_{i} f dx'))

    test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_l[i], function_test_integrals_fenics, rmsh.ds_parent_mesh_l[i], f'\int f ds_{i}_l'))
    test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_r[i], function_test_integrals_fenics, rmsh.ds_parent_mesh_r[i], f'\int f ds_{i}_r'))
    test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_t[i], function_test_integrals_fenics, rmsh.ds_parent_mesh_t[i], f'\int f ds_{i}_t'))
    test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_b[i], function_test_integrals_fenics, rmsh.ds_parent_mesh_b[i], f'\int f ds_{i}_b'))

    test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_lr[i], function_test_integrals_fenics, rmsh.ds_parent_mesh_lr[i], f'\int f ds_lr_{i}'))
    test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_tb[i], function_test_integrals_fenics, rmsh.ds_parent_mesh_tb[i], f'\int f ds_tb_{i}'))

    test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_lrtb[i], function_test_integrals_fenics, rmsh.ds_parent_mesh_lrtb[i], f'\int f ds_{i}'))

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds, function_test_integrals_fenics, rmsh.ds_parent_mesh, f'\int f ds'))
'''

# 2. check mesh integral in the sub_meshes
print(f'Check integrals on the sub_meshes: ')

# for i in range(len(lmsh.sub_meshes)):

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_dx[0], function_test_integrals_fenics, rmsh.dx_sub_mesh[0], f'\int_sub_mesh_{0} f dx'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_dx[1], function_test_integrals_fenics, rmsh.dx_sub_mesh[1], f'\int_sub_mesh_{1} f dx'))

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_l[0], function_test_integrals_fenics, rmsh.ds_sub_mesh[0]['l'], f'\int f ds_sub_mesh_{0}_l'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_r[0], function_test_integrals_fenics, rmsh.ds_sub_mesh[0]['r'], f'\int f ds_sub_mesh_{0}_r'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_t[0], function_test_integrals_fenics, rmsh.ds_sub_mesh[0]['t'], f'\int f ds_sub_mesh_{0}_t'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_b[0], function_test_integrals_fenics, rmsh.ds_sub_mesh[0]['b'], f'\int f ds_sub_mesh_{0}_b'))

# test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_lr[i], function_test_integrals_fenics, rmsh.ds_sub_mesh_lr[i], f'\int f ds_sub_mesh_lr_{i}'))
# test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_tb[i], function_test_integrals_fenics, rmsh.ds_parent_mesh_tb[i], f'\int f ds_sub_mesh_tb_{i}'))
#
# test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_lrtb[i], function_test_integrals_fenics, rmsh.ds_sub_mesh_lrtb[i], f'\int f ds_sub_mesh_lrtb_{i}'))

print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{max(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')
