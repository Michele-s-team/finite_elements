import colorama as col
from fenics import *
import importlib
import numpy as np

import calculus as cal
import input_output as io
import mesh.load as lmsh
import mesh.test_function as tf
import mesh.utils as msh
import runtime_arguments as rarg

rmsh = importlib.import_module('mesh.read.square_square')

print(f'Module {__file__} called {rmsh.__file__}', flush=True)


integral_exact = [''] * len(lmsh.sub_meshes)

# exact surface integrals
integral_exact[0] = dict([])
integral_exact[1] = dict([])

integral_exact[0]['dx'] = cal.surface_integral_rectangle(tf.function_test_integrals, rmsh.parameters["p"][:2], np.add(rmsh.parameters["p"][:2], [rmsh.parameters["L_in"], rmsh.parameters["h_in"]]))
integral_exact[1]['dx'] = cal.surface_integral_rectangle(tf.function_test_integrals, [0, 0], [rmsh.parameters["L"], rmsh.parameters["h"]]) - integral_exact[0]['dx']

integral_exact[0]['l'] = cal.curve_integral_line(tf.function_test_integrals, rmsh.parameters["p"][:2], np.add(rmsh.parameters["p"][:2], [0, rmsh.parameters["h_in"]]))
integral_exact[1]['out_l'] = cal.curve_integral_line(tf.function_test_integrals, [0, 0], [0, rmsh.parameters["h"]])
integral_exact[1]['in_l'] = integral_exact[0]['l']

integral_exact[0]['r'] = cal.curve_integral_line(tf.function_test_integrals, np.add(rmsh.parameters["p"][:2], [rmsh.parameters["L_in"], 0]), np.add(rmsh.parameters["p"][:2], [rmsh.parameters["L_in"], rmsh.parameters["h_in"]]))
integral_exact[1]['out_r'] = cal.curve_integral_line(tf.function_test_integrals, [rmsh.parameters["L"], 0], [rmsh.parameters["L"], rmsh.parameters["h"]])
integral_exact[1]['in_r'] = integral_exact[0]['r']

integral_exact[0]['t'] = cal.curve_integral_line(tf.function_test_integrals, np.add(rmsh.parameters["p"][:2], [0, rmsh.parameters["h_in"]]), np.add(rmsh.parameters["p"][:2], [rmsh.parameters["L_in"], rmsh.parameters["h_in"]]))
integral_exact[1]['out_t'] = cal.curve_integral_line(tf.function_test_integrals, [0, rmsh.parameters["h"]], [rmsh.parameters["L"], rmsh.parameters["h"]])
integral_exact[1]['in_t'] = integral_exact[0]['t']

integral_exact[0]['b'] = cal.curve_integral_line(tf.function_test_integrals, rmsh.parameters["p"][:2], np.add(rmsh.parameters["p"][:2], [rmsh.parameters["L_in"], 0]))
integral_exact[1]['out_b'] = cal.curve_integral_line(tf.function_test_integrals, [0, 0], [rmsh.parameters["L"], 0])
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

test_mesh_integral_errors = dict([])

# 2. check mesh integral in the sub_meshes
print(f'Check integrals on the sub_meshes: ')

# for i in range(len(lmsh.sub_meshes)):

test_mesh_integral_errors[f'\int_sub_mesh_{0} f dx'] = msh.test_mesh_integral(integral_exact[0]['dx'], tf.function_test_integrals_fenics, rmsh.dx_sub_mesh[0], f'\int_sub_mesh_{0} f dx')
test_mesh_integral_errors[f'\int_sub_mesh_{1} f dx'] = msh.test_mesh_integral(integral_exact[1]['dx'], tf.function_test_integrals_fenics, rmsh.dx_sub_mesh[1], f'\int_sub_mesh_{1} f dx')

test_mesh_integral_errors[f'\int f ds_sub_mesh_{0}_l'] = msh.test_mesh_integral(integral_exact[0]['l'], tf.function_test_integrals_fenics, rmsh.ds_sub_mesh[0]['l'], f'\int f ds_sub_mesh_{0}_l')
test_mesh_integral_errors[f'\int f ds_sub_mesh_{1}_out_l'] = msh.test_mesh_integral(integral_exact[1]['out_l'], tf.function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['out_l'], f'\int f ds_sub_mesh_{1}_out_l')
test_mesh_integral_errors[f'\int f ds_sub_mesh_{1}_in_l'] = msh.test_mesh_integral(integral_exact[1]['in_l'], tf.function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['in_l'], f'\int f ds_sub_mesh_{1}_in_l')

test_mesh_integral_errors[f'\int f ds_sub_mesh_{0}_r'] = msh.test_mesh_integral(integral_exact[0]['r'], tf.function_test_integrals_fenics, rmsh.ds_sub_mesh[0]['r'], f'\int f ds_sub_mesh_{0}_r')
test_mesh_integral_errors[f'\int f ds_sub_mesh_{1}_out_r'] = msh.test_mesh_integral(integral_exact[1]['out_r'], tf.function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['out_r'], f'\int f ds_sub_mesh_{1}_out_r')
test_mesh_integral_errors[f'\int f ds_sub_mesh_{1}_in_r'] = msh.test_mesh_integral(integral_exact[1]['in_r'], tf.function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['in_r'], f'\int f ds_sub_mesh_{1}_in_r')

test_mesh_integral_errors[f'\int f ds_sub_mesh_{0}_t'] = msh.test_mesh_integral(integral_exact[0]['t'], tf.function_test_integrals_fenics, rmsh.ds_sub_mesh[0]['t'], f'\int f ds_sub_mesh_{0}_t')
test_mesh_integral_errors[f'\int f ds_sub_mesh_{1}_out_t'] = msh.test_mesh_integral(integral_exact[1]['out_t'], tf.function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['out_t'], f'\int f ds_sub_mesh_{1}_out_t')
test_mesh_integral_errors[f'\int f ds_sub_mesh_{1}_in_t'] = msh.test_mesh_integral(integral_exact[1]['in_t'], tf.function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['in_t'], f'\int f ds_sub_mesh_{1}_in_t')

test_mesh_integral_errors[f'\int f ds_sub_mesh_{0}_b'] = msh.test_mesh_integral(integral_exact[0]['b'], tf.function_test_integrals_fenics, rmsh.ds_sub_mesh[0]['b'], f'\int f ds_sub_mesh_{0}_b')
test_mesh_integral_errors[f'\int f ds_sub_mesh_{1}_out_b'] = msh.test_mesh_integral(integral_exact[1]['out_b'], tf.function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['out_b'], f'\int f ds_sub_mesh_{1}_out_b')
test_mesh_integral_errors[f'\int f ds_sub_mesh_{1}_in_b'] = msh.test_mesh_integral(integral_exact[1]['in_b'], tf.function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['in_b'], f'\int f ds_sub_mesh_{1}_in_b')

test_mesh_integral_errors[f'\int f ds_sub_mesh_{0}_lr'] = msh.test_mesh_integral(integral_exact[0]['lr'], tf.function_test_integrals_fenics, rmsh.ds_sub_mesh[0]['lr'], f'\int f ds_sub_mesh_{0}_lr')
test_mesh_integral_errors[f'\int f ds_sub_mesh_{0}_tb'] = msh.test_mesh_integral(integral_exact[0]['tb'], tf.function_test_integrals_fenics, rmsh.ds_sub_mesh[0]['tb'], f'\int f ds_sub_mesh_{0}_tb')

test_mesh_integral_errors[f'\int f ds_sub_mesh_{0}_lrtb'] = msh.test_mesh_integral(integral_exact[0]['lrtb'], tf.function_test_integrals_fenics, rmsh.ds_sub_mesh[0]['lrtb'], f'\int f ds_sub_mesh_{0}_lrtb')

test_mesh_integral_errors[f'\int f ds_sub_mesh_{1}_in_lr'] = msh.test_mesh_integral(integral_exact[1]['in_lr'], tf.function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['in_lr'], f'\int f ds_sub_mesh_{1}_in_lr')
test_mesh_integral_errors[f'\int f ds_sub_mesh_{1}_in_tb'] = msh.test_mesh_integral(integral_exact[1]['in_tb'], tf.function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['in_tb'], f'\int f ds_sub_mesh_{1}_in_tb')

test_mesh_integral_errors[f'\int f ds_sub_mesh_{1}_in_lrtb'] = msh.test_mesh_integral(integral_exact[1]['in_lrtb'], tf.function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['in_lrtb'], f'\int f ds_sub_mesh_{1}_in_lrtb')

test_mesh_integral_errors[f'\int f ds_sub_mesh_{1}_out_lr'] = msh.test_mesh_integral(integral_exact[1]['out_lr'], tf.function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['out_lr'], f'\int f ds_sub_mesh_{1}_out_lr')
test_mesh_integral_errors[f'\int f ds_sub_mesh_{1}_out_tb'] = msh.test_mesh_integral(integral_exact[1]['out_tb'], tf.function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['out_tb'], f'\int f ds_sub_mesh_{1}_out_tb')

test_mesh_integral_errors[f'\int f ds_sub_mesh_{1}_out_lrtb'] = msh.test_mesh_integral(integral_exact[1]['out_lrtb'], tf.function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['out_lrtb'], f'\int f ds_sub_mesh_{1}_out_lrtb')

# print to file the residuals of the tests of the mesh integrals
io.write_parameters_to_csv_file(io.add_trailing_slash(rarg.args.output_directory) + 'test_integral_errors.csv', test_mesh_integral_errors)

print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{io.max_dictionary(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')