'''
this module checks the mesh tags for a mesh given by a line with a vertex in between, generated with generate_mesh/1d/line/vertex/generate_mesh.py
'''

import colorama as col
from fenics import *
import importlib

import calculus as cal
import input_output as io
import mesh.test_function as tf
import mesh.utils as msh
import runtime_arguments as rarg

rmsh = importlib.import_module('mesh.read.line_vertex')

print(f'Module {__file__} called {rmsh.__file__}', flush=True)


integral_exact_dx = cal.curve_integral_line(tf.function_test_integrals, rmsh.parameters['x_l'], rmsh.parameters['x_r'])

integral_exact_ds_l = tf.function_test_integrals_fenics(rmsh.parameters['x_l'])
integral_exact_ds_r = tf.function_test_integrals_fenics(rmsh.parameters['x_r'])
integral_exact_ds_m = tf.function_test_integrals_fenics(rmsh.parameters['x_m'])
integral_exact_ds_lr = integral_exact_ds_l + integral_exact_ds_r

test_mesh_integral_errors =  dict([])

test_mesh_integral_errors['\int f dx'] = msh.test_mesh_integral(integral_exact_dx, tf.function_test_integrals_fenics, rmsh.dx, '\int f dx')

test_mesh_integral_errors['\int f ds_l'] = msh.test_mesh_integral(integral_exact_ds_l, tf.function_test_integrals_fenics, rmsh.ds_l, '\int f ds_l')
test_mesh_integral_errors['\int f ds_r'] = msh.test_mesh_integral(integral_exact_ds_r, tf.function_test_integrals_fenics, rmsh.ds_r, '\int f ds_r')
test_mesh_integral_errors['\int f ds_m'] = msh.test_mesh_integral(integral_exact_ds_m, tf.function_test_integrals_fenics, rmsh.ds_m, '\int f ds_m')

test_mesh_integral_errors['\int f ds_lr'] = msh.test_mesh_integral(integral_exact_ds_lr, tf.function_test_integrals_fenics, rmsh.ds_lr, '\int f ds_lr')

# print to file the residuals of the tests of the mesh integrals
io.write_parameters_to_csv_file(io.add_trailing_slash(rarg.args.output_directory) + 'test_integral_errors.csv', test_mesh_integral_errors)

print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{io.max_dictionary(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')
