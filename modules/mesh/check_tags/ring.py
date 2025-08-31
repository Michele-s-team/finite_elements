import colorama as col
from fenics import *
import importlib

import calculus as cal
import input_output as io
import mesh.test_function as tf
import mesh.utils as msh
import runtime_arguments as rarg
import runtime_arguments as rarg

rmsh = importlib.import_module('mesh.read.ring')

print(f'Module {__file__} called {rmsh.__file__}', flush=True)



integral_exact_dx = cal.surface_integral_ring(tf.function_test_integrals, rmsh.parameters["r"], rmsh.parameters["R"], rmsh.parameters["c_r"])

integral_exact_ds_r = cal.curve_integral_circle(tf.function_test_integrals, rmsh.parameters["r"], rmsh.parameters["c_r"][:2])
integral_exact_ds_R = cal.curve_integral_circle(tf.function_test_integrals, rmsh.parameters["R"], rmsh.parameters["c_R"][:2])

integral_exact_ds = integral_exact_ds_r + integral_exact_ds_R

test_mesh_integral_errors = dict([])

test_mesh_integral_errors['\int f dx'] = msh.test_mesh_integral(integral_exact_dx, tf.function_test_integrals_fenics, rmsh.dx, '\int f dx')

test_mesh_integral_errors['\int f ds_r'] = msh.test_mesh_integral(integral_exact_ds_r, tf.function_test_integrals_fenics, rmsh.ds_r, '\int f ds_r')
test_mesh_integral_errors['\int f ds_R'] = msh.test_mesh_integral(integral_exact_ds_R, tf.function_test_integrals_fenics, rmsh.ds_R, '\int f ds_R')

test_mesh_integral_errors['\int f ds'] = msh.test_mesh_integral(integral_exact_ds, tf.function_test_integrals_fenics, rmsh.ds, '\int f ds')

# print to file the residuals of the tests of the mesh integrals
io.write_parameters_to_csv_file(io.add_trailing_slash(rarg.args.output_directory) + 'test_integral_errors.csv', test_mesh_integral_errors)

print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{io.max_dictionary(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')
