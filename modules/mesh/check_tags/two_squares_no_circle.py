import colorama as col
from fenics import *
import importlib

import calculus as cal
import input_output as io
import mesh.test_function as tf
import mesh.utils as msh
import runtime_arguments as rarg

rmsh = importlib.import_module('mesh.read.two_squares_no_circle')

print(f'Module {__file__} called {rmsh.__file__}', flush=True)

integral_exact_dx_l = cal.surface_integral_rectangle(tf.function_test_integrals, [0, 0], [rmsh.parameters["L_m"], rmsh.parameters["h"]])
integral_exact_dx_r = cal.surface_integral_rectangle(tf.function_test_integrals, [rmsh.parameters["L_m"], 0], [rmsh.parameters["L"], rmsh.parameters["h"]])

integral_exact_dx = cal.surface_integral_rectangle(tf.function_test_integrals, [0, 0], [rmsh.parameters["L"], rmsh.parameters["h"]])

integral_exact_ds_l = cal.curve_integral_line(tf.function_test_integrals, [0, 0], [0, rmsh.parameters["h"]])
integral_exact_ds_r = cal.curve_integral_line(tf.function_test_integrals, [rmsh.parameters["L"], 0], [rmsh.parameters["L"], rmsh.parameters["h"]])
integral_exact_ds_lb = cal.curve_integral_line(tf.function_test_integrals, [0, 0], [rmsh.parameters["L_m"], 0])
integral_exact_ds_rb = cal.curve_integral_line(tf.function_test_integrals, [rmsh.parameters["L_m"], 0], [rmsh.parameters["L"], 0])
integral_exact_ds_mid = cal.curve_integral_line(tf.function_test_integrals, [rmsh.parameters["L_m"], 0], [rmsh.parameters["L_m"], rmsh.parameters["h"]])
integral_exact_ds_lt = cal.curve_integral_line(tf.function_test_integrals, [0, rmsh.parameters["h"]], [rmsh.parameters["L_m"], rmsh.parameters["h"]])
integral_exact_ds_rt = cal.curve_integral_line(tf.function_test_integrals, [rmsh.parameters["L_m"], rmsh.parameters["h"]], [rmsh.parameters["L"], rmsh.parameters["h"]])

integral_exact_ds_b = integral_exact_ds_lb + integral_exact_ds_rb
integral_exact_ds_t = integral_exact_ds_lt + integral_exact_ds_rt

integral_exact_ds = integral_exact_ds_l + integral_exact_ds_r + integral_exact_ds_t + integral_exact_ds_b

test_mesh_integral_errors = dict([])

test_mesh_integral_errors['\int f dx_l'] = msh.test_mesh_integral(integral_exact_dx_l, tf.function_test_integrals_fenics, rmsh.dx_l, '\int f dx_l')
test_mesh_integral_errors['\int f dx_r'] = msh.test_mesh_integral(integral_exact_dx_r, tf.function_test_integrals_fenics, rmsh.dx_r, '\int f dx_r')

test_mesh_integral_errors['\int f dx'] = msh.test_mesh_integral(integral_exact_dx, tf.function_test_integrals_fenics, rmsh.dx, '\int f dx')

test_mesh_integral_errors['\int f ds_l'] = msh.test_mesh_integral(integral_exact_ds_l, tf.function_test_integrals_fenics, rmsh.ds_l, '\int f ds_l')
test_mesh_integral_errors['\int f ds_r'] = msh.test_mesh_integral(integral_exact_ds_r, tf.function_test_integrals_fenics, rmsh.ds_r, '\int f ds_r')
test_mesh_integral_errors['\int f ds_lb'] = msh.test_mesh_integral(integral_exact_ds_lb, tf.function_test_integrals_fenics, rmsh.ds_lb, '\int f ds_lb')
test_mesh_integral_errors['\int f ds_rb'] = msh.test_mesh_integral(integral_exact_ds_rb, tf.function_test_integrals_fenics, rmsh.ds_rb, '\int f ds_rb')
test_mesh_integral_errors['\int f ds_mid'] = msh.test_mesh_integral(integral_exact_ds_mid, tf.function_test_integrals_fenics, rmsh.ds_m, '\int f ds_mid')
test_mesh_integral_errors['\int f ds_lt'] = msh.test_mesh_integral(integral_exact_ds_lt, tf.function_test_integrals_fenics, rmsh.ds_lt, '\int f ds_lt')
test_mesh_integral_errors['\int f ds_rt'] = msh.test_mesh_integral(integral_exact_ds_rt, tf.function_test_integrals_fenics, rmsh.ds_rt, '\int f ds_rt')

test_mesh_integral_errors['\int f ds_t'] = msh.test_mesh_integral(integral_exact_ds_t, tf.function_test_integrals_fenics, rmsh.ds_t, '\int f ds_t')
test_mesh_integral_errors['\int f ds_b'] = msh.test_mesh_integral(integral_exact_ds_b, tf.function_test_integrals_fenics, rmsh.ds_b, '\int f ds_b')

test_mesh_integral_errors['\int f ds'] = msh.test_mesh_integral(integral_exact_ds, tf.function_test_integrals_fenics, rmsh.ds, '\int f ds')

# print to file the residuals of the tests of the mesh integrals
io.write_parameters_to_csv_file(io.add_trailing_slash(rarg.args.output_directory) + 'test_integral_errors.csv', test_mesh_integral_errors)

print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{io.max_dictionary(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')