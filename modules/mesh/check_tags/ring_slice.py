import colorama as col
from fenics import *
import importlib

import calculus as cal
import input_output as io
import mesh.test_function as tf
import mesh.utils as msh
import runtime_arguments as rarg

rmsh = importlib.import_module('mesh.read.ring_slice')

print(f'Module {__file__} called {rmsh.__file__}', flush=True)

integral_exact_dx = cal.surface_integral_ring_slice(tf.function_test_integrals, rmsh.parameters["r"], rmsh.parameters["R"], 0, rmsh.theta, rmsh.parameters["c_r"])

integral_exact_ds_arc_r = cal.curve_integral_circle_arc(tf.function_test_integrals, rmsh.parameters["r"], 0, rmsh.theta, rmsh.parameters["c_r"][:2])
integral_exact_ds_arc_R = cal.curve_integral_circle_arc(tf.function_test_integrals, rmsh.parameters["R"], 0, rmsh.theta, rmsh.parameters["c_R"][:2])

integral_exact_ds_line_t = cal.curve_integral_line(tf.function_test_integrals, rmsh.r_lt, rmsh.r_rt)
integral_exact_ds_line_b = cal.curve_integral_line(tf.function_test_integrals, rmsh.r_lb, rmsh.r_rb)

integral_exact_ds_line_tb = integral_exact_ds_line_t + integral_exact_ds_line_b
integral_exact_ds_arc_rR = integral_exact_ds_arc_r + integral_exact_ds_arc_R
integral_exact_ds = integral_exact_ds_arc_rR + integral_exact_ds_line_tb

test_mesh_integral_errors = dict([])

test_mesh_integral_errors['\int f dx'] = msh.test_mesh_integral(integral_exact_dx, tf.function_test_integrals_fenics, rmsh.dx, '\int f dx')

test_mesh_integral_errors['\int f ds_arc_r'] = msh.test_mesh_integral(integral_exact_ds_arc_r, tf.function_test_integrals_fenics, rmsh.ds_arc_r, '\int f ds_arc_r')
test_mesh_integral_errors['\int f ds_arc_R'] = msh.test_mesh_integral(integral_exact_ds_arc_R, tf.function_test_integrals_fenics, rmsh.ds_arc_R, '\int f ds_arc_R')
test_mesh_integral_errors['\int f ds_arc_rR'] = msh.test_mesh_integral(integral_exact_ds_arc_rR, tf.function_test_integrals_fenics, rmsh.ds_arc_rR, '\int f ds_arc_rR')

test_mesh_integral_errors['\int f ds_line_tb'] = msh.test_mesh_integral(integral_exact_ds_line_tb, tf.function_test_integrals_fenics, rmsh.ds_line_tb, '\int f ds_line_tb')

test_mesh_integral_errors['\int f ds'] = msh.test_mesh_integral(integral_exact_ds, tf.function_test_integrals_fenics, rmsh.ds, '\int f ds')

# print to file the residuals of the tests of the mesh integrals
io.write_parameters_to_csv_file(io.add_trailing_slash(rarg.args.output_directory) + 'test_integral_errors.csv', test_mesh_integral_errors)

print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{io.max_dictionary(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')
