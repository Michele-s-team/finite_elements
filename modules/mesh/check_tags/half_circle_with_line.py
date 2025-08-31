import colorama as col
from fenics import *
import importlib
import numpy as np

import calculus as cal
import input_output as io
import mesh.test_function as tf
import mesh.utils as msh
import runtime_arguments as rarg
import runtime_arguments as rarg

rmsh = importlib.import_module('mesh.read.half_circle_with_line')

print(f'Module {__file__} called {rmsh.__file__}', flush=True)


integral_exact_dx = cal.surface_integral_disk_slice(tf.function_test_integrals, rmsh.parameters["r"], np.pi, 2 * np.pi, rmsh.c_r)

integral_exact_dline_12 = cal.curve_integral_line(tf.function_test_integrals, rmsh.c_1, rmsh.c_2)
integral_exact_darc_21 = cal.curve_integral_circle_arc(tf.function_test_integrals, rmsh.parameters["r"], np.pi, 2 * np.pi, rmsh.c_r)

integral_exact_dline_34 = cal.curve_integral_line(tf.function_test_integrals, rmsh.parameters["c_3"][:2], rmsh.parameters["c_4"][:2])

integral_exact_dp1 = tf.function_test_integrals([rmsh.parameters["r"], 0])
integral_exact_dp2 = tf.function_test_integrals([-rmsh.parameters["r"], 0])

test_mesh_integral_errors = dict([])

test_mesh_integral_errors['\int dx f'] = msh.test_mesh_integral(integral_exact_dx, tf.function_test_integrals_fenics, rmsh.dx, '\int dx f')
test_mesh_integral_errors['\int dp f_{p_1}'] = msh.test_mesh_integral(integral_exact_dp1, tf.function_test_integrals_fenics, rmsh.dp_line_in_start, '\int dp f_{p_1}')
test_mesh_integral_errors['\int dp f_{p_2}'] = msh.test_mesh_integral(integral_exact_dp2, tf.function_test_integrals_fenics, rmsh.dp_line_in_end, '\int dp f_{p_2}')
test_mesh_integral_errors['\int dl f_{line_12}'] = msh.test_mesh_integral(integral_exact_dline_12, tf.function_test_integrals_fenics, rmsh.ds_line, '\int dl f_{line_12}')
test_mesh_integral_errors['\int dl f_{arc_21}'] = msh.test_mesh_integral(integral_exact_darc_21, tf.function_test_integrals_fenics, rmsh.ds_arc, '\int dl f_{arc_21}')
test_mesh_integral_errors['\int dl f_{line_34}'] = msh.test_mesh_integral(integral_exact_dline_34, tf.function_test_integrals_fenics, rmsh.ds_line_in, '\int dl f_{line_34}')

# print to file the residuals of the tests of the mesh integrals
io.write_parameters_to_csv_file(io.add_trailing_slash(rarg.args.output_directory) + 'test_integral_errors.csv', test_mesh_integral_errors)

print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{io.max_dictionary(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')
