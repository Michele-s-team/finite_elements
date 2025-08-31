import colorama as col
from fenics import *
import importlib

import calculus as cal
import input_output as io
import mesh.test_function as tf
import mesh.utils as msh
import runtime_arguments as rarg

rmsh = importlib.import_module('mesh.read.ball')

print(f'Module {__file__} called {rmsh.__file__}', flush=True)

integral_exact_dx = cal.volume_integral_ball(tf.function_test_integrals, rmsh.parameters["r"], rmsh.parameters["c_r"])
integral_exact_ds = cal.surface_integral_sphere(tf.function_test_integrals, rmsh.parameters["r"], rmsh.parameters["c_r"])

test_mesh_integral_errors = dict([])

# print out the integrals on the surface elements and compare them with the exact values to double check that the elements are tagged correctly
test_mesh_integral_errors['\int_ball f dx'] = msh.test_mesh_integral(integral_exact_dx, tf.function_test_integrals_fenics, rmsh.dx, '\int_ball f dx')
test_mesh_integral_errors['\int_sphere f ds'] = msh.test_mesh_integral(integral_exact_ds, tf.function_test_integrals_fenics, rmsh.ds, '\int_sphere f ds')

# print to file the residuals of the tests of the mesh integrals
io.write_parameters_to_csv_file(io.add_trailing_slash(rarg.args.output_directory) + 'test_integral_errors.csv', test_mesh_integral_errors)

print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{io.max_dictionary(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')