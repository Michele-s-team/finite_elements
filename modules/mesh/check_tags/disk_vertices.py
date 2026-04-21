import colorama as col
from fenics import *
import importlib

import calculus as cal
import input_output as io
import mesh.test_function as tf
import mesh.utils as msh
import numpy as np
import os
import runtime_arguments as rarg

rmsh = importlib.import_module('mesh.read.disk_vertices')

print(f'Module {__file__} called {rmsh.__file__}', flush=True)

delta_theta = 2 * np.pi / rmsh.parameters['N']


# exact integrals over surface
integral_exact_dx = cal.surface_integral_disk(tf.function_test_integrals, rmsh.parameters["r"], [0]*2)

# exact integral over exterior boundary lines
integral_exact_ds = cal.curve_integral_circle(tf.function_test_integrals, rmsh.parameters["r"], [0]*2)

# exact integral over interior boundary lines
integral_exact_dS = cal.curve_integral_dS(rmsh.lmsh.mesh, tf.function_test_integrals)

# exact integrals over vertices
integral_exact_dp = []
circle_coordinates = []

for i in range(rmsh.parameters['N']):

    circle_coordinates.append(cal.R(i * delta_theta).dot([rmsh.parameters['r'], 0]))
    integral_exact_dp.append(tf.function_test_integrals(circle_coordinates[i]))


test_mesh_integral_errors = dict([])

test_mesh_integral_errors['\int f dx'] = msh.test_mesh_integral(integral_exact_dx, tf.function_test_integrals_fenics, rmsh.dx, '\int f dx')

test_mesh_integral_errors['\int f ds'] = msh.test_mesh_integral(integral_exact_ds, tf.function_test_integrals_fenics, rmsh.ds, '\int f ds')

test_mesh_integral_errors['\int f dS'] = msh.test_mesh_integral(integral_exact_dS, tf.function_test_integrals_fenics, rmsh.dS, '\int f dS')

for i in range(rmsh.parameters['N']):
    test_mesh_integral_errors[f'\int f dp_{i}'] = msh.test_mesh_integral(integral_exact_dp[i], tf.function_test_integrals_fenics, rmsh.dp[i], f'\int f dp_{i}')

# print to file the residuals of the tests of the mesh integrals
io.write_parameters_to_csv_file(os.path.join(rarg.args.output_directory, 'test_integral_errors.csv'), test_mesh_integral_errors)

print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{io.max_dictionary(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')

