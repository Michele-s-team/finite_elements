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

# exact integrals over lines
integral_exact_ds = cal.curve_integral_circle(tf.function_test_integrals, rmsh.parameters["r"], [0]*2)

# exact integrals over vertices
integral_exact_dp = []
circle_coordinates = []

for i in range(rmsh.parameters['N']):

    circle_coordinates.append(cal.R(i * delta_theta).dot([rmsh.parameters['r'], 0]))
    integral_exact_dp.append(tf.function_test_integrals(circle_coordinates[i]))


test_mesh_integral_errors = dict([])

test_mesh_integral_errors['\int f dx'] = msh.test_mesh_integral(integral_exact_dx, tf.function_test_integrals_fenics, rmsh.dx, '\int f dx')

test_mesh_integral_errors['\int f ds'] = msh.test_mesh_integral(integral_exact_ds, tf.function_test_integrals_fenics, rmsh.ds, '\int f ds')

for i in range(rmsh.parameters['N']):
    test_mesh_integral_errors[f'\int f dp_{i}'] = msh.test_mesh_integral(integral_exact_dp[i], tf.function_test_integrals_fenics, rmsh.dp[i], f'\int f dp_{i}')

# print to file the residuals of the tests of the mesh integrals
io.write_parameters_to_csv_file(os.path.join(rarg.args.output_directory, 'test_integral_errors.csv'), test_mesh_integral_errors)

print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{io.max_dictionary(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')



# test dS - start
# total length via dS
total_length_dS = assemble(1 * rmsh.dS)

# total length by looping over interior facets directly
integral_exact_dS = 0.0
for facet in facets(rmsh.lmsh.mesh):

    if facet.exterior() == False:
        # the facet under consideration is an internal facet -> consider it for the check

        '''
        facet_vertices contains the coordinates of the endpoints of `facet`:
        facet_vertices = 
        [
            [p_0_x, p_0_y],
            [p_1_x, p_1_y]
        ]
        ''' 
        facet_vertices = []

        for v in vertices(facet):
            # run through the vertices of `facet`

            facet_vertices.append((v.point().array().tolist())[:2])

        print(f'\t facet vertices = {facet_vertices}')

        integral_exact_dS += cal.curve_integral_line(tf.function_test_integrals, facet_vertices[0], facet_vertices[1])
        
print(f'dS integral        = {assemble(1 * rmsh.dS)}')
print(f'direct edge sum    = {integral_exact_dS}')

# test dS - end