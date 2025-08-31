import colorama as col
from fenics import *
import importlib
import numpy as np

import calculus as cal
import differential_geometry.manifold.geometry as geo
import input_output as io
import mesh.load as lmsh
import mesh.test_function as tf
import mesh.utils as msh
import runtime_arguments as rarg

rmsh = importlib.import_module('mesh.read.square_ellipse_circle')

print(f'Module {__file__} called {rmsh.__file__}', flush=True)

integral_exact = [''] * len(lmsh.sub_meshes)

integral_exact[0] = dict([ \
    ('dx', 0), \
    ('ds_circle', 0), \
    ('ds_ellipse', 0), \
    ])

integral_exact[1] = dict([ \
    ('dx', 0), \
    ('ds_l', 0), \
    ('ds_r', 0), \
    ('ds_t', 0), \
    ('ds_b', 0), \
    ('ds_lr', 0), \
    ('ds_tb', 0), \
    ('ds_ellipse', 0), \
    ('ds', 0), \
    ])

# exact surface integrals
integral_exact[0]['dx'] = cal.surface_integral_ellipse(tf.function_test_integrals, rmsh.parameters['a'], rmsh.parameters['b'], rmsh.parameters['c'], 0) \
                          - cal.surface_integral_disk(tf.function_test_integrals, rmsh.parameters['r'], rmsh.parameters['c'])
integral_exact[1]['dx'] = cal.surface_integral_rectangle(tf.function_test_integrals, [0, 0], [rmsh.parameters['L'], rmsh.parameters['h']]) \
                          - cal.surface_integral_ellipse(tf.function_test_integrals, rmsh.parameters['a'], rmsh.parameters['b'], rmsh.parameters['c'], 0)
# exact line integrals
# form mesh #0
integral_exact[0]['ds_circle'] = cal.curve_integral_circle(tf.function_test_integrals, rmsh.parameters['r'], rmsh.parameters['c'][:2])
integral_exact[0]['ds_ellipse'] = cal.curve_integral_ellipse(tf.function_test_integrals, rmsh.parameters['a'], rmsh.parameters['b'], rmsh.parameters['c'][:2], 0)

# for mesh #1
integral_exact[1]['ds_l'] = cal.curve_integral_line(tf.function_test_integrals, [0, 0], [0, rmsh.parameters["h"]])
integral_exact[1]['ds_r'] = cal.curve_integral_line(tf.function_test_integrals, [rmsh.parameters['L'], 0], [rmsh.parameters['L'], rmsh.parameters["h"]])
integral_exact[1]['ds_t'] = cal.curve_integral_line(tf.function_test_integrals, [0, rmsh.parameters['h']], [rmsh.parameters['L'], rmsh.parameters["h"]])
integral_exact[1]['ds_b'] = cal.curve_integral_line(tf.function_test_integrals, [0, 0], [rmsh.parameters['L'], 0])

integral_exact[1]['ds_lr'] = integral_exact[1]['ds_l'] + integral_exact[1]['ds_r']
integral_exact[1]['ds_tb'] = integral_exact[1]['ds_t'] + integral_exact[1]['ds_b']

integral_exact[1]['ds_lrtb'] = integral_exact[1]['ds_lr'] + integral_exact[1]['ds_tb']
integral_exact[1]['ds_ellipse'] = cal.curve_integral_ellipse(tf.function_test_integrals, rmsh.parameters['a'], rmsh.parameters['b'], rmsh.parameters['c'][:2], 0)

integral_exact[1]['ds'] = integral_exact[1]['ds_lrtb'] + integral_exact[1]['ds_ellipse']

test_mesh_integral_errors = dict([])

# 2. check mesh integral in the sub_meshes
print(f'Check integrals on the sub_meshes: ')

# surface integrals
for i in range(len(lmsh.sub_meshes)):
    test_mesh_integral_errors[f'\int_sub_mesh_{i} f dx'] = msh.test_mesh_integral(integral_exact[i]['dx'], tf.function_test_integrals_fenics, rmsh.dx_sub_mesh[i], f'\int_sub_mesh_{i} f dx')

# line intergrals
# for mesh #0
test_mesh_integral_errors[f'\int f ds_sub_mesh_{0}_circle'] = msh.test_mesh_integral(integral_exact[0]['ds_circle'], tf.function_test_integrals_fenics, rmsh.ds_sub_mesh[0]['ds_circle'], f'\int f ds_sub_mesh_{0}_circle')
test_mesh_integral_errors[f'\int f ds_sub_mesh_{0}_ellipse'] = msh.test_mesh_integral(integral_exact[0]['ds_ellipse'], tf.function_test_integrals_fenics, rmsh.ds_sub_mesh[0]['ds_ellipse'], f'\int f ds_sub_mesh_{0}_ellipse')

# for mesh #1
test_mesh_integral_errors[f'\int f ds_sub_mesh_{1}_l'] = msh.test_mesh_integral(integral_exact[1]['ds_l'], tf.function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['ds_l'], f'\int f ds_sub_mesh_{1}_l')
test_mesh_integral_errors[f'\int f ds_sub_mesh_{1}_r'] = msh.test_mesh_integral(integral_exact[1]['ds_r'], tf.function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['ds_r'], f'\int f ds_sub_mesh_{1}_r')
test_mesh_integral_errors[f'\int f ds_sub_mesh_{1}_t'] = msh.test_mesh_integral(integral_exact[1]['ds_t'], tf.function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['ds_t'], f'\int f ds_sub_mesh_{1}_t')
test_mesh_integral_errors[f'\int f ds_sub_mesh_{1}_b'] = msh.test_mesh_integral(integral_exact[1]['ds_b'], tf.function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['ds_b'], f'\int f ds_sub_mesh_{1}_b')

test_mesh_integral_errors[f'\int f ds_sub_mesh_{1}_lr'] = msh.test_mesh_integral(integral_exact[1]['ds_lr'], tf.function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['ds_lr'], f'\int f ds_sub_mesh_{1}_lr')
test_mesh_integral_errors[f'\int f ds_sub_mesh_{1}_tb'] = msh.test_mesh_integral(integral_exact[1]['ds_tb'], tf.function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['ds_tb'], f'\int f ds_sub_mesh_{1}_tb')

test_mesh_integral_errors[f'\int f ds_sub_mesh_{1}_lrtb'] = msh.test_mesh_integral(integral_exact[1]['ds_lrtb'], tf.function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['ds_lrtb'], f'\int f ds_sub_mesh_{1}_lrtb')
test_mesh_integral_errors[f'\int f ds_sub_mesh_{1}_ellipse'] = msh.test_mesh_integral(integral_exact[1]['ds_ellipse'], tf.function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['ds_ellipse'], f'\int f ds_sub_mesh_{1}_ellipse')

test_mesh_integral_errors[f'\int f ds_sub_mesh_{1}'] = msh.test_mesh_integral(integral_exact[1]['ds'], tf.function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['ds'], f'\int f ds_sub_mesh_{1}')

# print to file the residuals of the tests of the mesh integrals
io.write_parameters_to_csv_file(io.add_trailing_slash(rarg.args.output_directory) + 'test_integral_errors.csv', test_mesh_integral_errors)

print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{io.max_dictionary(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')