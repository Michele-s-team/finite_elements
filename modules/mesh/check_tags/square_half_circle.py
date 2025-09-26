import colorama as col
from fenics import *
import importlib
import numpy as np

import calculus as cal
import differential_geometry.manifold.geometry as geo
import input_output as io
import list as li
import mesh.load as lmsh
import mesh.test_function as tf
import mesh.utils as msh
import runtime_arguments as rarg

rmsh = importlib.import_module('mesh.read.square_half_circle')

print(f'Module {__file__} called {rmsh.__file__}', flush=True)


integral_exact_dx = cal.surface_integral_rectangle(tf.function_test_integrals, [0, 0], [rmsh.parameters["L"], rmsh.parameters["h"]]) - \
    cal.surface_integral_disk_slice(tf.function_test_integrals, rmsh.parameters['r'], np.pi, 2*np.pi, [rmsh.parameters['c_r_x'], rmsh.parameters['h']])

integral_exact_ds_l = cal.curve_integral_line(tf.function_test_integrals, [0, 0], [0, rmsh.parameters["h"]])
integral_exact_ds_r = cal.curve_integral_line(tf.function_test_integrals, [rmsh.parameters['L'], 0], [rmsh.parameters['L'], rmsh.parameters['h']]) 


integral_exact_ds_tl = cal.curve_integral_line(tf.function_test_integrals, [0, rmsh.parameters['h']], [rmsh.parameters['c_r_x'] - rmsh.parameters['r'], rmsh.parameters['h']]) 
integral_exact_ds_tr = cal.curve_integral_line(tf.function_test_integrals,  [rmsh.parameters['c_r_x'] + rmsh.parameters['r'], rmsh.parameters['h']], [rmsh.parameters['L'], rmsh.parameters['h']]) 
integral_exact_ds_half_circle = cal.curve_integral_circle_arc(tf.function_test_integrals, rmsh.parameters['r'], np.pi, 2*np.pi, [rmsh.parameters['c_r_x'], rmsh.parameters['h']]) \

integral_exact_ds_tl_tr = integral_exact_ds_tl + integral_exact_ds_tr
integral_exact_ds_t = integral_exact_ds_tl_tr + integral_exact_ds_half_circle

integral_exact_ds_b = cal.curve_integral_line(tf.function_test_integrals, [0, 0], [rmsh.parameters["L"], 0])
    
    
integral_exact_ds_lr = integral_exact_ds_l + integral_exact_ds_r
integral_exact_ds_tb = integral_exact_ds_t + integral_exact_ds_b

integral_exact_ds = integral_exact_ds_lr + integral_exact_ds_tb

test_mesh_integral_errors = dict([])

test_mesh_integral_errors['\int f dx'] = msh.test_mesh_integral(integral_exact_dx, tf.function_test_integrals_fenics, rmsh.dx, '\int f dx')

test_mesh_integral_errors['\int f ds_l'] = msh.test_mesh_integral(integral_exact_ds_l, tf.function_test_integrals_fenics, rmsh.ds_l, '\int f ds_l')
test_mesh_integral_errors['\int f ds_r'] = msh.test_mesh_integral(integral_exact_ds_r, tf.function_test_integrals_fenics, rmsh.ds_r, '\int f ds_r')

test_mesh_integral_errors['\int f ds_tl'] = msh.test_mesh_integral(integral_exact_ds_tl, tf.function_test_integrals_fenics, rmsh.ds_tl, '\int f ds_tl')
test_mesh_integral_errors['\int f ds_tr'] = msh.test_mesh_integral(integral_exact_ds_tr, tf.function_test_integrals_fenics, rmsh.ds_tr, '\int f ds_tr')
test_mesh_integral_errors['\int f ds_half_circle'] = msh.test_mesh_integral(integral_exact_ds_half_circle, tf.function_test_integrals_fenics, rmsh.ds_half_circle, '\int f ds_half_circle')
test_mesh_integral_errors['\int f ds_t'] = msh.test_mesh_integral(integral_exact_ds_t, tf.function_test_integrals_fenics, rmsh.ds_t, '\int f ds_t')


test_mesh_integral_errors['\int f ds_b'] = msh.test_mesh_integral(integral_exact_ds_b, tf.function_test_integrals_fenics, rmsh.ds_b, '\int f ds_b')

test_mesh_integral_errors['\int f ds_half_circle'] = msh.test_mesh_integral(integral_exact_ds_half_circle, tf.function_test_integrals_fenics, rmsh.ds_half_circle, '\int f ds_half_circle')

test_mesh_integral_errors['\int f ds_lr'] = msh.test_mesh_integral(integral_exact_ds_lr, tf.function_test_integrals_fenics, rmsh.ds_lr, '\int f ds_lr')
test_mesh_integral_errors['\int f ds_tb'] = msh.test_mesh_integral(integral_exact_ds_tb, tf.function_test_integrals_fenics, rmsh.ds_tb, '\int f ds_tb')

test_mesh_integral_errors['\int f ds'] = msh.test_mesh_integral(integral_exact_ds, tf.function_test_integrals_fenics, rmsh.ds, '\int f ds')

# print to file the residuals of the tests of the mesh integrals
io.write_parameters_to_csv_file(io.add_trailing_slash(rarg.args.output_directory) + 'test_integral_errors.csv', test_mesh_integral_errors)

print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{io.max_dictionary(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')
