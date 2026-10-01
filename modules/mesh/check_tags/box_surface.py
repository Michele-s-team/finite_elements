import colorama as col
from fenics import *
import importlib

import calculus as cal
import input_output as io
import mesh.test_function as tf
import mesh.utils as msh
import runtime_arguments as rarg

rmsh = importlib.import_module('mesh.read.box_surface')

print(f'Module {__file__} called {rmsh.__file__}', flush=True)


r = rmsh.lmsh.parameters['r']
L = rmsh.lmsh.parameters['L']

test_mesh_integral_errors = dict([])

# 1. define exact integrals

# 1.1 volume integrals


integral_exact_dx_surface = cal.volume_integral_ball(tf.function_test_integrals, rmsh.lmsh.parameters['surface_radius'], rmsh.lmsh.parameters['surface_translation_vector'])
integral_exact_dx_box = cal.volume_integral_box(tf.function_test_integrals, L, r=r) - integral_exact_dx_surface

integral_exact_dx = integral_exact_dx_surface + integral_exact_dx_box

# 1.2 surface integrals

# 1.2.1 external surfaces

integral_exact_ds_le = cal.surface_integral_rectangle(lambda x: tf.function_test_integrals([x[0], r[1] + L[1], x[1]]), [r[0], r[2]], [r[0] + L[0], r[2] + L[2]])
integral_exact_ds_ri = cal.surface_integral_rectangle(lambda x: tf.function_test_integrals([x[0], r[1], x[1]]), [r[0], r[2]], [r[0] + L[0], r[2] + L[2]])

integral_exact_ds_to = cal.surface_integral_rectangle(lambda x: tf.function_test_integrals([x[0], x[1], r[2] + L[2]]), [r[0], r[1]], [r[0] + L[0], r[1] + L[1]])
integral_exact_ds_bo = cal.surface_integral_rectangle(lambda x: tf.function_test_integrals([x[0], x[1], r[2]]), [r[0], r[1]], [r[0] + L[0], r[1] + L[1]])

integral_exact_ds_fr = cal.surface_integral_rectangle(lambda x: tf.function_test_integrals([r[0], x[0], x[1]]), [r[1], r[2]], [r[1] + L[1], r[2] + L[2]])
integral_exact_ds_ba = cal.surface_integral_rectangle(lambda x: tf.function_test_integrals([r[0] + L[0], x[0], x[1]]), [r[1], r[2]], [r[1] + L[1], r[2] + L[2]])

integral_exact_ds_leri = integral_exact_ds_le + integral_exact_ds_ri
integral_exact_ds_tobo = integral_exact_ds_to + integral_exact_ds_bo
integral_exact_ds_frba = integral_exact_ds_fr + integral_exact_ds_ba

integral_exact_ds = integral_exact_ds_leri + integral_exact_ds_tobo + integral_exact_ds_frba


# 1.2.2 internal surfaces


# 2. print out the integrals on the surface elements and compare them with the exact values to double check that the elements are tagged correctly

# 2.1 volume integrals

test_mesh_integral_errors['\int_box f dx'] = msh.test_mesh_integral(integral_exact_dx_box, tf.function_test_integrals_fenics, rmsh.dx_box, '\int_ball f dx_box')
test_mesh_integral_errors['\int_surface f dx'] = msh.test_mesh_integral(integral_exact_dx_surface, tf.function_test_integrals_fenics, rmsh.dx_surface, '\int_ball f dx_surface')

test_mesh_integral_errors['\int f dx'] = msh.test_mesh_integral(integral_exact_dx, tf.function_test_integrals_fenics, rmsh.dx, '\int_ball f dx')

# 2.2 surface integrals

test_mesh_integral_errors['\int_le f ds'] = msh.test_mesh_integral(integral_exact_ds_le, tf.function_test_integrals_fenics, rmsh.ds_le, '\int_le f ds')
test_mesh_integral_errors['\int_ri f ds'] = msh.test_mesh_integral(integral_exact_ds_ri, tf.function_test_integrals_fenics, rmsh.ds_ri, '\int_ri f ds')
test_mesh_integral_errors['\int_to f ds'] = msh.test_mesh_integral(integral_exact_ds_to, tf.function_test_integrals_fenics, rmsh.ds_to, '\int_to f ds')
test_mesh_integral_errors['\int_bo f ds'] = msh.test_mesh_integral(integral_exact_ds_bo, tf.function_test_integrals_fenics, rmsh.ds_bo, '\int_bo f ds')
test_mesh_integral_errors['\int_fr f ds'] = msh.test_mesh_integral(integral_exact_ds_fr, tf.function_test_integrals_fenics, rmsh.ds_fr, '\int_fr f ds')
test_mesh_integral_errors['\int_ba f ds'] = msh.test_mesh_integral(integral_exact_ds_ba, tf.function_test_integrals_fenics, rmsh.ds_ba, '\int_ba f ds')

test_mesh_integral_errors['\int_leri f ds'] = msh.test_mesh_integral(integral_exact_ds_leri, tf.function_test_integrals_fenics, rmsh.ds_leri, '\int_leri f ds')
test_mesh_integral_errors['\int_tobo f ds'] = msh.test_mesh_integral(integral_exact_ds_tobo, tf.function_test_integrals_fenics, rmsh.ds_tobo, '\int_tobo f ds')
test_mesh_integral_errors['\int_frba f ds'] = msh.test_mesh_integral(integral_exact_ds_frba, tf.function_test_integrals_fenics, rmsh.ds_frba, '\int_frba f ds')

test_mesh_integral_errors['\int f ds'] = msh.test_mesh_integral(integral_exact_ds, tf.function_test_integrals_fenics, rmsh.ds, '\int f ds')



# test_mesh_integral_errors['\int f ds'] = msh.test_mesh_integral(integral_exact_ds, tf.function_test_integrals_fenics, rmsh.ds, '\int f ds')

# print to file the residuals of the tests of the mesh integrals
io.write_parameters_to_csv_file(io.add_trailing_slash(rarg.args.output_directory) + 'test_integral_errors.csv', test_mesh_integral_errors)

print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{io.max_dictionary(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')