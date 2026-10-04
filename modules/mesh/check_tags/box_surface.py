import colorama as col
from fenics import *
import importlib
import os
import pandas as pd

import calculus as cal
import input_output as io
import mesh.test_function as tf
import mesh.utils as msh
import runtime_arguments as rarg

rmsh = importlib.import_module('mesh.read.box_surface')

print(f'Module {__file__} called {rmsh.__file__}', flush=True)


r = rmsh.lmsh.parameters['r']
L = rmsh.lmsh.parameters['L']

# load coordinates of mesh vertices, triangles and tetrahedra, which will be needed to compute `integral_exact_*`
vertices = pd.read_csv(os.path.join(rarg.args.input_directory, 'vertices.csv')) 
triangles = pd.read_csv(os.path.join(rarg.args.input_directory, 'triangles.csv')) 
tetrahedra = pd.read_csv(os.path.join(rarg.args.input_directory, 'tetrahedra.csv')) 

# relabel `vertices` according to the `id` columns, so vertex with `id=n` can be called with vertices_by_id.loc[n, ...] (see below)
vertices_by_id = vertices.set_index('id')[[':0', ':1', ':2']]



test_mesh_integral_errors = dict([])

# 1. compute exact integrals

# 1.1 volume integrals

# 1.1.1 volume enclosed by the surface

# select only triangles with tag = surface_surface_id
tetrahedra = tetrahedra[tetrahedra['tag'] == rmsh.lmsh.parameters['surface_volume_id']]

# build a list of triangles definining the surface
surface_volume_tetrahedra = []
for _, row in tetrahedra.iterrows():

    p_1 = [vertices_by_id.loc[row["p_1"], f":{i}"] for i in range(3)]
    p_2 = [vertices_by_id.loc[row["p_2"], f":{i}"] for i in range(3)]
    p_3 = [vertices_by_id.loc[row["p_3"], f":{i}"] for i in range(3)]
    p_4 = [vertices_by_id.loc[row["p_4"], f":{i}"] for i in range(3)]

    surface_volume_tetrahedra.append([p_1, p_2, p_3, p_4])


# feed `surface_volume_tetrahedra` to volume_integral_tetrahedral_volume and compute the exact value of the integral of `tf.function_test_integrals` over `dx_surface`
# change this - start
# integral_exact_dx_surface = cal.volume_integral_tetrahedral_surface(tf.function_test_integrals, surface_volume_tetrahedra)
integral_exact_dx_surface = 1.0
# change this - end

# 1.1.2 volume between surface and box

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

# 1.2.2.1 surface 
# select only triangles with tag = surface_surface_id
triangles = triangles[triangles['tag'] == rmsh.lmsh.parameters['surface_surface_id']]

# build a list of triangles definining the surface
surface_triangles = []
for _, row in triangles.iterrows():

    p_1 = [vertices_by_id.loc[row["p_1"], f":{i}"] for i in range(3)]
    p_2 = [vertices_by_id.loc[row["p_2"], f":{i}"] for i in range(3)]
    p_3 = [vertices_by_id.loc[row["p_3"], f":{i}"] for i in range(3)]

    surface_triangles.append([p_1, p_2, p_3])

# feed `surface_triangles` to surface_integral_triangulated_surface and compute the exact value of the integral of `tf.function_test_integrals` over the dS_surface
integral_exact_dS_surface = cal.surface_integral_triangulated_surface(tf.function_test_integrals, surface_triangles)

# 1.2.2.2  triangles internal to surface

integral_exact_dS_I_surface = 1.0

# 1.2.2.2  triangles internal to the volume between surface and volume

integral_exact_dS_I_box = 1.0


# 2. print out the integrals on the surface elements and compare them with the exact values to double check that the elements are tagged correctly

# 2.1 volume integrals

test_mesh_integral_errors['\int_box f dx'] = msh.test_mesh_integral(integral_exact_dx_box, tf.function_test_integrals_fenics, rmsh.dx_box, '\int_ball f dx_box')
test_mesh_integral_errors['\int_surface f dx'] = msh.test_mesh_integral(integral_exact_dx_surface, tf.function_test_integrals_fenics, rmsh.dx_surface, '\int_ball f dx_surface')

test_mesh_integral_errors['\int f dx'] = msh.test_mesh_integral(integral_exact_dx, tf.function_test_integrals_fenics, rmsh.dx, '\int_ball f dx')

# 2.2 surface integrals

# 2.2.1 external surfaces

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

# 2.2.2 internal surfaces

# 2.2.2.1 surface surface
test_mesh_integral_errors['\int f dS_surface'] = msh.test_mesh_integral(integral_exact_dS_surface, tf.function_test_integrals_fenics, rmsh.dS_surface, '\int f dS_surface')

# 2.2.2.2 triangles internal to surface
test_mesh_integral_errors[f'\int f dS_I_surface'] = msh.test_mesh_integral(integral_exact_dS_I_surface, tf.function_test_integrals_fenics, rmsh.dS_I_surface, f'\int f dS_I_surface')

# 2.2.2.3 triangles in the volume between surface and box
test_mesh_integral_errors[f'\int f dS_I_box'] = msh.test_mesh_integral(integral_exact_dS_I_box, tf.function_test_integrals_fenics, rmsh.dS_I_box, f'\int f dS_I_box')



# test_mesh_integral_errors['\int f ds'] = msh.test_mesh_integral(integral_exact_ds, tf.function_test_integrals_fenics, rmsh.ds, '\int f ds')

# print to file the residuals of the tests of the mesh integrals
io.write_parameters_to_csv_file(io.add_trailing_slash(rarg.args.output_directory) + 'test_integral_errors.csv', test_mesh_integral_errors)

print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{io.max_dictionary(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')