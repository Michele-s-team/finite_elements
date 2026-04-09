'''
here integral_exact[i][j] is a dictionary containing the values of the exact integrals on the j-th submesh of the i-th mesh. If the i-th mesh contains no sub-meshes, then integral_exact[i][0] contains simply the exact integrals of the i-th mesh
'''



import colorama as col
from fenics import *
import importlib
import os

import calculus as cal
import input_output as io
import mesh.load as lmsh


import mesh.test_function as tf
import mesh.utils as msh
import runtime_arguments as rarg

rmsh = importlib.import_module('mesh.read.square_shape_line')
print(f'Module {__file__} called {rmsh.__file__}', flush=True)

integral_exact = [''] * lmsh.parameters['n_meshes']

integral_exact[0] = [
    # exact integrals of sub_mesh 0 of mesh 0
    dict([ \
    ('dx', 0),\
    ('ds', 0)
    ]), 
    # exact integrals of sub_mesh 1 of mesh 0
    dict([ \
    ('dx', 0),\
    ('ds_l', 0),\
    ('ds_r', 0),\
    ('ds_t', 0),\
    ('ds_b', 0),\
    ('ds_shape', 0)
    ])]

integral_exact[1] = dict([ \
    ('dx', 0)
    ])

'''
here the exact integrals are computed. Note that the structure of the nested lists integral_exact does not follow the same structure as the nested list of meshes / sub_meshes. 
For example, the exact integral over the circular boundary of sub_mesh 1 of mesh 0 is integral_exact[0][0]['ds'] (the integral over the only boundary of sub_mesh 0 of mesh 0), because they are the same thing.
'''
#1 exact integrals of mesh 0
#1.1 exact bulk integrals for mesh 0

#1.1.1 exact bulk integrals for sub_mesh 0 of mesh 0 
integral_exact[0][0]['dx'] = cal.surface_integral_polygon(tf.function_test_integrals[0], lmsh.mesh_parameters[0]['shape_coordinates'])

#1.1.2 exact bulk integrals for sub_mesh 1 of mesh 0 
integral_exact[0][1]['dx'] = cal.surface_integral_rectangle(tf.function_test_integrals[0], [0, 0], [lmsh.mesh_parameters[0]['L'], lmsh.mesh_parameters[0]['h']]) - integral_exact[0][0]['dx']


#1.2 exact boundary integrals for mesh 0

#1.2.1 exact boundary integrals for sub mesh 0 of mesh 0 
integral_exact[0][0]['ds'] = cal.curve_integral_polygon(tf.function_test_integrals[0], lmsh.mesh_parameters[0]['shape_coordinates'])

#1.2.2 exact boundary integrals for sub mesh 1 of mesh 0 
integral_exact[0][1]['ds_l'] = cal.curve_integral_line(tf.function_test_integrals[0], [0, 0], [0, lmsh.mesh_parameters[0]['h']])
integral_exact[0][1]['ds_r'] = cal.curve_integral_line(tf.function_test_integrals[0], [lmsh.mesh_parameters[0]['L'], 0], [lmsh.mesh_parameters[0]['L'], lmsh.mesh_parameters[0]['h']])
integral_exact[0][1]['ds_t'] = cal.curve_integral_line(tf.function_test_integrals[0], [0, lmsh.mesh_parameters[0]['h']], [lmsh.mesh_parameters[0]['L'], lmsh.mesh_parameters[0]['h']])
integral_exact[0][1]['ds_b'] = cal.curve_integral_line(tf.function_test_integrals[0], [0, 0], [lmsh.mesh_parameters[0]['L'], 0])


#2 exact integrals of mesh 1
#2.1 exact bulk integrals of mesh 1
integral_exact[1]['dx'] = cal.curve_integral_line(tf.function_test_integrals[1], lmsh.mesh_parameters[1]['x_l'], lmsh.mesh_parameters[1]['x_r'])

#2.2 exact boundary integrals of mesh 1
tf.function_test_integrals_fenics[1].set_allow_extrapolation(True)

integral_exact[1]['ds_l'] = (tf.function_test_integrals_fenics[1])(lmsh.mesh_parameters[1]['x_l'])
integral_exact[1]['ds_r'] = (tf.function_test_integrals_fenics[1])(lmsh.mesh_parameters[1]['x_r'])



test_mesh_integral_errors = dict([])

#1. check integrals on meshes
print(f'Check integrals on meshes: ')

# 1.1 bulk integrals

# 1.1.1 bulk integrals on mesh 0
test_mesh_integral_errors[f'\int_mesh_{0} f dx'] = msh.test_mesh_integral(integral_exact[0][0]['dx'] + integral_exact[0][1]['dx'], tf.function_test_integrals_fenics[0], rmsh.dx_mesh[0], f'\int_mesh_{0} f dx')

# 1.1.2 bulk integrals on mesh 1
test_mesh_integral_errors[f'\int_mesh_{1} f dx'] = msh.test_mesh_integral(integral_exact[1]['dx'], tf.function_test_integrals_fenics[1], rmsh.dx_mesh[1], f'\int_mesh_{1} f dx')

# 1.2 boundary integrals

# 1.2.1. boundary integrals on mesh 0
test_mesh_integral_errors[f'\int_mesh_{0} f ds_l'] = msh.test_mesh_integral(integral_exact[0][1]['ds_l'], tf.function_test_integrals_fenics[0], rmsh.ds_mesh[0]['ds_l'], f'\int_mesh_{0} f ds_l')
test_mesh_integral_errors[f'\int_mesh_{0} f ds_r'] = msh.test_mesh_integral(integral_exact[0][1]['ds_r'], tf.function_test_integrals_fenics[0], rmsh.ds_mesh[0]['ds_r'], f'\int_mesh_{0} f ds_r')
test_mesh_integral_errors[f'\int_mesh_{0} f ds_t'] = msh.test_mesh_integral(integral_exact[0][1]['ds_t'], tf.function_test_integrals_fenics[0], rmsh.ds_mesh[0]['ds_t'], f'\int_mesh_{0} f ds_t')
test_mesh_integral_errors[f'\int_mesh_{0} f ds_b'] = msh.test_mesh_integral(integral_exact[0][1]['ds_b'], tf.function_test_integrals_fenics[0], rmsh.ds_mesh[0]['ds_b'], f'\int_mesh_{0} f ds_b')
test_mesh_integral_errors[f'\int_mesh_{0} f ds_shape'] = msh.test_mesh_integral(integral_exact[0][0]['ds'], tf.function_test_integrals_fenics[0], rmsh.ds_mesh[0]['ds_shape'], f'\int_mesh_{0} f ds_shape')

# 1.2.2 boundary integrals on mesh 1
test_mesh_integral_errors[f'\int_mesh_{1} f ds_l'] = msh.test_mesh_integral(integral_exact[1]['ds_l'], tf.function_test_integrals_fenics[1], rmsh.ds_mesh[1]['ds_l'], f'\int_mesh_{1} f ds_l')
test_mesh_integral_errors[f'\int_mesh_{1} f ds_r'] = msh.test_mesh_integral(integral_exact[1]['ds_r'], tf.function_test_integrals_fenics[1], rmsh.ds_mesh[1]['ds_r'], f'\int_mesh_{1} f ds_r')




# 2. check mesh integral on sub_meshes
print(f'Check integrals on sub_meshes: ')

# 2.1 bulk integrals

# 2.1.1 bulk integrals on sub_meshes of mesh 0
for i in range(lmsh.mesh_parameters[0]['n_sub_meshes']):

    test_mesh_integral_errors[f'\int_sub_mesh_{0}_{i} f dx'] = msh.test_mesh_integral(integral_exact[0][i]['dx'], tf.function_test_integrals_fenics[0], rmsh.dx_sub_mesh[0][i], f'\int_sub_mesh_{0}_{i} f dx')


# 2.2 boundary integrals

# 2.2.1 boundary integrals on sub_meshes of mesh 0

# 2.2.1.1 boundary integrals on sub_mesh 0 of mesh 0
test_mesh_integral_errors[f'\int_mesh_{0}_{0} f ds'] = msh.test_mesh_integral(integral_exact[0][0]['ds'], tf.function_test_integrals_fenics[0], rmsh.ds_sub_mesh[0][0]['ds'], f'\int_mesh_{0}_{0} f ds')

# 2.2.1.2 boundary integrals on sub_mesh 1 of mesh 0
test_mesh_integral_errors[f'\int_mesh_{0}_{1} f ds_l'] = msh.test_mesh_integral(integral_exact[0][1]['ds_l'], tf.function_test_integrals_fenics[0], rmsh.ds_sub_mesh[0][1]['ds_l'], f'\int_mesh_{0}_{1} f ds_l')
test_mesh_integral_errors[f'\int_mesh_{0}_{1} f ds_r'] = msh.test_mesh_integral(integral_exact[0][1]['ds_r'], tf.function_test_integrals_fenics[0], rmsh.ds_sub_mesh[0][1]['ds_r'], f'\int_mesh_{0}_{1} f ds_r')
test_mesh_integral_errors[f'\int_mesh_{0}_{1} f ds_t'] = msh.test_mesh_integral(integral_exact[0][1]['ds_t'], tf.function_test_integrals_fenics[0], rmsh.ds_sub_mesh[0][1]['ds_t'], f'\int_mesh_{0}_{1} f ds_t')
test_mesh_integral_errors[f'\int_mesh_{0}_{1} f ds_b'] = msh.test_mesh_integral(integral_exact[0][1]['ds_b'], tf.function_test_integrals_fenics[0], rmsh.ds_sub_mesh[0][1]['ds_b'], f'\int_mesh_{0}_{1} f ds_b')
test_mesh_integral_errors[f'\int_mesh_{0}_{1} f ds_shape'] = msh.test_mesh_integral(integral_exact[0][0]['ds'], tf.function_test_integrals_fenics[0], rmsh.ds_sub_mesh[0][1]['ds_shape'], f'\int_mesh_{0}_{1} f ds_shape')



# print to file the residuals of the tests of the mesh integrals
io.write_parameters_to_csv_file(os.path.join(rarg.args.output_directory, 'test_integral_errors.csv'), test_mesh_integral_errors)

print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{io.max_dictionary(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')
