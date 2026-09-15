import colorama as col
from fenics import *
import importlib

import calculus as cal
import input_output as io
import mesh.load as lmsh
import mesh.test_function as tf
import mesh.utils as msh
import runtime_arguments as rarg

rmsh = importlib.import_module('mesh.read.square_no_circle_line')

print(f'Module {__file__} called {rmsh.__file__}', flush=True)


# `boundary_coordinates` is a list containing the vertices of the full boudnary of the mesh
full_boundary_cooordinates = list(rmsh.parameters['shape_coordinates'])
full_boundary_cooordinates.insert(0, [0, 0])
full_boundary_cooordinates.append([rmsh.parameters['L'], 0])

integral_exact = [''] * len(lmsh.mesh)
integral_exact[0] = dict([ \
    ('dx', 0)
])

integral_exact[1] = dict([ \
    ('dx', 0), \
    ('ds_l', 0)
])

# exact surface integrals
integral_exact[0]['dx'] = cal.surface_integral_polygon(tf.function_test_integrals[0], full_boundary_cooordinates)
integral_exact[1]['dx'] = cal.curve_integral_line(tf.function_test_integrals[1], lmsh.mesh_parameters[1]['x_l'], lmsh.mesh_parameters[1]['x_r'])

# exact line integrals
# form mesh #0
integral_exact[0]['ds_l'] = cal.curve_integral_line(tf.function_test_integrals[0], [0, 0], lmsh.mesh_parameters[0]['shape_coordinates'][0])
integral_exact[0]['ds_r'] = cal.curve_integral_line(tf.function_test_integrals[0], lmsh.mesh_parameters[0]['shape_coordinates'][-1], [lmsh.mesh_parameters[0]['L'], 0])
integral_exact[0]['ds_t'] = cal.curve_integral_polygon(tf.function_test_integrals[0], lmsh.mesh_parameters[0]['shape_coordinates'], 
                                                       open=True)
integral_exact[0]['ds_b'] = cal.curve_integral_line(tf.function_test_integrals[0], [0, 0], [lmsh.mesh_parameters[0]['L'], 0])

integral_exact[0]['ds_lr'] = integral_exact[0]['ds_l'] + integral_exact[0]['ds_r']
integral_exact[0]['ds_tb'] = integral_exact[0]['ds_t'] + integral_exact[0]['ds_b']

integral_exact[0]['ds'] = integral_exact[0]['ds_lr'] + integral_exact[0]['ds_tb']

# for mesh #1
integral_exact[1]['ds_l'] = (tf.function_test_integrals[1])(lmsh.mesh_parameters[1]['x_l'])
integral_exact[1]['ds_r'] = (tf.function_test_integrals[1])(lmsh.mesh_parameters[1]['x_r'])

integral_exact[1]['ds'] = integral_exact[1]['ds_l'] + integral_exact[1]['ds_r']

test_mesh_integral_errors = dict([])

# 2. check mesh integral in the meshes
print(f'Check integrals on meshes: ')

# surface integrals
for i in range(len(lmsh.mesh)):
    test_mesh_integral_errors[f'\int_mesh_{i} f dx'] = msh.test_mesh_integral(integral_exact[i]['dx'], tf.function_test_integrals_fenics[i], rmsh.dx_mesh[i], f'\int_mesh_{i} f dx')

# line intergrals
# for mesh #0
test_mesh_integral_errors[f'\int f ds_mesh_{0}_l'] = msh.test_mesh_integral(integral_exact[0]['ds_l'], tf.function_test_integrals_fenics[0], rmsh.ds_mesh[0]['ds_l'], f'\int f ds_mesh_{0}_l')
test_mesh_integral_errors[f'\int f ds_mesh_{0}_r'] = msh.test_mesh_integral(integral_exact[0]['ds_r'], tf.function_test_integrals_fenics[0], rmsh.ds_mesh[0]['ds_r'], f'\int f ds_mesh_{0}_r')
test_mesh_integral_errors[f'\int f ds_mesh_{0}_t'] = msh.test_mesh_integral(integral_exact[0]['ds_t'], tf.function_test_integrals_fenics[0], rmsh.ds_mesh[0]['ds_t'], f'\int f ds_mesh_{0}_t')
test_mesh_integral_errors[f'\int f ds_mesh_{0}_b'] = msh.test_mesh_integral(integral_exact[0]['ds_b'], tf.function_test_integrals_fenics[0], rmsh.ds_mesh[0]['ds_b'], f'\int f ds_mesh_{0}_b')

test_mesh_integral_errors[f'\int f ds_mesh_{0}_lr'] = msh.test_mesh_integral(integral_exact[0]['ds_lr'], tf.function_test_integrals_fenics[0], rmsh.ds_mesh[0]['ds_lr'], f'\int f ds_mesh_{0}_lr')
test_mesh_integral_errors[f'\int f ds_mesh_{0}_tb'] = msh.test_mesh_integral(integral_exact[0]['ds_tb'], tf.function_test_integrals_fenics[0], rmsh.ds_mesh[0]['ds_tb'], f'\int f ds_mesh_{0}_tb')

test_mesh_integral_errors[f'\int f ds_mesh_{0}'] = msh.test_mesh_integral(integral_exact[0]['ds'], tf.function_test_integrals_fenics[0], rmsh.ds_mesh[0]['ds'], f'\int f ds_mesh_{0}')

# for mesh #1
test_mesh_integral_errors[f'\int f ds_mesh_{1}_l'] = msh.test_mesh_integral(integral_exact[1]['ds_l'], tf.function_test_integrals_fenics[1], rmsh.ds_mesh[1]['ds_l'], f'\int f ds_mesh_{1}_l')
test_mesh_integral_errors[f'\int f ds_mesh_{1}_r'] = msh.test_mesh_integral(integral_exact[1]['ds_r'], tf.function_test_integrals_fenics[1], rmsh.ds_mesh[1]['ds_r'], f'\int f ds_mesh_{1}_r')

# print to file the residuals of the tests of the mesh integrals
io.write_parameters_to_csv_file(io.add_trailing_slash(rarg.args.output_directory) + 'test_integral_errors.csv', test_mesh_integral_errors)

print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{io.max_dictionary(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')