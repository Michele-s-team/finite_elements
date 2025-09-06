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

rmsh = importlib.import_module('mesh.read.square_no_circle_line')

print(f'Module {__file__} called {rmsh.__file__}', flush=True)

# CHANGE PARAMETERS HERE
c_test = [0.3, 0.76]
r_test = 0.345


# CHANGE PARAMETERS HERE

# function_test_integrals_fenics is a function of two variables, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
def function_test_integrals(x):
    return (np.cos(geo.np.linalg.norm(np.subtract(x, c_test)) - r_test) ** 2.0)
    # return 1


# analytical expression for a  scalar function used to test the ds
class FunctionTestIntegrals(UserExpression):
    def eval(self, values, x):
        values[0] = function_test_integrals(x)

    def value_shape(self):
        return (1,)


# construct function spaces and function_test_integrals_fenics for all sub meshes
Q_test = []
function_test_integrals_fenics = []
for p in range(len(lmsh.sub_meshes)):
    Q_test.append(FunctionSpace(lmsh.sub_meshes[p], 'P', 2))
    function_test_integrals_fenics.append(Function(Q_test[p]))
    function_test_integrals_fenics[p].interpolate(FunctionTestIntegrals(element=Q_test[p].ufl_element()))

integral_exact = [''] * len(lmsh.sub_meshes)
integral_exact[0] = dict([ \
    ('dx', 0)
])

integral_exact[1] = dict([ \
    ('dx', 0), \
    ('ds_l', 0)
])

# exact surface integrals
integral_exact[0]['dx'] = cal.surface_integral_rectangle(function_test_integrals, [0, 0], [rmsh.parameters['L'], rmsh.parameters['h']])
integral_exact[1]['dx'] = cal.curve_integral_line(function_test_integrals, 0.0, rmsh.parameters['L'])

# exact line integrals
# form mesh #0
integral_exact[0]['ds_l'] = cal.curve_integral_line(function_test_integrals, [0, 0], [0, rmsh.parameters["h"]])
integral_exact[0]['ds_r'] = cal.curve_integral_line(function_test_integrals, [rmsh.parameters['L'], 0], [rmsh.parameters['L'], rmsh.parameters["h"]])
integral_exact[0]['ds_t'] = cal.curve_integral_line(function_test_integrals, [0, rmsh.parameters["h"]], [rmsh.parameters['L'], rmsh.parameters["h"]])
integral_exact[0]['ds_b'] = cal.curve_integral_line(function_test_integrals, [0, 0], [rmsh.parameters['L'], 0])

integral_exact[0]['ds_lr'] = integral_exact[0]['ds_l'] + integral_exact[0]['ds_r']
integral_exact[0]['ds_tb'] = integral_exact[0]['ds_t'] + integral_exact[0]['ds_b']

integral_exact[0]['ds'] = integral_exact[0]['ds_lr'] + integral_exact[0]['ds_tb']

# for mesh #1
integral_exact[1]['ds_l'] = function_test_integrals(0)
integral_exact[1]['ds_r'] = function_test_integrals(rmsh.parameters['L'])

integral_exact[1]['ds'] = integral_exact[1]['ds_l'] + integral_exact[1]['ds_r']

test_mesh_integral_errors = dict([])

# 2. check mesh integral in the sub_meshes
print(f'Check integrals on the sub_meshes: ')

# surface integrals
for i in range(len(lmsh.sub_meshes)):
    test_mesh_integral_errors[f'\int_sub_mesh_{i} f dx'] = msh.test_mesh_integral(integral_exact[i]['dx'], function_test_integrals_fenics[i], rmsh.dx_sub_mesh[i], f'\int_sub_mesh_{i} f dx')

# line intergrals
# for mesh #0
test_mesh_integral_errors[f'\int f ds_sub_mesh_{0}_l'] = msh.test_mesh_integral(integral_exact[0]['ds_l'], function_test_integrals_fenics[0], rmsh.ds_sub_mesh[0]['ds_l'], f'\int f ds_sub_mesh_{0}_l')
test_mesh_integral_errors[f'\int f ds_sub_mesh_{0}_r'] = msh.test_mesh_integral(integral_exact[0]['ds_r'], function_test_integrals_fenics[0], rmsh.ds_sub_mesh[0]['ds_r'], f'\int f ds_sub_mesh_{0}_r')
test_mesh_integral_errors[f'\int f ds_sub_mesh_{0}_t'] = msh.test_mesh_integral(integral_exact[0]['ds_t'], function_test_integrals_fenics[0], rmsh.ds_sub_mesh[0]['ds_t'], f'\int f ds_sub_mesh_{0}_t')
test_mesh_integral_errors[f'\int f ds_sub_mesh_{0}_b'] = msh.test_mesh_integral(integral_exact[0]['ds_b'], function_test_integrals_fenics[0], rmsh.ds_sub_mesh[0]['ds_b'], f'\int f ds_sub_mesh_{0}_b')

test_mesh_integral_errors[f'\int f ds_sub_mesh_{0}_lr'] = msh.test_mesh_integral(integral_exact[0]['ds_lr'], function_test_integrals_fenics[0], rmsh.ds_sub_mesh[0]['ds_lr'], f'\int f ds_sub_mesh_{0}_lr')
test_mesh_integral_errors[f'\int f ds_sub_mesh_{0}_tb'] = msh.test_mesh_integral(integral_exact[0]['ds_tb'], function_test_integrals_fenics[0], rmsh.ds_sub_mesh[0]['ds_tb'], f'\int f ds_sub_mesh_{0}_tb')

test_mesh_integral_errors[f'\int f ds_sub_mesh_{0}'] = msh.test_mesh_integral(integral_exact[0]['ds'], function_test_integrals_fenics[0], rmsh.ds_sub_mesh[0]['ds'], f'\int f ds_sub_mesh_{0}')

# for mesh #1
test_mesh_integral_errors[f'\int f ds_sub_mesh_{1}_l'] = msh.test_mesh_integral(integral_exact[1]['ds_l'], function_test_integrals_fenics[1], rmsh.ds_sub_mesh[1]['ds_l'], f'\int f ds_sub_mesh_{1}_l')
test_mesh_integral_errors[f'\int f ds_sub_mesh_{1}_r'] = msh.test_mesh_integral(integral_exact[1]['ds_r'], function_test_integrals_fenics[1], rmsh.ds_sub_mesh[1]['ds_r'], f'\int f ds_sub_mesh_{1}_r')

# print to file the residuals of the tests of the mesh integrals
io.write_parameters_to_csv_file(io.add_trailing_slash(rarg.args.output_directory) + 'test_integral_errors.csv', test_mesh_integral_errors)

print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{io.max_dictionary(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')