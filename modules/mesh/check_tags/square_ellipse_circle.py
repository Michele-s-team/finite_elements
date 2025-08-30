import colorama as col
from fenics import *
import importlib
import numpy as np

import calculus as cal
import differential_geometry.manifold.geometry as geo
import input_output as io
import list as li
import mesh.load as lmsh
import mesh.utils as msh
rmsh = importlib.import_module('mesh.read.square_ellipse_circle')

print(f'Module {__file__} called {rmsh.__file__}', flush=True)

# CHANGE PARAMETERS HERE
c_test = [0.3, 0.76]
r_test = 0.345
# CHANGE PARAMETERS HERE


# a function space used solely to define function_test_integrals_fenics
Q_test = FunctionSpace(lmsh.mesh, 'P', 2)


# function_test_integrals_fenics is a function of two variables, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
def function_test_integrals(x):
    return (np.cos(geo.my_norm(np.subtract(x, c_test)) - r_test) ** 2.0)
    # return 1


# function_test_integrals_fenics is the same as function_test_integrals, but in fenics format
function_test_integrals_fenics = Function(Q_test)


# analytical expression for a  scalar function used to test the ds
class FunctionTestIntegrals(UserExpression):
    def eval(self, values, x):
        values[0] = function_test_integrals(x)

    def value_shape(self):
        return (1,)


function_test_integrals_fenics.interpolate(FunctionTestIntegrals(element=Q_test.ufl_element()))

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
integral_exact[0]['dx'] = cal.surface_integral_ellipse(function_test_integrals, rmsh.parameters['a'], rmsh.parameters['b'], rmsh.parameters['c'], 0) \
                          - cal.surface_integral_disk(function_test_integrals, rmsh.parameters['r'], rmsh.parameters['c'])
integral_exact[1]['dx'] = cal.surface_integral_rectangle(function_test_integrals, [0, 0], [rmsh.parameters['L'], rmsh.parameters['h']]) \
                          - cal.surface_integral_ellipse(function_test_integrals, rmsh.parameters['a'], rmsh.parameters['b'], rmsh.parameters['c'], 0)
# exact line integrals
# form mesh #0
integral_exact[0]['ds_circle'] = cal.curve_integral_circle(function_test_integrals, rmsh.parameters['r'], rmsh.parameters['c'][:2])
integral_exact[0]['ds_ellipse'] = cal.curve_integral_ellipse(function_test_integrals, rmsh.parameters['a'], rmsh.parameters['b'], rmsh.parameters['c'][:2], 0)

# for mesh #1
integral_exact[1]['ds_l'] = cal.curve_integral_line(function_test_integrals, [0, 0], [0, rmsh.parameters["h"]])
integral_exact[1]['ds_r'] = cal.curve_integral_line(function_test_integrals, [rmsh.parameters['L'], 0], [rmsh.parameters['L'], rmsh.parameters["h"]])
integral_exact[1]['ds_t'] = cal.curve_integral_line(function_test_integrals, [0, rmsh.parameters['h']], [rmsh.parameters['L'], rmsh.parameters["h"]])
integral_exact[1]['ds_b'] = cal.curve_integral_line(function_test_integrals, [0, 0], [rmsh.parameters['L'], 0])

integral_exact[1]['ds_lr'] = integral_exact[1]['ds_l'] + integral_exact[1]['ds_r']
integral_exact[1]['ds_tb'] = integral_exact[1]['ds_t'] + integral_exact[1]['ds_b']

integral_exact[1]['ds_lrtb'] = integral_exact[1]['ds_lr'] + integral_exact[1]['ds_tb']
integral_exact[1]['ds_ellipse'] = cal.curve_integral_ellipse(function_test_integrals, rmsh.parameters['a'], rmsh.parameters['b'], rmsh.parameters['c'][:2], 0)

integral_exact[1]['ds'] = integral_exact[1]['ds_lrtb'] + integral_exact[1]['ds_ellipse']

test_mesh_integral_errors = []

# 2. check mesh integral in the sub_meshes
print(f'Check integrals on the sub_meshes: ')

# surface integrals
for i in range(len(lmsh.sub_meshes)):
    test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[i]['dx'], function_test_integrals_fenics, rmsh.dx_sub_mesh[i], f'\int_sub_mesh_{i} f dx'))

# line intergrals
# for mesh #0
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[0]['ds_circle'], function_test_integrals_fenics, rmsh.ds_sub_mesh[0]['ds_circle'], f'\int f ds_sub_mesh_{0}_circle'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[0]['ds_ellipse'], function_test_integrals_fenics, rmsh.ds_sub_mesh[0]['ds_ellipse'], f'\int f ds_sub_mesh_{0}_ellipse'))

# for mesh #1
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[1]['ds_l'], function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['ds_l'], f'\int f ds_sub_mesh_{1}_l'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[1]['ds_r'], function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['ds_r'], f'\int f ds_sub_mesh_{1}_r'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[1]['ds_t'], function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['ds_t'], f'\int f ds_sub_mesh_{1}_t'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[1]['ds_b'], function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['ds_b'], f'\int f ds_sub_mesh_{1}_b'))

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[1]['ds_lr'], function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['ds_lr'], f'\int f ds_sub_mesh_{1}_lr'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[1]['ds_tb'], function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['ds_tb'], f'\int f ds_sub_mesh_{1}_tb'))

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[1]['ds_lrtb'], function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['ds_lrtb'], f'\int f ds_sub_mesh_{1}_lrtb'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[1]['ds_ellipse'], function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['ds_ellipse'], f'\int f ds_sub_mesh_{1}_ellipse'))

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[1]['ds'], function_test_integrals_fenics, rmsh.ds_sub_mesh[1]['ds'], f'\int f ds_sub_mesh_{1}'))

# print to file the residuals of the tests of the mesh integrals
li.print_to_csv_file(test_mesh_integral_errors, 'check/test_mesh_integrals.csv')

print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{max(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')
