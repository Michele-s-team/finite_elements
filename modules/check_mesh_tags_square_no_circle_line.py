import colorama as col
from fenics import *
import numpy as np

import calculus as cal
import load_mesh as lmsh
import geometry as geo
import mesh as msh

import input_output as io
import read_mesh_square_no_circle_line as rmsh

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
    ('dx', 0)
    ])

integral_exact[1] = dict([ \
    ('dx', 0)
    ])

# exact surface integrals
integral_exact[0]['dx'] = cal.surface_integral_rectangle(function_test_integrals, [0,0], [rmsh.parameters['L'], rmsh.parameters['h']])
# integral_exact[1]['dx'] = cal.curve_integral_line(function_test_integrals, [0, rmsh.parameters['h']], [rmsh.parameters['L'], rmsh.parameters['h']])

'''
test_mesh_integral_errors = []

# 2. check mesh integral in the sub_meshes
print(f'Check integrals on the sub_meshes: ')

# surface integrals
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[0]['dx'], function_test_integrals_fenics, rmsh.dx_sub_mesh[0], f'\int_sub_mesh_{0} f dx'))
# test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact[1]['dx'], function_test_integrals_fenics, rmsh.dx_sub_mesh[1], f'\int_sub_mesh_{1} f dx'))


print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{max(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')
'''