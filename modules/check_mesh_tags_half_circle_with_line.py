import colorama as col
import dolfin
from fenics import *
import numpy as np

import calculus as cal
import geometry as geo
import load_2d_mesh as lmsh
import mesh as msh

# the module read_mesh_square which is being called will be in the local folder, e.g., in steady-state-no-flow
import input_output as io
import read_mesh_half_circle_with_line as rmsh

print(f'Module {__file__} called {rmsh.__file__}', flush=True)

c_test = [0.3, 0.76]
r_test = 0.345

# analytical expression for a  scalar function used to test the ds
class FunctionTestIntegralExpression(UserExpression):
    def eval(self, values, x):
        # values[0] = 1
        values[0] = function_test_integral_expression(x)
        # values[0] = x[0]

    def value_shape(self):
        return (1,)


def function_test_integral_expression(x):
    return np.cos(geo.my_norm(np.subtract(x, c_test)) - r_test) ** 2.0


# curve relative to arc_21: it returns [[x[0](t), x[1](t)] , [x[0]'(t), x[1]'(t)]]
def curve_arc_21(t):
    return [[np.cos(np.pi * t), -np.sin(np.pi * t)], [-  np.pi * np.sin(np.pi * t), -np.pi * np.cos(np.pi * t)]]


# curve relative to line_12: it returns [[x[0](t), x[1](t)] , [x[0]'(t), x[1]'(t)]]
def curve_line_12(t):
    return cal.line([r, 0], [-r, 0], t)


Q = FunctionSpace(mesh, 'P', 1)

# f_test_ds is a scalar function defined on the mesh, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
f_test = Function(Q)
f_test.interpolate(FunctionTestIntegralExpression(element=Q.ufl_element()))

test_mesh_integral_errors = []

test_mesh_integral_errors.append(msh.test_mesh_integral(0.5287414193220428, f_test, dx, '\int dx f_surface'))
test_mesh_integral_errors.append(msh.test_mesh_integral(0.596540161473517, f_test, dp_1, '\int dp f_{p_1}'))
test_mesh_integral_errors.append(msh.test_mesh_integral(0.1588462551091818, f_test, dp_2, '\int dp f_{p_2}'))
test_mesh_integral_errors.append(msh.test_mesh_integral(cal.curve_integral(function_test_integral_expression, curve_line_12), f_test, dline_12, '\int dl f_{line_12}'))
test_mesh_integral_errors.append(msh.test_mesh_integral(cal.curve_integral(function_test_integral_expression, curve_arc_21), f_test, darc_21, '\int dl f_{arc_21}'))
test_mesh_integral_errors.append(msh.test_mesh_integral(0.652012217844941, f_test, dline_34, '\int dl f_{line_34}'))

print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{max(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')



