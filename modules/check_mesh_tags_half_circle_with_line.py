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

# CHANGE PARAMETERS HERE
c_test = [0.3, 0.76]
r_test = 0.345
# CHANGE PARAMETERS HERE

Q_test = FunctionSpace(lmsh.mesh, 'P', 1)


def function_test_integrals(x):
    return np.cos(geo.my_norm(np.subtract(x, c_test)) - r_test) ** 2.0


# analytical expression for a  scalar function used to test the ds
class FunctionTestIntegrals(UserExpression):
    def eval(self, values, x):
        # values[0] = 1
        values[0] = function_test_integrals(x)
        # values[0] = x[0]

    def value_shape(self):
        return (1,)


function_test_integral_fenics = Function(Q_test)
function_test_integral_fenics.interpolate(FunctionTestIntegrals(element=Q_test.ufl_element()))


# curve relative to arc_21: it returns [[x[0](t), x[1](t)] , [x[0]'(t), x[1]'(t)]]
def curve_arc_21(t):
    return [[np.cos(np.pi * t), -np.sin(np.pi * t)], [-  np.pi * np.sin(np.pi * t), -np.pi * np.cos(np.pi * t)]]


# curve relative to line_12: it returns [[x[0](t), x[1](t)] , [x[0]'(t), x[1]'(t)]]
def curve_line_12(t):
    return cal.line([rmsh.r, 0], [-rmsh.r, 0], t)

integral_exact_dline_12 = cal.curve_integral(function_test_integrals, curve_line_12)
integral_exact_darc_21 = cal.curve_integral(function_test_integrals, curve_arc_21)

test_mesh_integral_errors = []

# test_mesh_integral_errors.append(msh.test_mesh_integral(0.5287414193220428, function_test_integral_fenics, rmsh.dx, '\int dx f_surface'))
# test_mesh_integral_errors.append(msh.test_mesh_integral(0.596540161473517, function_test_integral_fenics, rmsh.dp_1, '\int dp f_{p_1}'))
# test_mesh_integral_errors.append(msh.test_mesh_integral(0.1588462551091818, function_test_integral_fenics, rmsh.dp_2, '\int dp f_{p_2}'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_dline_12, function_test_integral_fenics, rmsh.dline_12, '\int dl f_{line_12}'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_darc_21, function_test_integral_fenics, rmsh.darc_21, '\int dl f_{arc_21}'))
# test_mesh_integral_errors.append(msh.test_mesh_integral(0.652012217844941, function_test_integral_fenics, rmsh.dline_34, '\int dl f_{line_34}'))

print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{max(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')
