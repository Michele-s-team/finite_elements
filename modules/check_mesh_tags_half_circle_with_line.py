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

integral_exact_dx = cal.surface_integral_disk_slice(function_test_integrals, rmsh.r, np.pi, 2 * np.pi, rmsh.c_r)

integral_exact_dline_12 = cal.curve_integral_line(function_test_integrals, rmsh.c_1, rmsh.c_2)
integral_exact_darc_21 = cal.curve_integral_circle_arc(function_test_integrals, rmsh.r, np.pi, 2 * np.pi, rmsh.c_r)

integral_exact_dline_34 = cal.curve_integral_line(function_test_integrals, rmsh.c_3, rmsh.c_4)

integral_exact_dp1 = function_test_integrals([rmsh.r, 0])
integral_exact_dp2 = function_test_integrals([-rmsh.r, 0])

test_mesh_integral_errors = []

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_dx, function_test_integral_fenics, rmsh.dx, '\int dx f'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_dp1, function_test_integral_fenics, rmsh.dp_line_in_start, '\int dp f_{p_1}'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_dp2, function_test_integral_fenics, rmsh.dp_line_in_end, '\int dp f_{p_2}'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_dline_12, function_test_integral_fenics, rmsh.ds_line, '\int dl f_{line_12}'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_darc_21, function_test_integral_fenics, rmsh.ds_arc, '\int dl f_{arc_21}'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_dline_34, function_test_integral_fenics, rmsh.ds_line_in, '\int dl f_{line_34}'))

print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{max(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')
