import colorama as col
from fenics import *
import numpy as np

import boundary_geometry as bgeo
import calculus as cal
import geometry as geo
import mesh as msh

import input_output as io
# the module read_mesh_square which is being called will be in the local folder, e.g., in steady-state-no-flow
import read_mesh_ring_with_circle as rmsh

print(f'Module {__file__} called {rmsh.__file__}', flush=True)

# CHANGE PARAMETERS HERE
c_test = [0.3, 0.76]
r_test = 0.345
# CHANGE PARAMETERS HERE


# a function space used solely to define function_test_integrals_fenics
Q_test = FunctionSpace(bgeo.mesh, 'P', 2)

# function_test_integrals_fenics is a function of two variables, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
def function_test_integrals(x):
    return (np.cos(geo.my_norm(np.subtract(x, c_test)) - r_test) ** 2.0)

# function_test_integrals_fenics is the same as function_test_integrals, but in fenics format
function_test_integrals_fenics = Function(Q_test)

# analytical expression for a  scalar function used to test the ds
class FunctionTestIntegrals(UserExpression):
    def eval(self, values, x):
        values[0] = function_test_integrals(x)

    def value_shape(self):
        return (1,)


function_test_integrals_fenics.interpolate(FunctionTestIntegrals(element=Q_test.ufl_element()))

integral_exact_dx_r_rho = cal.surface_integral_ring(function_test_integrals, rmsh.r, rmsh.rho, rmsh.c_r)
integral_exact_dx_rho_R = cal.surface_integral_ring(function_test_integrals, rmsh.rho, rmsh.R, rmsh.c_r)

integral_exact_dx = integral_exact_dx_r_rho +  integral_exact_dx_rho_R

integral_exact_ds_r = cal.curve_integral_circle(function_test_integrals, rmsh.r, rmsh.c_r)
integral_exact_ds_rho = cal.curve_integral_circle(function_test_integrals, rmsh.rho, rmsh.c_rho)
integral_exact_ds_R = cal.curve_integral_circle(function_test_integrals, rmsh.R, rmsh.c_R)

integral_exact_ds = integral_exact_ds_r + integral_exact_ds_R

test_mesh_integral_errors = []


test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_dx_r_rho, function_test_integrals_fenics, rmsh.dx_r_rho, '\int f dx_r_rho'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_dx_rho_R, function_test_integrals_fenics, rmsh.dx_rho_R, '\int f dx_rho_R'))

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_dx, function_test_integrals_fenics, rmsh.dx, '\int f dx'))


test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_r, function_test_integrals_fenics, rmsh.ds_r, '\int f ds_r'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_rho, function_test_integrals_fenics, rmsh.ds_rho, '\int f ds_rho'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_R, function_test_integrals_fenics, rmsh.ds_R, '\int f ds_R'))

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds, function_test_integrals_fenics, rmsh.ds, '\int f ds'))

print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{max(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')


msh.check_mesh_symmetry(bgeo.mesh, rmsh.c_r)
