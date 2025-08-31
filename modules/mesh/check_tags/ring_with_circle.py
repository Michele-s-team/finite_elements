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

rmsh = importlib.import_module('mesh.read.ring_with_circle')

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


# function_test_integrals_fenics is the same as function_test_integrals, but in fenics format
function_test_integrals_fenics = Function(Q_test)


# analytical expression for a  scalar function used to test the ds
class FunctionTestIntegrals(UserExpression):
    def eval(self, values, x):
        values[0] = function_test_integrals(x)

    def value_shape(self):
        return (1,)


function_test_integrals_fenics.interpolate(FunctionTestIntegrals(element=Q_test.ufl_element()))

integral_exact_dx_r_rho = cal.surface_integral_ring(function_test_integrals, rmsh.parameters["r"], rmsh.parameters["rho"], rmsh.parameters["c_r"][:2])
integral_exact_dx_rho_R = cal.surface_integral_ring(function_test_integrals, rmsh.parameters["rho"], rmsh.parameters["R"], rmsh.parameters["c_r"][:2])

integral_exact_dx = integral_exact_dx_r_rho + integral_exact_dx_rho_R

integral_exact_ds_r = cal.curve_integral_circle(function_test_integrals, rmsh.parameters["r"], rmsh.parameters["c_r"][:2])
integral_exact_ds_rho = cal.curve_integral_circle(function_test_integrals, rmsh.parameters["rho"], rmsh.parameters["c_rho"][:2])
integral_exact_ds_R = cal.curve_integral_circle(function_test_integrals, rmsh.parameters["R"], rmsh.parameters["c_R"][:2])

integral_exact_ds = integral_exact_ds_r + integral_exact_ds_R

test_mesh_integral_errors = dict([])

test_mesh_integral_errors['\int f dx_r_rho'] = msh.test_mesh_integral(integral_exact_dx_r_rho, function_test_integrals_fenics, rmsh.dx_r_rho, '\int f dx_r_rho')
test_mesh_integral_errors['\int f dx_rho_R'] = msh.test_mesh_integral(integral_exact_dx_rho_R, function_test_integrals_fenics, rmsh.dx_rho_R, '\int f dx_rho_R')

test_mesh_integral_errors['\int f dx'] = msh.test_mesh_integral(integral_exact_dx, function_test_integrals_fenics, rmsh.dx, '\int f dx')

test_mesh_integral_errors['\int f ds_r'] = msh.test_mesh_integral(integral_exact_ds_r, function_test_integrals_fenics, rmsh.ds_r, '\int f ds_r')
test_mesh_integral_errors['\int f ds_rho'] = msh.test_mesh_integral(integral_exact_ds_rho, function_test_integrals_fenics, rmsh.ds_rho, '\int f ds_rho')
test_mesh_integral_errors['\int f ds_R'] = msh.test_mesh_integral(integral_exact_ds_R, function_test_integrals_fenics, rmsh.ds_R, '\int f ds_R')

test_mesh_integral_errors['\int f ds'] = msh.test_mesh_integral(integral_exact_ds, function_test_integrals_fenics, rmsh.ds, '\int f ds')

# print to file the residuals of the tests of the mesh integrals
li.print_to_csv_file(test_mesh_integral_errors, 'check/test_mesh_integrals.csv')

print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{max(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')

msh.check_mesh_symmetry(lmsh.mesh, rmsh.parameters["c_r"][:2])
