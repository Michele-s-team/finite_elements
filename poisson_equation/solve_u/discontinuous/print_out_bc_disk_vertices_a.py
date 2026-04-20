'''
here I check that the BCs are satisfied and that the solution agrees with the exact one. Note taht the solution of the variational problem is   defined modulo an additive constant, so I compare u - u([0, 0]) with u_exact - u_exact([0, 0]) (both functions have been shifted so they are 0 at x = [0, 0])
'''

import colorama as col
from fenics import *
import importlib
import ufl as ufl

import function as fu
import function_spaces as fsp
import differential_geometry.boundary.geometry as bgeo
import input_output as io
import mesh.utils as msh

import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)




# check if the boundary conditions (BCs) are satisfied
print("Check of BCs:")
print(f"\t\t<<(n_i \partial_i u - n_i \partial_i u_exact)^2>>_[partial Omega] = {col.Fore.RED}{msh.difference_wrt_measure(bgeo.facet_normal[i] * fsp.u.dx(i), bgeo.facet_normal[i] * fsp.u_exact.dx(i), rmsh.ds):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print("Comparison with exact solution: ")
print(f"\t\t<<((u - u([0, 0]))- (u_exact - u_exact([0, 0]))^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure(project(fsp.u - Constant(fsp.u([0, 0])), fsp.Q), project(fsp.u_exact - Constant(fsp.u_exact([0,0])), fsp.Q), rmsh.dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print(f"\t\terror_norm(u - u([0, 0]), u_exact - u_exact([0, 0]))_Omega = {col.Fore.RED}{fu.error_norm(project(fsp.u - Constant(fsp.u([0, 0])), fsp.Q), project(fsp.u_exact - Constant(fsp.u_exact([0,0])), fsp.Q), rmsh.dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

import print_out_solution