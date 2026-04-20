import colorama as col
from fenics import *
import importlib
import ufl as ufl

import function as fu
import function_spaces as fsp
import input_output as io
import mesh.utils as msh

import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)




# check if the boundary conditions (BCs) are satisfied
print("Check of BCs:")
print(f"\t\t<<(u - phi)^2>>_[partial Omega] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u, fsp.u_exact, rmsh.ds):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print("Comparison with exact solution: ")
print(f"\t\t<<((u - u([0, 0]))- (u_exact - u_exact([0, 0]))^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure(project(fsp.u - Constant(fsp.u([0, 0])), fsp.Q), project(fsp.u_exact - Constant(fsp.u_exact([0,0])), fsp.Q), rmsh.dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print(f"\t\terror_norm(u - u([0, 0]), u_exact - u_exact([0, 0]))_Omega = {col.Fore.RED}{fu.error_norm(project(fsp.u - Constant(fsp.u([0, 0])), fsp.Q), project(fsp.u_exact - Constant(fsp.u_exact([0,0])), fsp.Q), rmsh.dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

# print(
#     f"\t\t<<(hess_u - hess_u_exact)^2>>_Omega = {col.Fore.RED}{msh.abs_wrt_measure(sqrt((fsp.hess_u[i, j] - fsp.hess_u_exact[i, j]) * (fsp.hess_u[i, j] - fsp.hess_u_exact[i, j])), rmsh.dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

import print_out_solution