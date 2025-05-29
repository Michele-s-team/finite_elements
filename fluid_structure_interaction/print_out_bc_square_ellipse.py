import colorama as col
from fenics import *
import importlib
import ufl as ufl

import boundary_geometry as bgeo
import function_spaces as fsp
import input_output as io
import mesh as msh

import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)

# check if the boundary conditions (BCs) are satisfied
print("Check of BCs:")
print(f"\t\t<<(u_1 - phi_1)^2>>_[partial Omega ] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u[0], fsp.u_exact[0], rmsh.ds):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t\t<<(u_2 - phi_2)^2>>_[partial Omega ] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u[1], fsp.u_exact[1], rmsh.ds):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print("Comparison with exact solution: ")
print(f"\t\t<<(u_1 - u_1_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u[0], fsp.u_exact[0], rmsh.dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t\t<<(u_2 - u_2_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u[1], fsp.u_exact[1], rmsh.dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

import print_out_solution