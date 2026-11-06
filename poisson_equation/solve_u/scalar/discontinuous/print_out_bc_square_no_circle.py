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


u_output, s_output = fsp.psi.split( deepcopy=True )


# check if the boundary conditions (BCs) are satisfied
print("Check of BCs:")
print(f"\t\t<<(u - phi)^2>>_[partial Omega] = {col.Fore.RED}{msh.difference_wrt_measure(u_output, fsp.u_exact, rmsh.ds):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print("Comparison with exact solution: ")
print(f"\t\t<<(u - u_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure(u_output, fsp.u_exact, rmsh.dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t\t<<(s - s_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure(s_output, fsp.s_exact, rmsh.dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print(f"\t\terror_norm(u, u_exact)_Omega = {col.Fore.RED}{fu.error_norm(u_output, fsp.u_exact, rmsh.dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t\terror_norm(s, s_exact)_Omega = {col.Fore.RED}{fu.error_norm(s_output, fsp.s_exact, rmsh.dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

import print_out_solution