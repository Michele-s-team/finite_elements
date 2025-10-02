import colorama as col
from fenics import *
import importlib
import ufl as ufl

import function_spaces as fsp
import input_output as io
import mesh.utils as msh

import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)

u_output, v_output = fsp.psi.split(deepcopy=True)

print("BCs check: ")
print('\tMain variational problem:')
print(f"\t\t<<(u - u_exact)^2>>_partial Omega = {col.Fore.RED}{msh.difference_on_boundary(u_output, fsp.u_exact):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t\t<<(v - v_exact)^2>>_partial Omega = {col.Fore.RED}{msh.difference_on_boundary(v_output, fsp.v_exact):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print(f"\tPost-processing variational problem:")
print(f"\t\t<<(w - w_exact)^2>>_partial Omega = {col.Fore.RED}{msh.difference_on_boundary(fsp.w, fsp.w_exact):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print("Comparison with exact solution: ")
print('\tMain variational problem:')
print(f"\t\t<<(u - u_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_in_bulk(u_output, fsp.u_exact):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t\t<<(v - v_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_in_bulk(v_output, fsp.v_exact):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print(f"\tPost-processing variational problem:")
print(f"\t\t<<(w - w_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_in_bulk(fsp.w, fsp.w_exact):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

import print_out_solution
