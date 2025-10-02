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
print(f"\t\t<<(u - u_exact)^2>>_[partial Omega l] = {col.Fore.RED}{msh.difference_wrt_measure(u_output, fsp.u_exact, rmsh.ds_l):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t\t<<(v - v_exact)^2>>_[partial Omega] = {col.Fore.RED}{msh.difference_wrt_measure(v_output, fsp.v_exact, rmsh.ds_lr):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print("Comparison with exact solution: ")
print(f"\t\t<<(u - u_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_in_bulk(u_output, fsp.u_exact):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t\t<<(v - v_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_in_bulk(v_output, fsp.v_exact):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

import print_out_solution
