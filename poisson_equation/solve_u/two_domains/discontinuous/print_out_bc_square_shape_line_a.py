import colorama as col
from fenics import *
import importlib
import ufl as ufl

import input_output as io
import mesh.utils as msh

import switch_problem as swi

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)



# check if the boundary conditions (BCs) are satisfied
print("Check of BCs:")

print(f"\t\t<<(u - phi)^2>>_[partial Omega square] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u, fsp.u_exact, rmsh.ds_mesh[0]['ds']):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")


print("Comparison with exact solution: ")

print(f"\t\t<<(u - u_exact)^2>>_[Omega shape] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u, fsp.u_exact, rmsh.dx_mesh[0]['dx_shape']):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t\t<<(u - u_exact)^2>>_[Omega square] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u, fsp.u_exact, rmsh.dx_mesh[0]['dx_square']):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")


import print_out_solution