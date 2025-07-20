import colorama as col
from fenics import *
import importlib
import ufl as ufl

import function_spaces as fsp
import input_output as io
import mesh as msh

import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)

# check if the boundary conditions (BCs) are satisfied
# for i in range(len(rmsh.lmsh.sub_meshes)):
print(f"* Problem {0}:")
print(f"\t- Check of BCs:")
print(f"\t\t<<(u - phi)^2>>_[partial Omega{0}] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u[0], fsp.u_exact[0], rmsh.ds_sub_mesh[0]['ds']):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

# print(
#     f"\t\t<<|n^i partial_i u  - n^i grad_u_i|^2>>_[partial Omega out_lr + in_tb] = {col.Fore.RED}{msh.difference_wrt_measure(bgeo.facet_normal[i] * (fsp.u.dx(i)), bgeo.facet_normal[i] * fsp.grad_u[i], rmsh.ds_submesh_out_out_lr + rmsh.ds_submesh_out_in_tb):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
#
print(f"\t- Comparison with exact solution: ")
print(f"\t\t<<(u - u_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u[i], fsp.u_exact[i], rmsh.dx_sub_mesh[i]):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

import print_out_solution
