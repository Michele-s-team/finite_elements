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
print(f"\t- Check of BCs:")

print(f"\t\tBCs for sub_mesh {0}:")
print(f"\t\t\t<<(u - u_exact)^2>>_[partial Omega {0} lr] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u[0], fsp.u_exact[0], rmsh.ds_sub_mesh[0]['ds_lr']):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t\t\t<<(u - u_exact)^2>>_[partial Omega {0} t] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u[0], fsp.u_1_on_0, rmsh.ds_sub_mesh[0]['ds_t']):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t\t\t<<(u - u_exact)^2>>_[partial Omega {0} b] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u[0], fsp.u_exact[0], rmsh.ds_sub_mesh[0]['ds_b']):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print(f"\t\tBCs for sub_mesh {1}:")
print(f"\t\t\t<<(u - u_exact)^2>>_[partial Omega{1}] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u[1], fsp.u_exact[1], rmsh.ds_sub_mesh[1]['ds']):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")


print(f"\t- Comparison with exact solution: ")
for i in range(len(rmsh.lmsh.sub_meshes)):
    print(f"\t\t<<(u - u_exact)^2>>_[Omega {i}] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u[i], fsp.u_exact[i], rmsh.dx_sub_mesh[i]):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

import print_out_solution
