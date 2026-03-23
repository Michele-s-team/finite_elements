import colorama as col
from fenics import *
import importlib
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp
import input_output as io
import mesh.utils as msh

import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)

print(f"\t- Check of BCs:")




print(f"\t\tBCs for sub_mesh_{0}_{1}:")
print(f"\t\t\t<<(u - phi)^2>>_[partial Omega_{0}_{1}] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u[0][1], fsp.u_exact[0][1], rmsh.ds_sub_mesh[0][1]['ds']):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print(f"\t\tBCs for sub_mesh_{0}_{0}:")
print(f"\t\t\t<<(u - phi)^2>>_[partial Omega {0}_{0}] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u[0][0], fsp.u_1_on_0_0, rmsh.ds_sub_mesh[0][0]['ds']):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print(f"\t\tBCs for mesh_{1}:")
print(f'\t\t\t|u[1](x_l) - u[1](x_r)| = {col.Fore.RED}{abs(fsp.u[1](rmsh.lmsh.mesh_parameters[1]["x_l"]) - fsp.u[1](rmsh.lmsh.mesh_parameters[1]["x_r"])):.{io.number_of_decimals}e}{col.Style.RESET_ALL}')


print(f"\t- Comparison with exact solution: ")
for i in range(2):
    print(f"\t\t<<(u[{0}][{i}] - u[{0}][{i}]_exact)^2>>_[Omega {0} {i}] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u[0][i], fsp.u_exact[0][i], rmsh.dx_sub_mesh[0][i]):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print(f"\t\t<<(u[1] - u[1]_exact)^2>>_[Omega {1}] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u[1], fsp.u_exact[1], rmsh.dx_mesh[1]):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")


import print_out_solution
