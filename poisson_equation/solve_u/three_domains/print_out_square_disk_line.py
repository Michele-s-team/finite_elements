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
print(f"\t\tBCs for sub_mesh {0}:")
print(f"\t\t\t<<(u - phi)^2>>_[partial Omega {0} lrtb] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u[0], fsp.u_0_1_on_0_0, rmsh.ds_sub_mesh[0]['lrtb']):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print(f"\t\tBCs for sub_mesh {1}:")
print(f"\t\t\t<<|n^i partial_i u  - n^i grad_u_i|^2>>_[partial Omega {1} in_lrtb] = {col.Fore.RED}{msh.difference_wrt_measure(bgeo.sub_mesh_facet_normal[1][i] * (fsp.u[1].dx(i)), bgeo.sub_mesh_facet_normal[1][i] * fsp.grad_u[1][i], rmsh.ds_sub_mesh[1]['in_lrtb']):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t\t\t<<(u - phi)^2>>_[partial Omega {1} out_lrtb] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u[1], fsp.u_exact[1], rmsh.ds_sub_mesh[1]['out_lrtb']):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print(f"\t- Comparison with exact solution: ")
for i in range(len(rmsh.lmsh.sub_meshes)):
    print(f"\t\t<<(u - u_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u[i], fsp.u_exact[i], rmsh.dx_sub_mesh[i]):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")


import print_out_solution
