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
for i in range(len(rmsh.lmsh.sub_meshes)):
    print(f"* Check of BCs for problem {i}:")
    print(f"\t\t<<(u - phi)^2>>_[partial Omega{i}] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u[i], fsp.u_exact[i], rmsh.ds_sub_mesh_lrtb[i]):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

# print(
#     f"\t\t<<|n^i partial_i u  - n^i grad_u_i|^2>>_[partial Omega out_lr + in_tb] = {col.Fore.RED}{msh.difference_wrt_measure(bgeo.facet_normal[i] * (fsp.u.dx(i)), bgeo.facet_normal[i] * fsp.grad_u[i], rmsh.ds_submesh_out_out_lr + rmsh.ds_submesh_out_in_tb):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
#
# print("Comparison with exact solution: ")
# print(f"\t\t<<(u - u_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u, fsp.u_exact, rmsh.dx_out):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
# print(
#     f"\t\t<<(hess_u - hess_u_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure((fsp.hess_u[i, j] - fsp.hess_u_exact[i, j]) * (fsp.hess_u[i, j] - fsp.hess_u_exact[i, j]), Constant(0), rmsh.dx_out):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
#
# import print_out_solution