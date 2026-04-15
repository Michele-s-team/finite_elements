import colorama as col
from fenics import *
import importlib
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import input_output as io
import mesh.utils as msh

import switch_problem as swi

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)



# check if the boundary conditions (BCs) are satisfied
print("Check of BCs:")

print(f"\t\t<<(u - phi)^2>>_[partial Omega square] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u, fsp.u_exact, rmsh.ds_mesh[0]['ds']):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print(f"\t\t<<([u]_i - [u_exact]_i) * ([u]_i - [u_exact]_i)>>_[partial Omega lr] = {col.Fore.RED}{msh.abs_wrt_measure(sqrt((msh.jump(fsp.u, bgeo.facet_normal[0])[i] - msh.jump(fsp.u_exact, bgeo.facet_normal[0])[i]) * (msh.jump(fsp.u, bgeo.facet_normal[0])[i] - msh.jump(fsp.u_exact, bgeo.facet_normal[0])[i])), rmsh.ds_mesh[0]['dS_shape']):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print(f"\t\t<<([partial_i u]_i - d)^2>>_[partial Omega lr] = {col.Fore.RED}{msh.difference_wrt_measure(msh.jump(fsp.u.dx(i), bgeo.facet_normal[0])[i], msh.average(fsp.d), rmsh.ds_mesh[0]['dS_shape']):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")



print("Comparison with exact solution: ")

print(f"\t\t<<(u - u_exact)^2>>_[Omega shape] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u, fsp.u_exact, rmsh.dx_mesh[0]['dx_shape']):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t\t<<(u - u_exact)^2>>_[Omega square] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u, fsp.u_exact, rmsh.dx_mesh[0]['dx_square']):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")


import print_out_solution