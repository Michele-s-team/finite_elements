import colorama as col
from fenics import *
import importlib
import ufl as ufl

import function as fu
import differential_geometry.boundary.geometry as bgeo
import input_output as io
import mesh.utils as msh

import switch_problem as swi

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)

# check if the boundary conditions (BCs) are satisfied
print("Check of BCs:")

print(f"\t\t<<(u - phi)^2>>_[partial Omega square l + lt + lb] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u, fsp.u_exact_l, rmsh.ds_l + rmsh.ds_lt + rmsh.ds_lb):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t\t<<(u - phi)^2>>_[partial Omega square r + rt + rb] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u, fsp.u_exact_r, rmsh.ds_r + rmsh.ds_rt + rmsh.ds_rb):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print(f"\t\t<<[u]_i * [u]_i>>_[partial Omega lr] = {col.Fore.RED}{msh.abs_wrt_measure(sqrt(msh.jump(fsp.u, bgeo.facet_normal)[i] * msh.jump(fsp.u, bgeo.facet_normal)[i]), rmsh.dS_m):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print(f"\t\t<<([n_i partial_i u]_i - d)^2>>_[partial Omega lr] = {col.Fore.RED}{msh.difference_wrt_measure(msh.jump(fsp.u.dx(i), bgeo.facet_normal)[i], msh.average(fsp.d), rmsh.dS_m):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")


print("Comparison with exact solution: ")

print(f"\t\t<<(u - u_exact)^2>>_[Omega l] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u, fsp.u_exact_l, rmsh.dx_l):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t\t<<(u - u_exact)^2>>_[Omega r] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u, fsp.u_exact_r, rmsh.dx_r):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print(f"\t\terror_norm(u, u_exact)_[Omega l] = {col.Fore.RED}{fu.error_norm(fsp.u, fsp.u_exact_l, rmsh.dx_l):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t\terror_norm(u, u_exact)_[Omega r] = {col.Fore.RED}{fu.error_norm(fsp.u, fsp.u_exact_r, rmsh.dx_r):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")


import print_out_solution