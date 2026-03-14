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

# check if the boundary conditions (BCs) are satisfied
print("Check of BCs:")
print(f"\t\t<<(u - phi)^2>>_[partial Omega le ri to bo] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u, fsp.u_exact, rmsh.ds_leri + rmsh.ds_tobo):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<|n^i partial_i u  - n^i grad_u_i|^2>>_[partial Omega fr ba sphere] = {col.Fore.RED}{msh.difference_wrt_measure(bgeo.facet_normal[i] * (fsp.u.dx(i)), bgeo.facet_normal[i] * fsp.grad_u[i], rmsh.ds_frba + rmsh.ds_sphere):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")


print("Comparison with exact solution: ")
print(f"\t\t<<(u - u_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u, fsp.u_exact, rmsh.dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(hess_u - hess_u_exact)^2>>_Omega = {col.Fore.RED}{msh.abs_wrt_measure(sqrt((fsp.hess_u[i, j] - fsp.hess_u_exact[i, j]) * (fsp.hess_u[i, j] - fsp.hess_u_exact[i, j])), rmsh.dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

import print_out_solution
