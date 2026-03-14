import colorama as col
from fenics import *
import input_output as io
import importlib
import mesh.utils as msh
import sys
import ufl as ufl

module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)


import differential_geometry.boundary.geometry as bgeo
import input_output as sys_io
import switch_problem as swi

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)

# check if the boundary conditions (BCs) are satisfied
print("Check of BCs:")
print(
    f"\t\t<<(u - phi)^2>>_[partial Omega tb] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u, fsp.u_exact, rmsh.ds_tb):.{sys_io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<|n^i partial_i u  - n^i grad_u_i|^2>>_[partial Omega lr] = {col.Fore.RED}{msh.difference_wrt_measure(bgeo.facet_normal[i] * (fsp.u.dx(i)), bgeo.facet_normal[i] * fsp.grad_u[i], rmsh.ds_lr):.{sys_io.number_of_decimals}e}{col.Style.RESET_ALL}")

print("Comparison with exact solution: ")
print(
    f"\t\t<<(u - u_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u, fsp.u_exact, rmsh.dx):.{sys_io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(hess_u - hess_u_exact)^2>>_Omega = {col.Fore.RED}{msh.abs_wrt_measure(sqrt((fsp.hess_u[i, j] - fsp.hess_u_exact[i, j]) * (fsp.hess_u[i, j] - fsp.hess_u_exact[i, j])), rmsh.dx):.{sys_io.number_of_decimals}e}{col.Style.RESET_ALL}")
