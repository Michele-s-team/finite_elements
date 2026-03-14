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
    f"\t\t<<(u - phi)^2>>_[partial Omega l] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u, fsp.u_exact, rmsh.ds_l):.{sys_io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f'\t\t|u(x_l) - u(x_r)| = {col.Fore.RED}{abs(fsp.u(rmsh.parameters["x_l"]) - fsp.u(rmsh.parameters["x_r"])):.{sys_io.number_of_decimals}e}{col.Style.RESET_ALL}')


print("Comparison with exact solution: ")
print(
    f"\t\t<<(u - u_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u, fsp.u_exact, rmsh.dx):.{sys_io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(hess_u - hess_u_exact)^2>>_Omega = {col.Fore.RED}{msh.abs_wrt_measure(sqrt((fsp.hess_u[i, j] - fsp.hess_u_exact[i, j]) * (fsp.hess_u[i, j] - fsp.hess_u_exact[i, j])), rmsh.dx):.{sys_io.number_of_decimals}e}{col.Style.RESET_ALL}")
