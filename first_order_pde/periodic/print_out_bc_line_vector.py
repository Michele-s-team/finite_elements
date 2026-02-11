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
    f"\t\tu_[partial Omega l] - u_[partial Omega r] = {col.Fore.RED}{fsp.u(rmsh.parameters['x_l']) - fsp.u(rmsh.parameters['x_r']):.{sys_io.number_of_decimals}e}{col.Style.RESET_ALL}")


print("Comparison with exact solution: ")
print(
    f"\t\t<<(u - u_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u, fsp.u_exact, rmsh.dx):.{sys_io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(grad_u - grad_u_exact)^2>>_Omega = {col.Fore.RED}{msh.abs_wrt_measure(sqrt((fsp.grad_u[i] - fsp.grad_u_exact[i]) * (fsp.grad_u[i] - fsp.grad_u_exact[i])), rmsh.dx):.{sys_io.number_of_decimals}e}{col.Style.RESET_ALL}")
