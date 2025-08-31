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
vp = importlib.import_module(swi.vp)

i, j, k, l = ufl.indices(4)

# check if the boundary conditions (BCs) are satisfied
print("Check of BCs:")
print(f"\t\t<<|u_i - u_in_i|^2>>_[partial Omega in] = {col.Fore.RED}{msh.abs_wrt_measure((fsp.u[i] - fsp.u_in[i]) * (fsp.u[i] - fsp.u_in[i]), rmsh.ds_ellipse):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t\t<<|u_i - u_out_i|^2>>_[partial Omega in] = {col.Fore.RED}{msh.abs_wrt_measure((fsp.u[i] - fsp.u_out[i]) * (fsp.u[i] - fsp.u_out[i]), rmsh.ds_square):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

# print("Comparison with exact solution: ")
# print(f"\t\t<<(u_1 - u_1_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u[0], fsp.u_in[0], rmsh.dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
# print(f"\t\t<<(u_2 - u_2_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u[1], fsp.u_in[1], rmsh.dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

import print_out_solution
