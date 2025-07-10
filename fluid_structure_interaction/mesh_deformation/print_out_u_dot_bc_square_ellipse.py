import colorama as col
from fenics import *
import importlib
import ufl_legacy as ufl

import function_spaces as fsp
import input_output as io
import mesh as msh

import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)
vp_dot = importlib.import_module(swi.vp_dot)

i, j, k, l = ufl.indices(4)

# check if the boundary conditions (BCs) are satisfied
print("Check of BCs:")
print(f"\t\t<<|u_dot_i - u_dot_in_i|^2>>_[partial Omega in] = {col.Fore.RED}{msh.abs_wrt_measure((fsp.u_dot[i] - fsp.u_dot_in[i]) * (fsp.u_dot[i] - fsp.u_dot_in[i]), rmsh.ds_ellipse):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t\t<<|u_dot_i - u_dot_out_i|^2>>_[partial Omega in] = {col.Fore.RED}{msh.abs_wrt_measure((fsp.u_dot[i] - fsp.u_dot_out[i]) * (fsp.u_dot[i] - fsp.u_dot_out[i]), rmsh.ds_square):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

import print_out_solution
