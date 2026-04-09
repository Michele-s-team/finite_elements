import colorama as col
from fenics import *
import importlib
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import physics.elasticity as ela
import function_spaces as fsp
import input_output as io
import mesh.utils as msh
import parameters.read.solution as rpam


import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

i, j, k, l = ufl.indices(4)

# check if the boundary conditions (BCs) are satisfied
print("Check of BCs:")
print(f"\t\t<<|u|^2>>_[partial Omega l] = {col.Fore.RED}{msh.abs_wrt_measure(fsp.u[i] * fsp.u[i], rmsh.ds_l):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t\t<<(n_i P_ik)^2>>_[partial Omega r t b] = {col.Fore.RED}{msh.abs_wrt_measure( (bgeo.facet_normal[j] * ela.P(fsp.u, rpam.parameters['K'], rpam.parameters['mu'])[i, j]) * (bgeo.facet_normal[k] * ela.P(fsp.u, rpam.parameters['K'], rpam.parameters['mu'])[i, k]), rmsh.ds_r + rmsh.ds_t + rmsh.ds_b):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")


import print_out_solution
