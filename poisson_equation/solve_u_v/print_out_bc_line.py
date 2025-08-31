import colorama as col
from fenics import *
import importlib
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp
import differential_geometry.manifold.geometry as geo
import input_output as io
import mesh.utils as msh
import switch_problem as swi


rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)

# print out the solution
u_output, v_output = fsp.psi.split(deepcopy=True)


# check if the boundary conditions are satisfied
print("BCs check: ")
print(f"\t\t<<(u - u_exact)^2>>_[partial Omega] = {col.Fore.RED}{msh.difference_wrt_measure(u_output, fsp.u_exact, rmsh.ds):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t\t<<(n.v-n.v_exact)^2>>_[partial Omega] = {col.Fore.RED}{msh.difference_wrt_measure(bgeo.facet_normal[i] * v_output[i], bgeo.facet_normal[i] * fsp.v_exact[i], rmsh.ds):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

# check if the FE solution agrees with the exact one
print("Comparison with exact solution: ")
print(f"\t\t<<(u - u_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure(u_output, fsp.u_exact, rmsh.dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t\t<<|v - v_exact|^2>>_Omega = {col.Fore.RED}{msh.abs_wrt_measure(geo.ufl_norm(v_output - fsp.v_exact), rmsh.dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
