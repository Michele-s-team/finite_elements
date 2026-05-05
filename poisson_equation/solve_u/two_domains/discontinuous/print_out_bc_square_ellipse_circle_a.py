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
vp = importlib.import_module(swi.vp)

i, j, k, l = ufl.indices(4)



# check if the boundary conditions (BCs) are satisfied
print("Check of BCs:")

print(f"\t\t<<(u - phi)^2>>_[partial Omega square] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u, fsp.u_exact, rmsh.ds_lrtb):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t\t<<(u - phi)^2>>_[partial Omega shape] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u(vp.sub_mesh_0_1_label), fsp.u_exact(vp.sub_mesh_0_1_label), rmsh.dS_ellipse):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")



print("Comparison with exact solution: ")

print(f"\t\t<<(u - u_exact)^2>>_[Omega shape] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u, fsp.u_exact, rmsh.dx_sub_mesh[0]):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t\t<<(u - u_exact)^2>>_[Omega square] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u, fsp.u_exact, rmsh.dx_sub_mesh[1]):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")


prout_sol = importlib.import_module(swi.prout_sol)
