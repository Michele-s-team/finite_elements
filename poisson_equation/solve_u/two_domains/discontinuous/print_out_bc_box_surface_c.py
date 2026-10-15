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

print(f"\t\t<<(u - phi)^2>>_[partial Omega box] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u, fsp.u_exact, rmsh.ds):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t\t<<(u - phi)^2>>_[partial Omega surface] = {col.Fore.RED}{msh.difference_wrt_measure(msh.side(fsp.u, rmsh.I_box), msh.side(fsp.u_exact, rmsh.I_box), rmsh.dS_surface):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")



print("Comparison with exact solution: ")

print(f"\t\t<<(u - u_exact)^2>>_[Omega surface] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u, fsp.u_exact, rmsh.dx_surface):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t\t<<(u - u_exact)^2>>_[Omega box] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u, fsp.u_exact, rmsh.dx_box):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")


prout_sol = importlib.import_module(swi.prout_sol)
