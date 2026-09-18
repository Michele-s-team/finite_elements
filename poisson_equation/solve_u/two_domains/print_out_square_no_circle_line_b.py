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

i, j, k, l = ufl.indices(4)


# check if the boundary conditions (BCs) are satisfied
print(f"\t- Check of BCs:")

print(f"\t\tBCs for mesh {0}:")
print(f"\t\t\t<<|n^i partial_i u  - n^i grad_u_i|^2>>_[partial Omega{0}] = {col.Fore.RED}{msh.difference_wrt_measure(bgeo.facet_normal[0][i] * (fsp.u[0].dx(i)), bgeo.facet_normal[0][i] * fsp.grad_u[0][i], rmsh.ds_mesh[0]['ds']):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")


print(f"\t\tBCs for mesh {1}:")
print(f"\t\t\t<<(u - u_exact)^2>>_[partial Omega{1}] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u[1], fsp.u_exact[1], rmsh.ds_mesh[1]['ds']):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")


print(f"\t- Comparison with exact solution: ")
for i in range(len(rmsh.lmsh.mesh)):
    print(f"\t\t<<(u - u_exact)^2>>_[Omega {i}] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u[i], fsp.u_exact[i], rmsh.dx_mesh[i]):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

import print_out_solution
