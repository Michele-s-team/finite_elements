import colorama as col
from fenics import *
import ufl as ufl

# import boundary_geometry as bgeo
import function_spaces as fsp
import input_output as io
from load_mesh.interval import load_interval_mesh as lmsh
import mesh as msh

# rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)

# check if the boundary conditions (BCs) are satisfied
print("Check of BCs:")
print(f"\t\t<<(u - phi)^2>>_[partial Omega l] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u, fsp.u_exact, lmsh.ds_l):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t\t<<(u - phi)^2>>_[partial Omega r] = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u, fsp.u_exact, lmsh.ds_r):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print("Comparison with exact solution:")
print(f"\t\t<<(u - u_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure(fsp.u, fsp.u_exact, lmsh.dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")


import print_out_solution
