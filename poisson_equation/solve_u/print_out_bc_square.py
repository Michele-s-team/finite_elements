from fenics import *
import ufl as ufl
import colorama as col

import boundary_geometry as bgeo
import input_output as io
import mesh as msh
import print_out_solution as prout
import read_mesh_square as rmsh
import variational_problem_bc_square_a as vp

i, j, k, l = ufl.indices(4)



# check if the boundary conditions (BCs) are satisfied
print("Check of BCs:")
print(f"\t\t<<(u - phi)^2>>_[partial Omega tb] = {col.Fore.RED}{msh.difference_wrt_measure(u, u_exact, ds_tb):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<|n^i partial_i u  - n^i grad_u_i|^2>>_[partial Omega lr] = {col.Fore.RED}{msh.difference_wrt_measure(n[i] * (u.dx(i)), n[i] * grad_u[i], ds_lr):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print("Comparison with exact solution: ")
print(f"\t\t<<(u - u_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure(u, u_exact, dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(hess_u - hess_u_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure((hess_u[i, j] - hess_u_exact[i, j]) * (hess_u[i, j] - hess_u_exact[i, j]), Constant(0), dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
