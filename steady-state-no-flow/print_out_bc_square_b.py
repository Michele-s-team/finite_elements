import colorama as col
from fenics import *
import importlib
import ufl as ufl

import boundary_geometry as bgeo
import geometry as geo
import input_output as io
import mesh as msh
import print_out_solution as prout
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

i, j, k, l = ufl.indices(4)

print("Check of BCs:")
print(
    f"\t\t<<(z - phi)^2>>_square = {col.Fore.RED}{msh.difference_wrt_measure(prout.z_output, vp.z_square_const, rmsh.ds_square):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<|omega_i - psi_i|^2>>_circle = {col.Fore.RED}{msh.abs_wrt_measure(sqrt((prout.omega_output[i] - vp.omega_circle[i]) * (prout.omega_output[i] - vp.omega_circle[i])), rmsh.ds_circle):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(n^i \omega_i - psi )^2>>_lr = {col.Fore.RED}{msh.difference_wrt_measure((bgeo.n_lr(prout.omega_output))[i] * prout.omega_output[i], vp.n_omega_square, rmsh.ds_lr):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(n^i \omega_i - psi )^2>>_tb = {col.Fore.RED}{msh.difference_wrt_measure((bgeo.n_tb(prout.omega_output))[i] * prout.omega_output[i], vp.n_omega_square, rmsh.ds_tb):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print(
    f"\t\t<<[mu - H(omega)]^2>>_[partial Omega] = {col.Fore.RED}{msh.difference_wrt_measure(prout.mu_output, geo.H(prout.omega_output), rmsh.ds):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")



print(f"\n<z>_circle = {assemble(prout.z_output * rmsh.ds_circle) / assemble(Constant(1.0) * rmsh.ds_circle)}")

import print_out_forces
import print_out_force_on_boundary_bc_square