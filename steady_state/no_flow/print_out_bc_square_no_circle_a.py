import colorama as col
from fenics import *
import importlib
import ufl as ufl

import boundary_geometry as bgeo
import geometry as geo
import input_output as io
import mesh as msh
import print_out_solution as prout
import read_parameters_solve as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

i, j, k, l = ufl.indices(4)



print("Check of BCs:")
print("1)")
print(
    f"\t\t<<(z - phi)^2>>_partial Omega t = {col.Fore.RED}{msh.difference_wrt_measure(prout.z_output, rpam.parameters['z_t_const'], rmsh.ds_t):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(z - phi)^2>>_partial Omega b = {col.Fore.RED}{msh.difference_wrt_measure(prout.z_output, rpam.parameters['z_b_const'], rmsh.ds_b):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print("2)")
print(
    f"\t\t<<(n^i \omega_i - psi )^2>>_partial Omega t = {col.Fore.RED}{msh.difference_wrt_measure((bgeo.n_tb(prout.omega_output))[i] * prout.omega_output[i], vp.n_omega_t, rmsh.ds_t):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(n^i \omega_i - psi )^2>>_partial Omega b = {col.Fore.RED}{msh.difference_wrt_measure((bgeo.n_tb(prout.omega_output))[i] * prout.omega_output[i], vp.n_omega_b, rmsh.ds_b):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print("3)")
print(
    f"\t\t<<[mu - H(omega)]^2>>_[partial Omega] = {col.Fore.RED}{msh.difference_wrt_measure(prout.mu_output, geo.H(prout.omega_output), rmsh.ds):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")


import print_out_force_on_boundary_bc_square_no_circle



