import colorama as col
from fenics import *
import importlib
import ufl as ufl

import boundary_geometry as bgeo
import geometry as geo
import input_output as io
import mesh as msh
import physics as phys
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

i, j, k, l = ufl.indices(4)

import print_out_solution as prout

print("Check of BCs:")
print(
    f"\t\t<<|v^i - v_r^i|^2>>_[partial Omega r] = {col.Fore.RED}{msh.difference_wrt_measure((prout.v_output[i] - vp.v_r[i]) * (prout.v_output[i] - vp.v_r[i]), Constant(0), rmsh.ds_r):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(v^i n_i - v_R)^2>>_[partial Omega R] = {col.Fore.RED}{msh.difference_wrt_measure(bgeo.n_circle(prout.omega_output)[i] * geo.g(prout.omega_output)[i, j] * prout.v_output[j], rpam.parameters['v_R_const'], rmsh.ds_R):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print(
    f"\t\t<<(w - w_r)^2>>_[partial Omega r] = {col.Fore.RED}{msh.difference_wrt_measure(prout.w_output, rpam.parameters['w_r_const'], rmsh.ds_r):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(w - w_R)^2>>_[partial Omega R] = {col.Fore.RED}{msh.difference_wrt_measure(prout.w_output, rpam.parameters['w_R_const'], rmsh.ds_R):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print(
    f"\t\t<<(sigma - sigma_r)^2>>_[partial Omega r] = {col.Fore.RED}{msh.difference_wrt_measure(prout.sigma_output, rpam.parameters['sigma_r_const'], rmsh.ds_r):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print(
    f"\t\t<<(z - phi)^2>>_[partial Omega r] = {col.Fore.RED}{msh.difference_wrt_measure(prout.z_output, vp.z_r, rmsh.ds_r):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(z - phi)^2>>_[partial Omega R] = {col.Fore.RED}{msh.difference_wrt_measure(prout.z_output, vp.z_R, rmsh.ds_R):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print(
    f"\t\t<<(n^i \omega_i - psi )^2>>_[partial Omega r] = {col.Fore.RED}{msh.difference_wrt_measure((bgeo.n_circle(prout.omega_output))[i] * prout.omega_output[i], vp.omega_r, rmsh.ds_r):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(n^i \omega_i - psi )^2>>_[partial Omega R] = {col.Fore.RED}{msh.difference_wrt_measure((bgeo.n_circle(prout.omega_output))[i] * prout.omega_output[i], vp.omega_R, rmsh.ds_R):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print(
    f"\t\t<<[mu - H(omega)]^2>>_[partial Omega] = {col.Fore.RED}{msh.difference_wrt_measure(prout.mu_output, geo.H(prout.omega_output), rmsh.ds):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")


import print_out_force_on_boundary_bc_ring
