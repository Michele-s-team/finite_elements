import colorama as col
from fenics import *
import importlib
import ufl as ufl

import input_output as io
import mesh.utils as msh
import print_out_solution as prout
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

i, j, k, l = ufl.indices(4)

print("Check of BCs:")
print(
    f"\t\t<<(psi - psi_l)^2>>_[partial Omega l] = {col.Fore.RED}{msh.difference_wrt_measure(prout.psi_output, rpam.parameters['psi_l'], rmsh.ds_l):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(psi - psi_r)^2>>_[partial Omega r] = {col.Fore.RED}{msh.difference_wrt_measure(prout.psi_output, rpam.parameters['psi_r'], rmsh.ds_r):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(omega - omega_l)^2>>_[partial Omega l] = {col.Fore.RED}{msh.difference_wrt_measure(prout.omega_output, rpam.parameters['omega_l'], rmsh.ds_l):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(rho - rho_l)^2>>_[partial Omega l] = {col.Fore.RED}{msh.difference_wrt_measure(prout.rho_output, rpam.parameters['rho_l'], rmsh.ds_l):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(zeta - zeta_l)^2>>_[partial Omega l] = {col.Fore.RED}{msh.difference_wrt_measure(prout.zeta_output, rpam.parameters['zeta_l'], rmsh.ds_l):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
