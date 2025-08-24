from fenics import *
import importlib
import ufl as ufl
import colorama as col

import input_output as io
import mesh as msh
import print_out_solution as prout
import read_parameters_solve as rpam
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
    f"\t\t<<(mu - mu_l)^2>>_[partial Omega l] = {col.Fore.RED}{msh.difference_wrt_measure(prout.mu_output, rpam.parameters['mu_l'], rmsh.ds_l):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<|X - X_l|^2>>_[partial Omega l] = {col.Fore.RED}{msh.abs_wrt_measure(sqrt((prout.X_output[0] - rpam.parameters['X_l'][0]) * (prout.X_output[0] - rpam.parameters['X_l'][0]) + (prout.X_output[1] - rpam.parameters['X_l'][1]) * (prout.X_output[1] - rpam.parameters['X_l'][1])), rmsh.ds_l):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
