from fenics import *
import importlib
import ufl as ufl
import colorama as col

import input_output as io
import mesh.utils as msh
import parameters.read.solution as rpam
import switch_problem as swi

prout_sol = importlib.import_module(swi.prout_sol)
rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

i, j, k, l = ufl.indices(4)

print("Check of BCs:")
print(
    f"\t\t<<(psi - psi_l)^2>>_[partial Omega l] = {col.Fore.RED}{msh.difference_wrt_measure(prout_sol.psi_output, rpam.parameters['psi_l'], rmsh.ds_l):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(psi - psi_r)^2>>_[partial Omega r] = {col.Fore.RED}{msh.difference_wrt_measure(prout_sol.psi_output, rpam.parameters['psi_r'], rmsh.ds_r):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print(
    f"\t\t<<|X - X_l|^2>>_[partial Omega l] = {col.Fore.RED}{msh.abs_wrt_measure(sqrt((prout_sol.X_output[0] - rpam.parameters['X_l'][0]) * (prout_sol.X_output[0] - rpam.parameters['X_l'][0]) + (prout_sol.X_output[1] - rpam.parameters['X_l'][1]) * (prout_sol.X_output[1] - rpam.parameters['X_l'][1])), rmsh.ds_l):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(X1 - X1_r)^2>>_[partial Omega r] = {col.Fore.RED}{msh.difference_wrt_measure(prout_sol.X_output[0], rpam.parameters['X_r'][0], rmsh.ds_r):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(X2 - X2_r)^2>>_[partial Omega r] = {col.Fore.RED}{msh.difference_wrt_measure(prout_sol.X_output[1], rpam.parameters['X_r'][1], rmsh.ds_r):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

