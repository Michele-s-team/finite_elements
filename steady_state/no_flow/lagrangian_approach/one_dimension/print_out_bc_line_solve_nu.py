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
    f"\t\t<<|u - u_l|^2>>_[partial Omega l] = {col.Fore.RED}{msh.abs_wrt_measure(sqrt((prout_sol.u_output[0] - rpam.parameters['u_l'][0])**2 + (prout_sol.u_output[1] - rpam.parameters['u_l'][1])**2), rmsh.ds_l):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(u1 - u1_r)^2>>_[partial Omega r] = {col.Fore.RED}{msh.difference_wrt_measure(prout_sol.u_output[0], rpam.parameters['u_r'][0], rmsh.ds_r):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(u2 - u2_r)^2>>_[partial Omega r] = {col.Fore.RED}{msh.difference_wrt_measure(prout_sol.u_output[1], rpam.parameters['u_r'][1], rmsh.ds_r):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

