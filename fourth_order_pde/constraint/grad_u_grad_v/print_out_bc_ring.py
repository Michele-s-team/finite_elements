import colorama as col
from fenics import *
import importlib
import ufl as ufl

import function_spaces as fsp
import differential_geometry.manifold.geometry as geo
import input_output as io
import mesh.utils as msh

import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)

z_output, u_output, omega_z_output, omega_u_output, mu_output = fsp.psi.split(deepcopy=True)

print("Check of BCs: ")
print(f"\t<<(z - z_exact)^2>>_[partial Omega] = {col.Fore.RED}{msh.difference_wrt_measure(z_output, fsp.z_exact, rmsh.ds):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t<<(mu - mu_exact)^2>>_[partial Omega] = {col.Fore.RED}{msh.difference_wrt_measure(mu_output, fsp.mu_exact, rmsh.ds):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print("Comparison with exact solution: ")
print(f"\t<<(z - z_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure(z_output, fsp.z_exact, rmsh.dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t<<(u - u_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure(u_output, fsp.u_exact, rmsh.dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t<<(mu - mu_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure(mu_output, fsp.mu_exact, rmsh.dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t<<|omega_z - omega_z_exact|^2>>_Omega = {col.Fore.RED}{msh.abs_wrt_measure(geo.ufl_norm(omega_z_output - fsp.omega_z_exact), rmsh.dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t<<|omega_u - omega_u_exact|^2>>_Omega = {col.Fore.RED}{msh.abs_wrt_measure(geo.ufl_norm(omega_u_output - fsp.omega_u_exact), rmsh.dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

import print_out_solution
