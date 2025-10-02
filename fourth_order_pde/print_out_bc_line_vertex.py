import colorama as col
from fenics import *
import importlib
import numpy as np
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp
import input_output as io
import mesh.utils as msh

import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)

z_output, omega_output, mu_output = fsp.psi.split( deepcopy=True )
rho_output, tau_output = fsp.psi_pp.split( deepcopy=True )

print( "Check of BCs: " )

print( "\tMain variational problem:" )

print( f"\t\t<<(z - z_exact)^2>>_[partial Omega l] = {col.Fore.RED}{msh.difference_wrt_measure( z_output, fsp.z_exact, rmsh.ds_l):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
print( f"\t\t<<(z - z_exact)^2>>_[partial Omega r] = {col.Fore.RED}{msh.difference_wrt_measure( z_output, fsp.z_exact, rmsh.ds_r):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
print( f"\t\t<<(omega - omega_exact)^2>>_[ds_m] = {col.Fore.RED}{ msh.difference_wrt_measure( omega_output, fsp.omega_exact, rmsh.ds_m):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
print( f"\t\t<<(mu - mu_exact)^2>>_[ds_m] = {col.Fore.RED}{ msh.difference_wrt_measure( mu_output, fsp.mu_exact, rmsh.ds_m):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )


print( "\tPost-processing variational problem:" )
print(
    f"\t\t<<(rho - rho_exact)^2>>_[partial Omega] = {col.Fore.RED}{msh.difference_on_boundary( rho_output, fsp.rho_exact ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
print( f"\t\t<<(tau - tau_exact)^2>>_[partial Omega] = {col.Fore.RED}{msh.difference_on_boundary( tau_output, fsp.f ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )


print("Comparison with exact solution: ")

print("\tMain variational problem:")

print(f"\t\t<<(z - z_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure(fsp.z, fsp.z_exact, rmsh.dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t\t<<(omega - omega_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure(fsp.omega, fsp.omega_exact, rmsh.dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t\t<<(mu - mu_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure(fsp.mu, fsp.mu_exact, rmsh.dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

print("\tPost-processing variational problem:")

print(f"\t\t<<(rho - rho_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure(fsp.rho, fsp.rho_exact, rmsh.dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(f"\t\t<<(tau - tau_exact)^2>>_Omega = {col.Fore.RED}{msh.difference_wrt_measure(fsp.tau, fsp.tau_exact, rmsh.dx):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")


import print_out_solution