from fenics import *
import ufl as ufl
import colorama as col

import boundary_geometry as bgeo
import input_output as io
import mesh as msh
import print_out_solution as prout
import read_mesh_ring as rmsh
import variational_problem_bc_ring as vp

i, j, k, l = ufl.indices(4)

print("Check of BCs:")
print("1)")
print(
    f"\t\t<<(z - phi)^2>>_[partial Omega r] = {col.Fore.RED}{msh.difference_wrt_measure(prout.z_output, vp.z_r_const, rmsh.ds_r):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(z - phi)^2>>_[partial Omega R] = {col.Fore.RED}{msh.difference_wrt_measure(prout.z_output, vp.z_R_const, rmsh.ds_R):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print("2)")
print(
    f"\t\t<<(n^i \omega_i - psi )^2>>_[partial Omega r] = {col.Fore.RED}{msh.difference_wrt_measure((bgeo.n_circle(prout.omega_output))[i] * prout.omega_output[i], vp.omega_r, rmsh.ds_r):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
print(
    f"\t\t<<(n^i \omega_i - psi )^2>>_[partial Omega R] = {col.Fore.RED}{msh.difference_wrt_measure((bgeo.n_circle(prout.omega_output))[i] * prout.omega_output[i], vp.omega_R, rmsh.ds_R):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")

'''
print( "Check if the intermediate PDEs are satisfied:" )
print(
    f"1)\t\t<<(fel + flaplace)^2>>_Omega =  {msh.difference_in_bulk( project( phys.fel_n( prout.omega_output, mu_output, fsp.tau, vp.kappa ), fsp.Q_z ), project( -phys.flaplace( fsp.sigma, prout.omega_output ), fsp.Q_z ) ):.{io.number_of_decimals}e}" )
print(
    f"2) \t\t<<|omega - partial z|^2>>_Omega = {msh.difference_in_bulk( project( sqrt( (prout.omega_output[i] - prout.z_output.dx( i )) * (prout.omega_output[i] - prout.z_output.dx( i )) ), fsp.Q_z ), project( Constant( 0 ), fsp.Q_z ) ):.{io.number_of_decimals}e}" )
print( f"3)\t\t<<[mu - H(omega)]^2>>_Omega =  {msh.difference_in_bulk( mu_output, project( geo.H( prout.omega_output ), fsp.Q_z ) ):.{io.number_of_decimals}e}" )
print( f"4)\t\t<<[tau - Nabla^i nu_i]^2>>_Omega =  {msh.difference_in_bulk( fsp.tau, project( - geo.Nabla_LB(mu_output,prout.omega_output), fsp.Q_z ) ):.{io.number_of_decimals}e}" )
'''

'''
print( "Comparison with exact solution: " )
print( f"1)\t\t<<(z - z_exact)^2>>_Omega = {col.Fore.BLUE}{msh.difference_in_bulk( prout.z_output, fsp.z_exact ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
print(
    f"2)\t\t<<|omega - omega_exact|^2>>_Omega = {col.Fore.BLUE}{msh.difference_in_bulk( project( sqrt( (prout.omega_output[i] - fsp.omega_exact[i]) * (prout.omega_output[i] - fsp.omega_exact[i]) ), fsp.Q_z ), project( Constant( 0 ), fsp.Q_z ) ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
print( f"3)\t\t<<(mu - mu_exact)^2>>_Omega = {col.Fore.BLUE}{msh.difference_in_bulk( mu_output, fsp.mu_exact ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
print(
    f"4)\t\t<<|nu - nu_exact|^2>>_Omega = {col.Fore.BLUE}{msh.difference_in_bulk( project( sqrt( (fsp.nu[i] - fsp.nu_exact[i]) * (fsp.nu[i] - fsp.nu_exact[i]) ), fsp.Q_z ), project( Constant( 0 ), fsp.Q_z ) ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
print( f"5)\t\t<<(tau - tau_exact)^2>>_Omega = {col.Fore.BLUE}{msh.difference_in_bulk( fsp.tau, fsp.tau_exact ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}")
'''
import print_out_forces
import print_out_force_on_circle