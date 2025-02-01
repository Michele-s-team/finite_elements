from fenics import *
import ufl as ufl
import colorama as col
import numpy as np

import function_spaces as fsp
import geometry as geo
import input_output as io
import mesh as msh
import physics as phys
import read_mesh_ring as rmsh
import runtime_arguments as rarg
# import variational_problem_bc_square_a as vp
import variational_problem_bc_ring as vp

# import variational_problem_bc_square_no_circle_a as vp

i, j, k, l = ufl.indices( 4 )

xdmffile_check = XDMFFile( (rarg.args.output_directory) + "/check.xdmf" )
xdmffile_check.parameters.update( {"functions_share_mesh": True, "rewrite_function_mesh": False} )

# copy the data of the  solution psi into v_output, ..., z_output, which will be allocated or re-allocated here
z_output, omega_output, mu_output, nu_output = fsp.psi.split( deepcopy=True )

print( "Check of BCs:" )
print("1)")
print( f"\t\t<<(z - phi)^2>>_[partial Omega r] = {col.Fore.RED}{msh.difference_wrt_measure( z_output, vp.z_r_const, rmsh.ds_r ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
print( f"\t\t<<(z - phi)^2>>_[partial Omega R] = {col.Fore.RED}{msh.difference_wrt_measure( z_output, vp.z_R_const, rmsh.ds_R ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
print("2)")
print(
    f"\t\t<<(n^i \omega_i - psi )^2>>_[partial Omega r] = {col.Fore.RED}{msh.difference_wrt_measure( (rmsh.n_circle( omega_output ))[i] * omega_output[i], vp.omega_r, rmsh.ds_r ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
print(
    f"\t\t<<(n^i \omega_i - psi )^2>>_[partial Omega R] = {col.Fore.RED}{msh.difference_wrt_measure( (rmsh.n_circle( omega_output ))[i] * omega_output[i], vp.omega_R, rmsh.ds_R ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
# print( f"3)\t\t<<[mu - H(omega)]^2>>_partial Omega = {col.Fore.RED}{msh.difference_on_boundary( mu_output, project(geo.H(omega_output), fsp.Q_z) ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
# print(
#     f"4)\t\t<<|nu_i - \partial_i mu|^2>>_partial Omega = {col.Fore.RED}{msh.difference_wrt_measure(project(sqrt((nu_output[i] - (mu_output.dx(i))) * (nu_output[i] - (mu_output.dx(i)))), fsp.Q_z), project(Constant(0), fsp.Q_z), rmsh.ds):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
# print( f"5)\t\t<<[tau - Nabla^i nu_i]^2>>_partial Omega =  {col.Fore.RED}{msh.difference_on_boundary( fsp.tau, project( geo.g_c(fsp.omega)[i, j] * geo.Nabla_f(nu_output, omega_output)[i, j], fsp.Q_z ) ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )


print( "Check if the intermediate PDEs are satisfied:" )
print( "1)\t\tCannot be computed" )
print(
    f"2) \t\t<<|omega - partial z|^2>>_Omega = {col.Fore.CYAN}{msh.difference_in_bulk( project( sqrt( (omega_output[i] - z_output.dx( i )) * (omega_output[i] - z_output.dx( i )) ), fsp.Q_z ), project( Constant( 0 ), fsp.Q_z ) ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
print( f"3)\t\t<<[mu - H(omega)]^2>>_Omega =  {col.Fore.CYAN}{msh.difference_in_bulk( mu_output, project( geo.H( omega_output ), fsp.Q_z ) ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
print( f"4)\t\t<<|nu - partial  mu|^2>>_Omega =  {col.Fore.CYAN}{msh.difference_in_bulk( project((nu_output[i] - mu_output.dx( i )) * (nu_output[i] - mu_output.dx( i )), fsp.Q_z), project( Constant(0), fsp.Q_z ) ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
print( f"5)\t\t<<[tau - Nabla^i nu_i]^2>>_Omega =  {col.Fore.CYAN}{msh.difference_in_bulk( fsp.tau, project( geo.g_c(fsp.omega)[i, j] * geo.Nabla_f(nu_output, omega_output)[i, j], fsp.Q_z ) ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )


'''
print( "Comparison with exact solution: " )
print( f"1)\t\t<<(z - z_exact)^2>>_Omega = {col.Fore.BLUE}{msh.difference_in_bulk( z_output, fsp.z_exact ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
print(
    f"2)\t\t<<|omega - omega_exact|^2>>_Omega = {col.Fore.BLUE}{msh.difference_in_bulk( project( sqrt( (omega_output[i] - fsp.omega_exact[i]) * (omega_output[i] - fsp.omega_exact[i]) ), fsp.Q_z ), project( Constant( 0 ), fsp.Q_z ) ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
print( f"3)\t\t<<(mu - mu_exact)^2>>_Omega = {col.Fore.BLUE}{msh.difference_in_bulk( mu_output, fsp.mu_exact ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
print(
    f"4)\t\t<<|nu - nu_exact|^2>>_Omega = {col.Fore.BLUE}{msh.difference_in_bulk( project( sqrt( (nu_output[i] - fsp.nu_exact[i]) * (nu_output[i] - fsp.nu_exact[i]) ), fsp.Q_z ), project( Constant( 0 ), fsp.Q_z ) ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
print( f"5)\t\t<<(tau - tau_exact)^2>>_Omega = {col.Fore.BLUE}{msh.difference_in_bulk( fsp.tau, fsp.tau_exact ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
'''

print( "Check if the PDE is satisfied:" )
print(
    f"\t\t<<(fel + flaplace)^2>>_Omega =  {col.Fore.GREEN}{msh.difference_in_bulk( project( phys.fel_n( omega_output, mu_output, fsp.tau, vp.kappa ), fsp.Q_z ), project( -phys.flaplace( fsp.sigma, omega_output ), fsp.Q_z ) ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )


xdmffile_check.write( project( z_output - fsp.z_exact, fsp.Q_z ), 0 )
xdmffile_check.write( project( sqrt( (omega_output[i] - fsp.omega_exact[i]) * (omega_output[i] - fsp.omega_exact[i]) ), fsp.Q_z ), 0 )
xdmffile_check.write( project( mu_output - fsp.mu_exact, fsp.Q_z ), 0 )
xdmffile_check.write( project( sqrt( (nu_output[i] - fsp.nu_exact[i]) * (nu_output[i] - fsp.nu_exact[i]) ), fsp.Q_z ), 0 )
xdmffile_check.write( project( fsp.tau - fsp.tau_exact, fsp.Q_tau ), 0 )
