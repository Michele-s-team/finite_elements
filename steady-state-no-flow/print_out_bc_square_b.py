from fenics import *
import ufl as ufl
import colorama as col

import boundary_geometry as bgeo
import function_spaces as fsp
import geometry as geo
import input_output as io
import mesh as msh
import physics as phys
import read_mesh_square as rmsh
import runtime_arguments as rarg
import variational_problem_bc_square_b as vp

i, j, k, l = ufl.indices( 4 )


xdmffile_check = XDMFFile( (rarg.args.output_directory) + "/check.xdmf" )
xdmffile_check.parameters.update( {"functions_share_mesh": True, "rewrite_function_mesh": False} )

# copy the data of the  solution psi into v_output, ..., z_output, which will be allocated or re-allocated here
z_output, omega_output, mu_output = fsp.psi.split( deepcopy=True )

print( "Check of BCs:" )
print("1)")
print( f"\t\t<<(z - phi)^2>>_square = {col.Fore.RED}{msh.difference_wrt_measure( z_output, vp.z_square_const, rmsh.ds_square ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
print( f"\t\t<<(z - phi)^2>>_circle = {col.Fore.RED}{msh.difference_wrt_measure( z_output, vp.z_circle_const, rmsh.ds_circle ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
print("2)")
print(
    f"\t\t<<(n^i \omega_i - psi )^2>>_lr = {col.Fore.RED}{msh.difference_wrt_measure( (bgeo.n_lr( omega_output ))[i] * omega_output[i], vp.n_omega_square, rmsh.ds_lr ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
print(
    f"\t\t<<(n^i \omega_i - psi )^2>>_tb = {col.Fore.RED}{msh.difference_wrt_measure( (bgeo.n_tb( omega_output ))[i] * omega_output[i], vp.n_omega_square, rmsh.ds_tb ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
print(
    f"\t\t<<(n^i \omega_i - psi )^2>>_circle = {col.Fore.RED}{msh.difference_wrt_measure( (bgeo.n_circle( omega_output ))[i] * omega_output[i], vp.omega_circle, rmsh.ds_circle ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )

'''
print( "Check if the intermediate PDEs are satisfied:" )
print(
    f"1)\t\t<<(fel + flaplace)^2>>_Omega =  {msh.difference_in_bulk( project( phys.fel_n( omega_output, mu_output, fsp.tau, vp.kappa ), fsp.Q_z ), project( -phys.flaplace( fsp.sigma, omega_output ), fsp.Q_z ) ):.{io.number_of_decimals}e}" )
print(
    f"2) \t\t<<|omega - partial z|^2>>_Omega = {msh.difference_in_bulk( project( sqrt( (omega_output[i] - z_output.dx( i )) * (omega_output[i] - z_output.dx( i )) ), fsp.Q_z ), project( Constant( 0 ), fsp.Q_z ) ):.{io.number_of_decimals}e}" )
print( f"3)\t\t<<[mu - H(omega)]^2>>_Omega =  {msh.difference_in_bulk( mu_output, project( geo.H( omega_output ), fsp.Q_z ) ):.{io.number_of_decimals}e}" )
print( f"4)\t\t<<[tau - Nabla^i nu_i]^2>>_Omega =  {msh.difference_in_bulk( fsp.tau, project( - geo.Nabla_LB(mu_output,omega_output), fsp.Q_z ) ):.{io.number_of_decimals}e}" )
'''

xdmffile_check.write( project( project( phys.fel_n( omega_output, mu_output, fsp.tau, vp.kappa ) + phys.flaplace( fsp.sigma, omega_output ), fsp.Q_z ), fsp.Q_z ), 0 )
xdmffile_check.write( project( project( sqrt( (omega_output[i] - (z_output.dx( i ))) * (omega_output[i] - (z_output.dx( i ))) ), fsp.Q_z ), fsp.Q_z ), 0 )
xdmffile_check.write( project( project( mu_output - geo.H( omega_output ), fsp.Q_z ), fsp.Q_z ), 0 )
xdmffile_check.write( project( project( sqrt( (fsp.nu[i] - (mu_output.dx( i ))) * (fsp.nu[i] - (mu_output.dx( i ))) ), fsp.Q_z ), fsp.Q_z ), 0 )
xdmffile_check.write( project( project( fsp.tau - geo.g_c(omega_output)[i, j] * geo.Nabla_f(fsp.nu, omega_output)[i, j], fsp.Q_z ), fsp.Q_tau ), 0 )
