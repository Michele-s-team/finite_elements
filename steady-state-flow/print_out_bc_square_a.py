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

import variational_problem_bc_square_a as vp

i, j, k, l = ufl.indices( 4 )


xdmffile_check = XDMFFile( (rarg.args.output_directory) + "/check.xdmf" )
xdmffile_check.parameters.update( {"functions_share_mesh": True, "rewrite_function_mesh": False} )

# copy the data of the  solution psi into v_output, ..., z_output, which will be allocated or re-allocated here
v_output, w_output, sigma_output, z_output, omega_output, mu_output = fsp.psi.split( deepcopy=True )

print( "Check of BCs:" )

print( f"\t\t<<|v^i - v_r^i|^2>>_[partial Omega r] = {col.Fore.RED}{msh.difference_wrt_measure( (v_output[i] - vp.v_r[i]) * (v_output[i] - vp.v_r[i]), Constant(0), rmsh.ds_r ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
print( f"\t\t<<(v^i n_i - v_R)^2>>_[partial Omega R] = {col.Fore.RED}{msh.difference_wrt_measure( bgeo.n_circle( omega_output )[i] * geo.g( omega_output )[i, j] * v_output[j], vp.v_R_const, rmsh.ds_R ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )

print( f"\t\t<<(w - w_r)^2>>_[partial Omega r] = {col.Fore.RED}{msh.difference_wrt_measure( w_output, vp.w_r_const, rmsh.ds_r ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
print( f"\t\t<<(w - w_R)^2>>_[partial Omega R] = {col.Fore.RED}{msh.difference_wrt_measure( w_output, vp.w_R_const, rmsh.ds_R ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )

print( f"\t\t<<(sigma - sigma_r)^2>>_[partial Omega r] = {col.Fore.RED}{msh.difference_wrt_measure( sigma_output, vp.sigma_r_const, rmsh.ds_r ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )


print( f"\t\t<<(z - phi)^2>>_[partial Omega r] = {col.Fore.RED}{msh.difference_wrt_measure( z_output, vp.z_r, rmsh.ds_r ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
print( f"\t\t<<(z - phi)^2>>_[partial Omega R] = {col.Fore.RED}{msh.difference_wrt_measure( z_output, vp.z_R, rmsh.ds_R ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )

print(
    f"\t\t<<(n^i \omega_i - psi )^2>>_[partial Omega r] = {col.Fore.RED}{msh.difference_wrt_measure( (bgeo.n_circle( omega_output ))[i] * omega_output[i], vp.omega_r, rmsh.ds_r ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
print(
    f"\t\t<<(n^i \omega_i - psi )^2>>_[partial Omega R] = {col.Fore.RED}{msh.difference_wrt_measure( (bgeo.n_circle( omega_output ))[i] * omega_output[i], vp.omega_R, rmsh.ds_R ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )


xdmffile_check.write( project( phys.fvisc_n(v_output, w_output, omega_output, mu_output, vp.eta)  + phys.fel_n( omega_output, mu_output, fsp.tau, vp.kappa ) + phys.flaplace( fsp.sigma, omega_output ), fsp.Q_z ), 0 )
xdmffile_check.write( project( project( sqrt( (omega_output[i] - (z_output.dx( i ))) * (omega_output[i] - (z_output.dx( i ))) ), fsp.Q_z ), fsp.Q_z ), 0 )
xdmffile_check.write( project( project( mu_output - geo.H( omega_output ), fsp.Q_z ), fsp.Q_z ), 0 )
xdmffile_check.write( project( project( sqrt( (fsp.nu[i] - (mu_output.dx( i ))) * (fsp.nu[i] - (mu_output.dx( i ))) ), fsp.Q_z ), fsp.Q_z ), 0 )
xdmffile_check.write( project( project( fsp.tau - geo.g_c(omega_output)[i, j] * geo.Nabla_f(fsp.nu, omega_output)[i, j], fsp.Q_z ), fsp.Q_tau ), 0 )
