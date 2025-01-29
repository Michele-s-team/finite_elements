from fenics import *
import ufl as ufl
import colorama as col
import numpy as np

import function_spaces as fsp
import geometry as geo
import mesh as msh
import physics as phys
import read_mesh_ring as rmsh
import runtime_arguments as rarg
#import variational_problem_bc_square_a as vp
import variational_problem_bc_ring as vp
#import variational_problem_bc_square_no_circle_a as vp

i, j, k, l = ufl.indices(4)


xdmffile_check = XDMFFile( (rarg.args.output_directory) + "/check.xdmf" )
xdmffile_check.parameters.update( {"functions_share_mesh": True, "rewrite_function_mesh": False} )


# copy the data of the  solution psi into v_output, ..., z_output, which will be allocated or re-allocated here
z_output, omega_output, mu_output, nu_output = fsp.psi.split( deepcopy=True )

print("Check of BCs:")
print( f"\t<<(z - phi)^2>>_[partial Omega r] = {col.Fore.RED}{msh.difference_wrt_measure(z_output, vp.z_r_const, rmsh.ds_r):4e}{col.Style.RESET_ALL}")
print( f"\t<<(z - phi)^2>>_[partial Omega R] = {col.Fore.RED}{msh.difference_wrt_measure(z_output, vp.z_R_const, rmsh.ds_R):4e}{col.Style.RESET_ALL}")
print( f"\t<<(n^i \omega_i - psi )^2>>_[partial Omega r] = {col.Fore.RED}{msh.difference_wrt_measure((rmsh.n_circle( omega_output ))[i] * omega_output[i], vp.omega_r, rmsh.ds_r):4e}{col.Style.RESET_ALL}")
print( f"\t<<(n^i \omega_i - psi )^2>>_[partial Omega R] = {col.Fore.RED}{msh.difference_wrt_measure((rmsh.n_circle( omega_output ))[i] * omega_output[i], vp.omega_R, rmsh.ds_R):4e}{col.Style.RESET_ALL}")


print("Check if the PDE is satisfied:")
# print(f"\t<<(fel + flaplace)^2>> = {col.Fore.GREEN}{np.sqrt(assemble( ( (  phys.fel_n( omega_output, mu_output, nu_output, vp.kappa ) + phys.flaplace( fsp.sigma, omega_output) ) ** 2 * rmsh.dx ) ) / assemble(Constant(1.0) * rmsh.dx))}{col.Style.RESET_ALL}")
print(f"\t<<(fel + flaplace)^2>>_Omega =  {col.Fore.GREEN}{msh.difference_in_bulk(project(phys.fel_n( omega_output, mu_output, nu_output, vp.kappa ), fsp.Q_z), project(-phys.flaplace( fsp.sigma, omega_output), fsp.Q_z)):4e}{col.Style.RESET_ALL}")




print( "Comparison with exact solution: " )
print( f"\t<<(z - z_exact)^2>>_Omega = {col.Fore.BLUE}{msh.difference_in_bulk( z_output, fsp.z_exact ):4e}{col.Style.RESET_ALL}" )
print(
    f"\t<<|omega - omega_exact|^2>>_Omega = {col.Fore.BLUE}{msh.difference_in_bulk( project( sqrt( (omega_output[i] - fsp.omega_exact[i]) * (omega_output[i] - fsp.omega_exact[i]) ), fsp.Q_z ), project( Constant( 0 ), fsp.Q_z ) ):4e}{col.Style.RESET_ALL}" )
print( f"\t<<(mu - mu_exact)^2>>_Omega = {col.Fore.BLUE}{msh.difference_in_bulk( mu_output, fsp.mu_exact ):4e}{col.Style.RESET_ALL}" )


xdmffile_check.write( project( z_output - fsp.z_exact , fsp.Q_z ), 0 )
xdmffile_check.write( project( mu_output - fsp.mu_exact , fsp.Q_z ), 0 )
# xdmffile_check.write( project( phys.fel_n( omega_output, mu_output, nu_output, vp.kappa ) + phys.flaplace( fsp.sigma, omega_output) , fsp.Q_sigma ), 0 )
# xdmffile_check.close()