from fenics import *
import ufl as ufl
import colorama as col

import boundary_geometry as bgeo
import function_spaces as fsp
import geometry as geo
import input_output as io
import mesh as msh
import physics as phys
import read_mesh_ring as rmsh
import runtime_arguments as rarg

import variational_problem_bc_ring as vp

i, j, k, l = ufl.indices( 4 )


xdmffile_check = XDMFFile( (rarg.args.output_directory) + "/check.xdmf" )
xdmffile_check.parameters.update( {"functions_share_mesh": True, "rewrite_function_mesh": False} )

# copy the data of the  solution psi into v_output, ..., z_output, which will be allocated or re-allocated here
v_output, w_output, sigma_output, z_output, omega_output, mu_output = fsp.psi.split( deepcopy=True )


# print( "\int_{\partial \Omega_r} (n^i \omega_i - psi )^2 dS = ", \
#    assemble( ( (((n_circle( omega_output ))[i] * omega_output[i] - omega_r) ) ** 2 * ds_r ) )
#   )

print(
    f"\t\t<<(n^i \omega_i - psi )^2>>_[partial Omega r] = {col.Fore.RED}{msh.difference_wrt_measure( (bgeo.n_circle( omega_output ))[i] * omega_output[i], vp.omega_r, rmsh.ds_r ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )


# print( "\int_{\partial \Omega_R} (n^i \omega_i - psi )^2 dS = ", \
#    assemble( ( (((n_circle( omega_output ))[i] * omega_output[i] - omega_R) ) ** 2 * ds_R ) )
#   )

print(
    f"\t\t<<(n^i \omega_i - psi )^2>>_[partial Omega R] = {col.Fore.RED}{msh.difference_wrt_measure( (bgeo.n_circle( omega_output ))[i] * omega_output[i], vp.omega_R, rmsh.ds_R ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )


# print( "\int_{\partial \Omega_R} (v^i n_i)^2 dS = ", \
#    assemble( ( (n_circle( omega_output )[i] * g( omega_output )[i, j] * v_output[j] - v_R) ** 2 * ds_R ) )
#   )

print( f"\t\t<<(v^i n_i - v_R)^2>>_[partial Omega R] = {col.Fore.RED}{msh.difference_wrt_measure( bgeo.n_circle( omega_output )[i] * geo.g( omega_output )[i, j] * v_output[j], vp.v_R, rmsh.ds_R ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )
