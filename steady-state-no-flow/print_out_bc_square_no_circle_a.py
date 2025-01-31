from fenics import *
import ufl as ufl
import colorama as col
import numpy as np

import function_spaces as fsp
import geometry as geo
import mesh as msh
import physics as phys
import read_mesh_square_no_circle as rmsh
import runtime_arguments as rarg
import variational_problem_bc_square_no_circle_a as vp


i, j, k, l = ufl.indices( 4 )


xdmffile_check = XDMFFile( (rarg.args.output_directory) + "/check.xdmf" )
xdmffile_check.parameters.update( {"functions_share_mesh": True, "rewrite_function_mesh": False} )

# copy the data of the  solution psi into v_output, ..., z_output, which will be allocated or re-allocated here
z_output, omega_output, mu_output, nu_output, tau_output = fsp.psi.split( deepcopy=True )

# print( "\int_{\partial \Omega} (n^i \omega_i - psi )^2 dS = ", \
#        assemble( ( ((n_lr( omega_output ))[i] * omega_output[i] - omega_l)) ** 2 * (ds_l + ds_r) ) \
#        + assemble( ( ((n_tb( omega_output ))[i] * omega_output[i] - omega_l)) ** 2 * (ds_t + ds_b) ) \
#        )

xdmffile_check.write( project( z_output - fsp.z_exact, fsp.Q_z ), 0 )
xdmffile_check.write( project( sqrt( (omega_output[i] - fsp.omega_exact[i]) * (omega_output[i] - fsp.omega_exact[i]) ), fsp.Q_z ), 0 )
xdmffile_check.write( project( mu_output - fsp.mu_exact, fsp.Q_z ), 0 )
xdmffile_check.write( project( sqrt( (nu_output[i] - fsp.nu_exact[i]) * (nu_output[i] - fsp.nu_exact[i]) ), fsp.Q_z ), 0 )
xdmffile_check.write( project( tau_output - fsp.tau_exact, fsp.Q_tau ), 0 )