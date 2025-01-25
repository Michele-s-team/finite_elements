from __future__ import print_function
from fenics import *
from mshr import *
import physics as phys
# from variational_problem_bc_a import *
from variational_problem_bc_ring import *

xdmffile_check = XDMFFile( (rmsh.args.output_directory) + "/check.xdmf" )
xdmffile_check.parameters.update( {"functions_share_mesh": True, "rewrite_function_mesh": False} )


# copy the data of the  solution psi into v_output, ..., z_output, which will be allocated or re-allocated here
z_output, omega_output, mu_output, nu_output = fsp.psi.split( deepcopy=True )

print("Check of BCs:")
print( "\t<<(z - phi)^2>>_r = ", \
   sqrt(assemble( ( (z_output - z_r_const ) ** 2 * rmsh.ds_r ) ) / assemble(Constant(1.0) * rmsh.ds_r))
  )
print( "\t<<(z - phi)^2>>_R = ", \
   sqrt(assemble( ( (z_output - z_R_const ) ** 2 * rmsh.ds_R ) ) / assemble(Constant(1.0) * rmsh.ds_R))
  )
print( "\t<<(n^i \omega_i - psi )^2>>_r = ", \
   sqrt(assemble( ( ((n_circle( omega_output ))[i] * omega_output[i] - omega_r ) ** 2 * rmsh.ds_r ) ) / assemble(Constant(1.0) * rmsh.ds_r))
  )
print( "\t<<(n^i \omega_i - psi )^2>>_R = ", \
   sqrt(assemble( ( ((n_circle( omega_output ))[i] * omega_output[i] - omega_R ) ** 2 * rmsh.ds_R ) ) / assemble( Constant(1.0) * rmsh.ds_R))
  )

print("Check if the PDE is satisfied:")
print( "\t<<(fel + flaplace)^2>> = ", \
   sqrt(assemble( ( (  phys.fel_n( omega_output, mu_output, nu_output, kappa ) + flaplace( fsp.sigma, omega_output) ) ** 2 * dx ) ) / assemble(Constant(1.0) * dx))
  )

xdmffile_check.write( project( phys.fel_n( omega_output, mu_output, nu_output, kappa ) + flaplace( fsp.sigma, omega_output) , fsp.Q_sigma ), 0 )
xdmffile_check.close()
