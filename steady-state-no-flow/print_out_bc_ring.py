from fenics import *
import ufl as ufl
import termcolor

import function_spaces as fsp
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
print( "\t<<(z - phi)^2>>_r = ", \
   sqrt(assemble( ( (z_output - vp.z_r_const ) ** 2 * rmsh.ds_r ) ) / assemble(Constant(1.0) * rmsh.ds_r))
  )

# msh.boundary_points_circle(rmsh.mesh, rmsh.r/2, (rmsh.r + rmsh.R)/2, [0, 0])
print(f"\t<<(z - phi)^2>>_r = {termcolor.colored(msh.difference_on_boundary_circle(z_output, vp.z_r, rmsh.r/2.0, (rmsh.r + rmsh.R)/2.0, rmsh.c_r), 'red')}")


print( "\t<<(z - phi)^2>>_R = ", \
   sqrt(assemble( ( (z_output - vp.z_R_const ) ** 2 * rmsh.ds_R ) ) / assemble(Constant(1.0) * rmsh.ds_R))
  )
print( "\t<<(n^i \omega_i - psi )^2>>_r = ", \
   sqrt(assemble( ( ((rmsh.n_circle( omega_output ))[i] * omega_output[i] - vp.omega_r ) ** 2 * rmsh.ds_r ) ) / assemble(Constant(1.0) * rmsh.ds_r))
  )
print( "\t<<(n^i \omega_i - psi )^2>>_R = ", \
   sqrt(assemble( ( ((rmsh.n_circle( omega_output ))[i] * omega_output[i] - vp.omega_R ) ** 2 * rmsh.ds_R ) ) / assemble( Constant(1.0) * rmsh.ds_R))
  )

print("Check if the PDE is satisfied:")
print( "\t<<(fel + flaplace)^2>> = ", \
   sqrt(assemble( ( (  phys.fel_n( omega_output, mu_output, nu_output, vp.kappa ) + phys.flaplace( fsp.sigma, omega_output) ) ** 2 * rmsh.dx ) ) / assemble(Constant(1.0) * rmsh.dx))
  )

xdmffile_check.write( project( phys.fel_n( omega_output, mu_output, nu_output, vp.kappa ) + phys.flaplace( fsp.sigma, omega_output) , fsp.Q_sigma ), 0 )
xdmffile_check.close()
