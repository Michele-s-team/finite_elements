from __future__ import print_function
from fenics import *
from mshr import *
from variational_problem_bc_a import *

# Create XDMF files for visualization output
xdmffile_v = XDMFFile( (args.output_directory) + '/v.xdmf' )
xdmffile_w = XDMFFile( (args.output_directory) + '/w.xdmf' )
xdmffile_sigma = XDMFFile( (args.output_directory) + '/sigma.xdmf' )
xdmffile_omega = XDMFFile( (args.output_directory) + '/omega.xdmf' )
xdmffile_z = XDMFFile( (args.output_directory) + '/z.xdmf' )

xdmffile_n = XDMFFile( (args.output_directory) + '/n.xdmf' )
xdmffile_n.write( facet_normal_smooth(), 0 )

# get the solution and write it to file
v_output, w_output, sigma_output, omega_output, z_output = psi.split( deepcopy=True )

# print solution to file
xdmffile_v.write( v_output, 0 )
xdmffile_w.write( w_output, 0 )
xdmffile_sigma.write( sigma_output, 0 )
xdmffile_omega.write( omega_output, 0 )
xdmffile_z.write( z_output, 0 )

# write the solutions in .h5 format so it can be read from other codes
HDF5File( MPI.comm_world, (args.output_directory) + "/h5/v.h5", "w" ).write( v_output, "/f" )
HDF5File( MPI.comm_world, (args.output_directory) + "/h5/w.h5", "w" ).write( w_output, "/f" )
HDF5File( MPI.comm_world, (args.output_directory) + "/h5/sigma.h5", "w" ).write( sigma_output, "/f" )
HDF5File( MPI.comm_world, (args.output_directory) + "/h5/omega.h5", "w" ).write( omega_output, "/f" )
HDF5File( MPI.comm_world, (args.output_directory) + "/h5/z.h5", "w" ).write( z_output, "/f" )


print( "\int_{\partial \Omega_W U \partial \Omega_O} (n_i v^i)^2 dS = ", \
       assemble( ((n_tb(omega))[i] * g( omega_output )[i, j] * v_output[j]) ** 2 * (ds_t + ds_b) ) \
       + assemble( ((n_circle(omega))[i] * g( omega_output )[i, j] * v_output[j]) ** 2 * ds_circle ) \
       )

print( "\int_{\partial \Omega} (n^i \omega_i - psi )^2 dS = ", \
       assemble( ( ((n_lr( omega_output ))[i] * omega_output[i] - omega_square)) ** 2 * (ds_l + ds_r) ) \
       + assemble( ( ((n_tb( omega_output ))[i] * omega_output[i] - omega_square)) ** 2 * (ds_t + ds_b) ) \
       + assemble( ( ((n_circle( omega_output ))[i] * omega_output[i] - omega_circle)) ** 2 * ds_circle ) \
       )

print( "\int_{\partial \Omega_ou} ( n_i d^{i 1})^2 dS =}", \
       assemble( (d_c( v_output, w_output, omega_output )[i, 0] * g( omega_output )[i, j] * (n_lr( omega_output ))[j]) ** 2 * ds_r ) \
    )