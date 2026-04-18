from fenics import *

import runtime_arguments as rarg

# Create XDMF files for visualization output
xdmffile_v = XDMFFile( (rarg.args.output_directory) + "/v_n.xdmf" )
xdmffile_v_ = XDMFFile( (rarg.args.output_directory) + "/v_.xdmf" )
xdmffile_sigma = XDMFFile( (rarg.args.output_directory) + "/sigma_n_12.xdmf" )
xdmffile_phi = XDMFFile( (rarg.args.output_directory) + "/phi.xdmf" )

xdmffile_z = XDMFFile( (rarg.args.output_directory) + "/z.xdmf" )
xdmffile_omega = XDMFFile( (rarg.args.output_directory) + "/omega.xdmf" )
