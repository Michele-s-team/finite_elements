from fenics import *

import runtime_arguments as rarg

# Create XDMF files for visualization output
xdmffile_u_n = XDMFFile((rarg.args.output_directory) + "/u_n.xdmf")
xdmffile_u_dot_n = XDMFFile((rarg.args.output_directory) + "/u_dot_n.xdmf")

xdmffile_v_n = XDMFFile((rarg.args.output_directory) + "/v_n.xdmf")
xdmffile_v_ = XDMFFile((rarg.args.output_directory) + "/v_.xdmf")
xdmffile_sigma = XDMFFile((rarg.args.output_directory) + "/sigma_n_12.xdmf")
xdmffile_phi = XDMFFile((rarg.args.output_directory) + "/phi.xdmf")
