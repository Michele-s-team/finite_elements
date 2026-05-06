from fenics import *
import os

import runtime_arguments as rarg

# Create XDMF files for visualization output
xdmffile_v_n = XDMFFile(os.path.join(rarg.args.output_directory, "v_n.xdmf"))
xdmffile_v_ = XDMFFile(os.path.join(rarg.args.output_directory, "v_.xdmf"))

xdmffile_sigma_n_12 = XDMFFile(os.path.join(rarg.args.output_directory, "sigma_n_12.xdmf"))
xdmffile_phi = XDMFFile(os.path.join(rarg.args.output_directory, "phi.xdmf"))

xdmffile_u_n = XDMFFile(os.path.join(rarg.args.output_directory, "u_n.xdmf"))
xdmffile_u_dot_n = XDMFFile(os.path.join(rarg.args.output_directory, "u_dot_n.xdmf"))
