from fenics import *
import os

import runtime_arguments as rarg

# Create XDMF files for visualization output
xdmffile_v_n = XDMFFile(os.path.join(rarg.args.output_directory, "v_n.xdmf"))
xdmffile_sigma_n = XDMFFile(os.path.join(rarg.args.output_directory, "sigma_n.xdmf"))

xdmffile_u_n = XDMFFile(os.path.join(rarg.args.output_directory, "u_n.xdmf"))
xdmffile_u_dot_n = XDMFFile(os.path.join(rarg.args.output_directory, "u_dot_n.xdmf"))

xdmffile_det_F_n = XDMFFile(os.path.join(rarg.args.output_directory, "det_F_n.xdmf"))
xdmffile_u_0 = XDMFFile(os.path.join(rarg.args.output_directory, "u_0.xdmf"))

