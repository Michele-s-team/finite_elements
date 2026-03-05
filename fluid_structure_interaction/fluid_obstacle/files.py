from fenics import *
import os

import runtime_arguments as rarg

xdmffile_u_n_di = XDMFFile(os.path.join(rarg.args.output_directory, "u_n_di.xdmf"))
xdmffile_u_n_di_dot = XDMFFile(os.path.join(rarg.args.output_directory, "u_n_di_dot.xdmf"))

xdmffile_v_sq_n = XDMFFile(os.path.join(rarg.args.output_directory, "v_sq_n.xdmf"))
xdmffile_v_sq__ = XDMFFile(os.path.join(rarg.args.output_directory, "v_sq__.xdmf"))
xdmffile_sigma_sq = XDMFFile(os.path.join(rarg.args.output_directory, "sigma_sq_n_12.xdmf"))
xdmffile_phi_sq = XDMFFile(os.path.join(rarg.args.output_directory, "phi_sq.xdmf"))
