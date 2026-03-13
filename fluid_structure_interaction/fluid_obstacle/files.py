from fenics import *
import os

import runtime_arguments as rarg

# 1) I 

xdmffile_U_n_12 = XDMFFile(os.path.join(rarg.args.output_directory, "U_n_12.xdmf"))

# 2) D

xdmffile_u_n_di = XDMFFile(os.path.join(rarg.args.output_directory, "u_n_di.xdmf"))
xdmffile_u_n_di_dot = XDMFFile(os.path.join(rarg.args.output_directory, "u_n_di_dot.xdmf"))

xdmffile_u_n_sq = XDMFFile(os.path.join(rarg.args.output_directory, "u_n_sq.xdmf"))
xdmffile_u_n_sq_dot = XDMFFile(os.path.join(rarg.args.output_directory, "u_n_sq_dot.xdmf"))


# 3) disk fluid

xdmffile_v_di_n = XDMFFile(os.path.join(rarg.args.output_directory, "v_di_n.xdmf"))
xdmffile_v_di__ = XDMFFile(os.path.join(rarg.args.output_directory, "v_di__.xdmf"))
xdmffile_sigma_di_n_12 = XDMFFile(os.path.join(rarg.args.output_directory, "sigma_di_n_12.xdmf"))
xdmffile_phi_di = XDMFFile(os.path.join(rarg.args.output_directory, "phi_di.xdmf"))

# 4) square fluid

xdmffile_v_sq_n = XDMFFile(os.path.join(rarg.args.output_directory, "v_sq_n.xdmf"))
xdmffile_v_sq__ = XDMFFile(os.path.join(rarg.args.output_directory, "v_sq__.xdmf"))
xdmffile_sigma_sq_n_12 = XDMFFile(os.path.join(rarg.args.output_directory, "sigma_sq_n_12.xdmf"))
xdmffile_phi_sq = XDMFFile(os.path.join(rarg.args.output_directory, "phi_sq.xdmf"))


# 5) M

xdmffile_c_n = XDMFFile(os.path.join(rarg.args.output_directory, "c_n.xdmf"))
