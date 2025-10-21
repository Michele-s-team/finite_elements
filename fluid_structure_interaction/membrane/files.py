from fenics import *

import runtime_arguments as rarg

# Create XDMF files for visualization output
# 1) membrane problem
xdmffile_v_bar = XDMFFile( (rarg.args.output_directory) + '/v_bar.xdmf' )
xdmffile_w_bar = XDMFFile( (rarg.args.output_directory) + '/w_bar.xdmf' )
xdmffile_v_n = XDMFFile( (rarg.args.output_directory) + '/v_n.xdmf' )
xdmffile_w_n = XDMFFile( (rarg.args.output_directory) + '/w_n.xdmf' )
xdmffile_phi = XDMFFile( (rarg.args.output_directory) + '/phi.xdmf' )
xdmffile_sigma_n_12 = XDMFFile( (rarg.args.output_directory) + '/sigma_n_12.xdmf' )
xdmffile_u_n_12 = XDMFFile( (rarg.args.output_directory) + '/X_n_12.xdmf' )
xdmffile_nu_n_12 = XDMFFile( (rarg.args.output_directory) + '/nu_n_12.xdmf' )
xdmffile_psi_n_12 = XDMFFile( (rarg.args.output_directory) + '/psi_n_12.xdmf' )
xdmffile_mu_n_12 = XDMFFile( (rarg.args.output_directory) + '/mu_n_12.xdmf' )

# 2) mesh problem
xdmffile_u_n = XDMFFile((rarg.args.output_directory) + "/u_n.xdmf")
xdmffile_u_dot_n = XDMFFile((rarg.args.output_directory) + "/u_dot_n.xdmf")

# 3) fluid problem 
xdmffile_v_fl_n = XDMFFile((rarg.args.output_directory) + "/v_fl_n.xdmf")
xdmffile_v_fl_bar = XDMFFile((rarg.args.output_directory) + "/v_var_fl.xdmf")
xdmffile_sigma_fl = XDMFFile((rarg.args.output_directory) + "/sigma_fl_n_12.xdmf")
xdmffile_phi_fl = XDMFFile((rarg.args.output_directory) + "/phi_fl.xdmf")
