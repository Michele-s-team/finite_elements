from fenics import *

import runtime_arguments as rarg


# Create XDMF files for visualization output
xdmffile_v_bar = XDMFFile( (rarg.args.output_directory) + '/v_bar.xdmf' )
xdmffile_w_bar = XDMFFile( (rarg.args.output_directory) + '/w_bar.xdmf' )
xdmffile_v = XDMFFile( (rarg.args.output_directory) + '/v_n.xdmf' )
xdmffile_w = XDMFFile( (rarg.args.output_directory) + '/w_n.xdmf' )
xdmffile_phi = XDMFFile( (rarg.args.output_directory) + '/phi.xdmf' )
xdmffile_sigma = XDMFFile( (rarg.args.output_directory) + '/sigma_n_12.xdmf' )
xdmffile_X_n_12 = XDMFFile( (rarg.args.output_directory) + '/X_n_12.xdmf' )
xdmffile_nu_n_12 = XDMFFile( (rarg.args.output_directory) + '/nu_n_12.xdmf' )
xdmffile_psi_n_12 = XDMFFile( (rarg.args.output_directory) + '/psi_n_12.xdmf' )
xdmffile_mu_n_12 = XDMFFile( (rarg.args.output_directory) + '/mu_n_12.xdmf' )