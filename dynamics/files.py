from fenics import *

import runtime_arguments as rarg


# Create XDMF files for visualization output
xdmffile_v_bar = XDMFFile( (rarg.args.output_directory) + '/v_bar.xdmf' )
xdmffile_w_bar = XDMFFile( (rarg.args.output_directory) + '/w_bar.xdmf' )
xdmffile_v = XDMFFile( (rarg.args.output_directory) + '/v_n.xdmf' )
xdmffile_w = XDMFFile( (rarg.args.output_directory) + '/w_n.xdmf' )
xdmffile_phi = XDMFFile( (rarg.args.output_directory) + '/phi.xdmf' )
xdmffile_sigma = XDMFFile( (rarg.args.output_directory) + '/sigma_n_12.xdmf' )
xdmffile_z = XDMFFile( (rarg.args.output_directory) + '/z_n_12.xdmf' )
xdmffile_omega = XDMFFile( (rarg.args.output_directory) + '/omega_n_12.xdmf' )
xdmffile_mu = XDMFFile( (rarg.args.output_directory) + '/mu_n_12.xdmf' )

xdmffile_nu = XDMFFile( (rarg.args.output_directory) + '/nu_n_12.xdmf' )
xdmffile_tau = XDMFFile( (rarg.args.output_directory) + '/tau_n_12.xdmf' )
xdmffile_d = XDMFFile( (rarg.args.output_directory) + '/d_n.xdmf' )


xdmffile_f = XDMFFile( (rarg.args.output_directory) + '/f.xdmf' )
xdmffile_f.parameters.update( {"functions_share_mesh": True, "rewrite_function_mesh": False} )

# xdmffile_d = XDMFFile( (rarg.args.output_directory) + '/d.xdmf' )
# xdmffile_d.parameters.update( {"functions_share_mesh": True, "rewrite_function_mesh": False} )

xdmffile_dFdl = XDMFFile( (rarg.args.output_directory) + '/dFdl.xdmf' )
xdmffile_dFdl.parameters.update( {"functions_share_mesh": True, "rewrite_function_mesh": False} )

xdmffile_dFds = XDMFFile( (rarg.args.output_directory) + '/dFds.xdmf' )
xdmffile_dFds.parameters.update( {"functions_share_mesh": True, "rewrite_function_mesh": False} )

