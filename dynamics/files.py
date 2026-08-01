from fenics import *

import csv
import os
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



# 4 data file
filepath_data = os.path.join(rarg.args.output_directory, 'data.csv')
os.makedirs(os.path.dirname(filepath_data), exist_ok=True)

csvfile_data = open(filepath_data, 'a', newline='')
fieldnames_data = [ \
    'step',
    '<(n^{n-1/2}_i \overline{v}^i)^2>^{n-1/2}_{partial Omega O}',
    '<(nabla_i v^i - 2 w H)^2>^{n-1/2}_{Omega}',
    '<(nabla_i v^i)^2>^{n-1/2}_{Omega}',
    '<(2 w H)^2>^{n-1/2}_{Omega}',
    '< 1 >^{n-1/2}_{Omega}',
    'dTdt_L',
    'dTdt_R',
    '(dTdt_L - dTdt_R)/|(dTdt_L + dTdt_R)/2.0|',
    ]
writer_data = csv.DictWriter(csvfile_data, fieldnames=fieldnames_data)
writer_data.writeheader()


