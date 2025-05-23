from fenics import *

import runtime_arguments as rarg

# Create XDMF files for visualization output
xdmffile_u_bar = XDMFFile(rarg.args.output_directory + '/v_bar.xdmf')
xdmffile_u_n = XDMFFile(rarg.args.output_directory + '/v_n.xdmf')
xdmffile_p_n = XDMFFile(rarg.args.output_directory + '/p_n.xdmf')
