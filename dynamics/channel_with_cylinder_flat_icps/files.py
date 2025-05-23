from fenics import *

import runtime_arguments as rarg

# Create XDMF files for visualization output
xdmffile_u = XDMFFile(rarg.args.output_directory + '/v_n.xdmf')
xdmffile_p = XDMFFile(rarg.args.output_directory + '/p_n.xdmf')
