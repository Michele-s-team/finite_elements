from fenics import *

import runtime_arguments as rarg

# Create XDMF files for visualization output
xdmffile_u = XDMFFile( (rarg.args.output_directory) + '/u.xdmf' )
xdmffile_v = XDMFFile( (rarg.args.output_directory) + '/v.xdmf' )

