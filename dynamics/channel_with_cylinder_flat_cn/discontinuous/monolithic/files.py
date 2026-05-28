from fenics import *
import os 

import runtime_arguments as rarg

# Create XDMF files for visualization output
xdmffile_v = XDMFFile( os.path.join(rarg.args.output_directory, "v_n.xdmf") )
xdmffile_sigma = XDMFFile( os.path.join(rarg.args.output_directory, "sigma_n.xdmf") )
