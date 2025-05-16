from fenics import *
import ufl as ufl

import function_spaces as fsp
import input_output as io
import load_2d_mesh as lmsh
import runtime_arguments as rarg
import solution_paths as solpath

i, j, k, l = ufl.indices(4)

xdmffile_check = XDMFFile( (args.output_directory) + "/check.xdmf" )
xdmffile_check.parameters.update( {"functions_share_mesh": True, "rewrite_function_mesh": False} )

xdmffile_u.write( u_output, 0 )
xdmffile_v.write( v_output, 0 )
xdmffile_w.write( w_output, 0 )

io.print_scalar_to_csvfile(u_output, (args.output_directory) + '/u.csv')
io.print_scalar_to_csvfile(v_output, (args.output_directory) + '/v.csv')
io.print_scalar_to_csvfile(w_output, (args.output_directory) + '/w.csv')




xdmffile_check.write( project( w_output - f , Q_w ), 0 )
xdmffile_check.close()

msh.bulk_points(mesh)
