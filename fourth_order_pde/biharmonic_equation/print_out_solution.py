from fenics import *
import ufl as ufl

import function_spaces as fsp
import input_output as io
import load_2d_mesh as lmsh
import mesh as msh
import runtime_arguments as rarg
import solution_paths as solpath

i, j, k, l = ufl.indices(4)

fsp.u_output, fsp.v_output, fsp.w_output = fsp.psi.split(deepcopy=True)


# xdmffile_u = XDMFFile( (rarg.args.output_directory) + "/u.xdmf" )
# xdmffile_v = XDMFFile( (rarg.args.output_directory) + "/v.xdmf" )
# xdmffile_w = XDMFFile( (rarg.args.output_directory) + "/w.xdmf" )


xdmffile_check = XDMFFile( (rarg.args.output_directory) + "/check.xdmf" )
xdmffile_check.parameters.update( {"functions_share_mesh": True, "rewrite_function_mesh": False} )

# xdmffile_u.write( fsp.u_output, 0 )
# xdmffile_v.write( fsp.v_output, 0 )
# xdmffile_w.write( fsp.w_output, 0 )

# io.print_scalar_to_csvfile(fsp.u_output, (rarg.args.output_directory) + '/u.csv')
# io.print_scalar_to_csvfile(fsp.v_output, (rarg.args.output_directory) + '/v.csv')
# io.print_scalar_to_csvfile(fsp.w_output, (rarg.args.output_directory) + '/w.csv')

io.full_print(fsp.u_output, 'u', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              lmsh.mesh, 'scalar')
io.full_print(fsp.v_output, 'v', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              lmsh.mesh, 'scalar')
io.full_print(fsp.w_output, 'w', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              lmsh.mesh, 'scalar')



xdmffile_check.write( project( fsp.w_output - fsp.f , fsp.Q_w ), 0 )
xdmffile_check.close()

msh.bulk_points(lmsh.mesh)
