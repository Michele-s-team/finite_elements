from fenics import *
import ufl as ufl

import function_spaces as fsp
import input_output as io
import mesh.load as lmsh
import mesh.utils as msh
import runtime_arguments as rarg
import solution_paths as solpath

i, j, k, l = ufl.indices(4)

fsp.u_output, fsp.v_output = fsp.psi.split(deepcopy=True)



xdmffile_check = XDMFFile( (rarg.args.output_directory) + "/check.xdmf" )
xdmffile_check.parameters.update( {"functions_share_mesh": True, "rewrite_function_mesh": False} )


io.full_print(fsp.u_output, 'u', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path)
io.full_print(fsp.v_output, 'v', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path)
io.full_print(fsp.w, 'w', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path)



xdmffile_check.write( project( fsp.w - fsp.f , fsp.Q_w ), 0 )
xdmffile_check.close()

msh.bulk_points(lmsh.mesh)
