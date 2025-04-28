from fenics import *
import ufl as ufl

import boundary_geometry as bgeo
import function_spaces as fsp
import input_output as io
import runtime_arguments as rarg
import solution_paths as solpath

# CHANGE VARIATIONAL PROBLEM OR MESH HERE
# import read_mesh_ring as rmsh
# import read_mesh_ring_slice as rmsh
import read_mesh_square_no_circle as rmsh
# import read_mesh_square as rmsh

# CHANGE VARIATIONAL PROBLEM OR MESH HERE
# import variational_problem_bc_ring as vp
# import variational_problem_bc_ring_slice as vp
import variational_problem_bc_square_no_circle as vp
# import variational_problem_bc_square as vp

i, j, k, l = ufl.indices(4)

xdmffile_check = XDMFFile(rarg.args.output_directory + "/check.xdmf")
xdmffile_check.parameters.update({"functions_share_mesh": True, "rewrite_function_mesh": False})

xdmffile_check.write(project(fsp.hess_u[i, i], fsp.Q), 0)
xdmffile_check.write(fsp.f, 0)
xdmffile_check.write(project(fsp.hess_u[i, i] - fsp.f, fsp.Q), 0)
xdmffile_check.close()

io.full_print(fsp.u, 'u', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              bgeo.mesh, 'scalar')
