from fenics import *
import ufl as ufl

import boundary_geometry as bgeo
import function_spaces as fsp
import geometry as geo
import input_output as io
import physics as phys
import solution_paths as solpath
import runtime_arguments as rarg

# CHANGE VARIATIONAL PROBLEM OR MESH HERE
# import read_mesh_ring as rmsh
import read_mesh_square as rmsh
# import read_mesh_square as rmsh

# CHANGE VARIATIONAL PROBLEM OR MESH HERE
# import variational_problem_bc_ring as vp
import variational_problem_bc_square as vp
# import variational_problem_bc_square_a as vp
# import variational_problem_bc_square_b as vp

i, j, k, l = ufl.indices(4)


xdmffile_u = XDMFFile((args.output_directory) + "/u.xdmf")

xdmffile_check = XDMFFile((args.output_directory) + "/check.xdmf")
xdmffile_check.parameters.update({"functions_share_mesh": True, "rewrite_function_mesh": False})


xdmffile_u.write(u, 0)
xdmffile_check.write(project(hess_u[i, i], Q), 0)
xdmffile_check.write(f, 0)
xdmffile_check.write(project(hess_u[i, i] - f, Q), 0)
xdmffile_check.close()

io.print_scalar_to_csvfile(u, (args.output_directory) + "/u.csv");
