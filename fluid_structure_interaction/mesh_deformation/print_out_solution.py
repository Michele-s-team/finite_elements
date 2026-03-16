from fenics import *
import importlib
import ufl as ufl

import function_spaces as fsp
import input_output as io
import mesh.load as lmsh
import mesh.utils as msh
import runtime_arguments as rarg
import solution_paths as solpath
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

i, j, k, l = ufl.indices(4)

xdmffile_check = XDMFFile(rarg.args.output_directory + "/check.xdmf")
xdmffile_check.parameters.update({"functions_share_mesh": True, "rewrite_function_mesh": False})

io.full_print(fsp.u, 'u', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path)
io.full_print(fsp.u_dot, 'u_dot', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path)

# Write the deformed mesh to XDMF
deformed_mesh = msh.deform_mesh(lmsh.mesh, fsp.u)
with XDMFFile(rarg.args.output_directory + "/deformed_mesh.xdmf") as xdmf:
    xdmf.write(deformed_mesh)
