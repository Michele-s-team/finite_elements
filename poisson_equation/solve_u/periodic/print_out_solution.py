import sys

module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import switch_problem as swi
import solution_paths as solpath
import runtime_arguments as rarg
import mesh.load as lmsh
import input_output as sys_io
from fenics import *
import importlib
import ufl as ufl

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

i, j, k, l = ufl.indices(4)

xdmffile_check = XDMFFile(rarg.args.output_directory + "/check.xdmf")
xdmffile_check.parameters.update(
    {"functions_share_mesh": True, "rewrite_function_mesh": False})

xdmffile_check.write(project(fsp.hess_u[i, i], fsp.Q), 0)
xdmffile_check.write(fsp.f, 0)
xdmffile_check.write(project(fsp.hess_u[i, i] - fsp.f, fsp.Q), 0)
xdmffile_check.close()

sys_io.full_print(fsp.u, 'u', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path)
