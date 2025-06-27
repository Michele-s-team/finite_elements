from fenics import *
import importlib
import ufl as ufl

import function_spaces as fsp
import input_output as io
import load_mesh as lmsh
import runtime_arguments as rarg
import solution_paths as solpath
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

i, j, k, l = ufl.indices(4)

# xdmffile_check.write(project(fsp.hess_u[i, i], fsp.Q), 0)
# xdmffile_check.write(fsp.f, 0)
# xdmffile_check.write(project(fsp.hess_u[i, i] - fsp.f, fsp.Q), 0)
# xdmffile_check.close()

for i in range(len(rmsh.lmsh.sub_meshes)):
    io.full_print(fsp.u[i], f'u_{i}', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
                  solpath.nodal_values_path,
                  rmsh.lmsh.sub_meshes[i], 'scalar')
