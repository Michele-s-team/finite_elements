from fenics import *
import importlib
import ufl as ufl

import function_spaces as fsp
import input_output as io
import solution_paths as solpath
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)

for i in range(len(rmsh.lmsh.sub_meshes)):
    io.full_print(fsp.u[i], f'u_{i}', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
                  solpath.nodal_values_path)
