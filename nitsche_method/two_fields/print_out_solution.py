from fenics import *
import importlib
import ufl_legacy as ufl

import function_spaces as fsp
import input_output as io
import load_mesh as lmsh
import solution_paths as solpath
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

i, j, k, l = ufl.indices(4)


# xdmffile_u = XDMFFile("solution/u.xdmf")
# xdmffile_u.write(fsp.u, 0)
# io.print_vector_to_csvfile(fsp.u, 'solution/u.csv')

io.full_print(fsp.u, 'u', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              lmsh.mesh, 'vector')
