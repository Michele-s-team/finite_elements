from fenics import *
import importlib
import ufl as ufl

import input_output as io
import mesh.load as lmsh
import solution_paths as solpath
import switch_problem as swi

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)

vp = importlib.import_module(swi.vp)

i, j, k, l = ufl.indices(4)

u_output, v_output = fsp.psi.split(deepcopy=True)


io.full_print(u_output, 'u', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              'scalar')

io.full_print(v_output, 'v', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              'scalar')