from fenics import *

import importlib
import input_output as io
import solution_paths as solpath
import switch_problem as swi

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)


io.full_print(fsp.u_0, 'u_0', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,  rmsh.sf[0])
io.full_print(fsp.u, 'u', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path,  rmsh.sf[0])
    