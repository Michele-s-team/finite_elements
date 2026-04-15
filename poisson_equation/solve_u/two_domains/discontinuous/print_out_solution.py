from fenics import *
import importlib
import ufl as ufl

import input_output as io
import solution_paths as solpath
import switch_problem as swi

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)

vp = importlib.import_module(swi.vp)

# io.full_print(fsp.u, 'u', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
            #   solpath.nodal_values_path)