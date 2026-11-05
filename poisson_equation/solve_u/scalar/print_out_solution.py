from fenics import *
import importlib
import ufl as ufl

import function_spaces as fsp
import input_output as io
import runtime_arguments as rarg
import solution_paths as solpath
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

vp = importlib.import_module(swi.vp)


u_output, s_output = fsp.psi.split( deepcopy=True )

io.full_print(u_output, 'u', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path)

io.full_print(s_output, 's', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path)