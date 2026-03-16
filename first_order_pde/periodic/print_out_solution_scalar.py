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

sys_io.full_print(fsp.u, 'u', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              'scalar')

sys_io.full_print(fsp.grad_u, 'grad_u', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path,
              'vector')
