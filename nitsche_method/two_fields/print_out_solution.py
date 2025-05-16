from fenics import *
import importlib
import ufl as ufl

import function_spaces as fsp
import input_output as io
import load_2d_mesh as lmsh
import runtime_arguments as rarg
import solution_paths as solpath
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

i, j, k, l = ufl.indices(4)

