import csv
import importlib
from fenics import *
import os
import ufl as ufl

import boundary_geometry as bgeo
import function_spaces as fsp
import geometry as geo
import runtime_arguments as rarg
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)


i, j, k, l = ufl.indices( 4 )
