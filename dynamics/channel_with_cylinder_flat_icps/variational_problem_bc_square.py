from fenics import *
import importlib
import numpy as np
import ufl as ufl

import function_spaces as fsp
import geometry as geo
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices( 4 )
