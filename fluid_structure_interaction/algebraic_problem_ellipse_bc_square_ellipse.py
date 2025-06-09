'''
this module solves for the variables theta, omega whic define the state of the ellipse
'''

from fenics import *
import importlib
import ufl as ufl


import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices(4)


