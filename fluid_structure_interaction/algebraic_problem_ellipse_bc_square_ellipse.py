'''
this module solves for the variables theta, omega whic define the state of the ellipse
'''

from fenics import *
import importlib
import ufl as ufl

import calculus as cal
import geometry as geo
import read_parameters as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

dt = rpam.T / rpam.num_steps  # time step size


i, j, k, l = ufl.indices(4)


print(f'curve = {cal.ellipse(rmsh.a, rmsh.b,rmsh.c, rmsh.phi, 0.5)})

domega = assemble(Constant(1) * rmsh.ds_ellipse)

print(f'domega = {domega}')