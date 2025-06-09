'''
this module solves for the variables theta, omega whic define the state of the ellipse
'''

from fenics import *
import importlib
import numpy as np
import ufl as ufl

import calculus as cal
import function_spaces as fsp
import geometry as geo
import input_output as io
import load_mesh as lmsh
import read_parameters as rpam
import solution_paths as solpath
import switch_problem as swi
from calculus import atan_quad

rmsh = importlib.import_module(swi.rmsh)

dt = rpam.T / rpam.num_steps  # time step size

i, j, k, l = ufl.indices(4)


class dyds_expression(UserExpression):
    def eval(self, values, x):
        s = 1 / (2 * np.pi) * atan_quad([rmsh.b * (x[0] - rmsh.c[0]), rmsh.a * (x[1] - rmsh.c[1])])

        t = cal.ellipse(rmsh.a, rmsh.b, rmsh.c[:2], 0, s)[1]

        # print(f't={t}')

        values[0] = t[0]
        values[1] = t[1]

    def value_shape(self):
        return (2,)

fsp.dyds.interpolate(dyds_expression(element=fsp.Q_dyds.ufl_element()))

io.full_print(fsp.dyds, 'dyds', \
              solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path, solpath.nodal_values_path, \
              lmsh.mesh, 'vector')


'''
by replacing '1' in the integrant with a function of 0 =< s < 1, integral_ellipse gives \int ds f(s)
'''
integral_ellipse = assemble(1 / sqrt(fsp.dyds[i] * fsp.dyds[i]) * rmsh.ds_ellipse)

print(f'int_ellipse = {integral_ellipse}')
