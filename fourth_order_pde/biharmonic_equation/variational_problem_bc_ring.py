import dolfin
from fenics import *
import importlib
import numpy as np
import ufl as ufl

import boundary_geometry as bgeo
import function_spaces as fsp
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices( 4 )



class u_exact_expression( UserExpression ):
    def eval(self, values, x):
        values[0] = cos(x[0]+x[1]) * sin(x[0]-x[1])

    def value_shape(self):
        return (1,)


class v_exact_expression( UserExpression ):
    def eval(self, values, x):
        values[0] = - 4 * cos(x[0])*sin(x[0]) + 4 * cos(x[1])*sin(x[1])

    def value_shape(self):
        return (1,)


class w_exact_expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 8 * (sin(2*x[0]) - sin(2*x[1]))

    def value_shape(self):
        return (1,)