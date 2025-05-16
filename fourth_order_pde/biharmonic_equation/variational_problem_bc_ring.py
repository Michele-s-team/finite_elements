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
        values[0] = np.cos(x[0]+x[1]) * np.sin(x[0]-x[1])

    def value_shape(self):
        return (1,)


class v_exact_expression( UserExpression ):
    def eval(self, values, x):
        values[0] = - 4 * np.cos(x[0])*np.sin(x[0]) + 4 * np.cos(x[1])*np.sin(x[1])

    def value_shape(self):
        return (1,)


class w_exact_expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 8 * (np.sin(2*x[0]) - np.sin(2*x[1]))

    def value_shape(self):
        return (1,)

fsp.u_exact.interpolate( u_exact_expression( element=fsp.Q_u.ufl_element() ) )
fsp.v_exact.interpolate( v_exact_expression( element=fsp.Q_v.ufl_element() ) )
fsp.w_exact.interpolate( w_exact_expression( element=fsp.Q_w.ufl_element() ) )
fsp.f.interpolate( w_exact_expression( element=fsp.Q_w.ufl_element() ) )

u_profile = Expression( 'cos(x[0]+x[1]) * sin(x[0]-x[1])', element=fsp.Q.sub( 0 ).ufl_element() )
v_profile = Expression( '- 4 * cos(x[0])*sin(x[0]) + 4 * cos(x[1])*sin(x[1])', element=fsp.Q.sub( 1 ).ufl_element() )
w_profile = Expression( '8 * (sin(2*x[0]) - sin(2*x[1]))', element=fsp.Q.sub( 2 ).ufl_element() )
bc_u = DirichletBC( fsp.Q.sub( 0 ), u_profile, boundary )
bc_v = DirichletBC( fsp.Q.sub( 1 ), v_profile, boundary )
bc_w = DirichletBC( fsp.Q.sub( 2 ), w_profile, boundary )

F_v = ((v.dx( i )) * (nu_v.dx( i )) + f * nu_v) * rmsh.dx \
      - n[i] * (v.dx( i )) * nu_v * rmsh.ds
F_u = ((u.dx( i )) * (nu_u.dx( i )) + v * nu_u) * rmsh.dx \
      - n[i] * (u.dx( i )) * nu_u * rmsh.ds
F_w = ((v.dx( i )) * (nu_w.dx( i )) + w * nu_w) * rmsh.dx \
      - n[i] * (v.dx( i )) * nu_w * rmsh.ds

F = F_u + F_v + F_w
bcs = [bc_u, bc_v, bc_w]