'''
in this variational problme we use Dirichlet conditions on a vertex within the mesh: we impose
- Dirichlet BCs for v on ds_l and ds_r
- a Dirichlet condition for u on the vertex in the middle of the line, ds_m
'''


from fenics import *
import importlib
import numpy as np
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j, k, l = ufl.indices( 4 )



class u_exact_expression( UserExpression ):
    def eval(self, values, x):
        values[0] = np.cos(x[0]) * np.sin(x[0])

    def value_shape(self):
        return (1,)


class v_exact_expression( UserExpression ):
    def eval(self, values, x):
        values[0] =  np.cos(2 * x[0])

    def value_shape(self):
        return (1,)
    
class f_expression( UserExpression ):
    def eval(self, values, x):
        values[0] =  -4 * np.cos(2 * x[0])

    def value_shape(self):
        return (1,)


fsp.u_exact.interpolate( u_exact_expression( element=fsp.Q_u.ufl_element() ) )
fsp.v_exact.interpolate( v_exact_expression( element=fsp.Q_v.ufl_element() ) )
fsp.f.interpolate( f_expression( element=fsp.Q_u.ufl_element() ) )


# main variational problem
bc_u_m = DirichletBC( fsp.Q.sub( 0 ), fsp.u_exact, rmsh.vf, rmsh.parameters['vertex_m_id'] )

bc_v_l = DirichletBC( fsp.Q.sub( 1 ), fsp.v_exact, rmsh.vf, rmsh.parameters['vertex_l_id'] )
bc_v_r = DirichletBC( fsp.Q.sub( 1 ), fsp.v_exact, rmsh.vf, rmsh.parameters['vertex_r_id'] )

bcs = [bc_v_l, bc_v_r, bc_u_m]

F_v = ((fsp.v.dx( i )) * (fsp.nu_v.dx( i )) + fsp.f * fsp.nu_v) * rmsh.dx \
      - bgeo.facet_normal[i] * (fsp.v.dx( i )) * fsp.nu_v * rmsh.ds_lr
F_u = ( fsp.u.dx( 0 ) - fsp.v )  * fsp.nu_u * rmsh.dx 
      
F = F_u + F_v
