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
        values[0] = np.cos(2 * np.pi * x[0])**3 / (1 + 10 * x[0]**4 * np.cos(np.pi * x[0])**4)

    def value_shape(self):
        return (1,)


class v_exact_expression( UserExpression ):
    def eval(self, values, x):
        values[0] = (np.cos(2 * np.pi * x[0])**2 * (40 * x[0]**3 * np.cos(np.pi * x[0])**3 * np.cos(2 * np.pi * x[0]) * (-np.cos(np.pi * x[0]) + np.pi * x[0] * np.sin(np.pi * x[0])) - 6 * np.pi * (1 + 10 * x[0]**4 * np.cos(np.pi * x[0])**4) * np.sin(2 * np.pi * x[0]))) / (1 + 10 * x[0]**4 * np.cos(np.pi * x[0])**4)**2

    def value_shape(self):
        return (1,)
    
class f_expression( UserExpression ):
    def eval(self, values, x):
        values[0] = 1/(1 + 10*x[0]**4*np.cos(np.pi*x[0])**4)**4*(-1440*x[0]**3*np.cos(np.pi*x[0])**3*(np.pi + 10*np.pi*x[0]**4*np.cos(np.pi*x[0])**4)**2*np.cos(2*np.pi*x[0])**3*(-np.cos(np.pi*x[0]) + np.pi*x[0]*np.sin(np.pi*x[0])) + 384000*x[0]**9*np.cos(np.pi*x[0])**9*np.cos(2*np.pi*x[0])**3*(-np.cos(np.pi*x[0]) + np.pi*x[0]*np.sin(np.pi*x[0]))**3 + 168*(np.pi + 10*np.pi*x[0]**4*np.cos(np.pi*x[0])**4)**3*np.cos(2*np.pi*x[0])**2*np.sin(2*np.pi*x[0]) - 57600*np.pi*x[0]**6*np.cos(np.pi*x[0])**6*(1 + 10*x[0]**4*np.cos(np.pi*x[0])**4)*np.cos(2*np.pi*x[0])**2*(np.cos(np.pi*x[0]) - np.pi*x[0]*np.sin(np.pi*x[0]))**2*np.sin(2*np.pi*x[0]) + 2880*np.pi**2*x[0]**3*np.cos(np.pi*x[0])**3*(1 + 10*x[0]**4*np.cos(np.pi*x[0])**4)**2*np.cos(2*np.pi*x[0])*(-np.cos(np.pi*x[0]) + np.pi*x[0]*np.sin(np.pi*x[0]))*np.sin(2*np.pi*x[0])**2 - 48*np.pi**3*(1 + 10*x[0]**4*np.cos(np.pi*x[0])**4)**3*np.sin(2*np.pi*x[0])**3 + 1200*x[0]**5*np.cos(np.pi*x[0])**5*np.cos(2*np.pi*x[0])**3*(4 + 15*x[0]**4 + 5*x[0]**4*(4*np.cos(2*np.pi*x[0]) + np.cos(4*np.pi*x[0])))*(-np.cos(np.pi*x[0]) + np.pi*x[0]*np.sin(np.pi*x[0]))*(-3 - 2*np.pi**2*x[0]**2 + (-3 + 4*np.pi**2*x[0]**2)*np.cos(2*np.pi*x[0]) + 8*np.pi*x[0]*np.sin(2*np.pi*x[0])) - 360*np.pi*x[0]**2*np.cos(np.pi*x[0])**2*(1 + 10*x[0]**4*np.cos(np.pi*x[0])**4)**2*np.cos(2*np.pi*x[0])**2*np.sin(2*np.pi*x[0])*(-3 - 2*np.pi**2*x[0]**2 + (-3 + 4*np.pi**2*x[0]**2)*np.cos(2*np.pi*x[0]) + 8*np.pi*x[0]*np.sin(2*np.pi*x[0])) + 5/4*x[0]*np.cos(np.pi*x[0])*np.cos(2*np.pi*x[0])**3*(4 + 15*x[0]**4 + 5*x[0]**4*(4*np.cos(2*np.pi*x[0]) + np.cos(4*np.pi*x[0])))**2*(-9*np.cos(np.pi*x[0]) + 3*(-1 + 8*np.pi**2*x[0]**2)*np.cos(3*np.pi*x[0]) + 2*np.pi*x[0]*((9 + 2*np.pi**2*x[0]**2)*np.sin(np.pi*x[0]) + (9 - 4*np.pi**2*x[0]**2)*np.sin(3*np.pi*x[0]))))
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
