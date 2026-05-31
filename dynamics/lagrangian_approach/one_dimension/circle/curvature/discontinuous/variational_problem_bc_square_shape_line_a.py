'''
NOTE: this VP is well posed only of the degree of the function space of u is <= 2. If it is > 2, some DOFs sitting within the mesh triangles will appear, and these will not be constrained by the VP F
'''

from fenics import *
import importlib
import numpy as np
import ufl as ufl

import calculus as cal
import differential_geometry.boundary.geometry as bgeo
import mesh.utils as msh
import physics.fluid_mechanics as flu
import physics.elasticity as ela
import parameters.read.solution as rpam
import switch_problem as swi


fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)

i, j, k, l, m = ufl.indices(5)


sub_mesh_0_label, sub_mesh_1_label = msh.plus_minus(rmsh.lmsh.mesh[0], rmsh.sf[0], rmsh.lmsh.parameters["sub_mesh_0_0_id"], rmsh.lmsh.parameters["sub_mesh_0_1_id"], rmsh.ds_mesh[0]['dS_shape'])



# 1. define expressions for BCs

class e_expression(UserExpression):
    def eval(self, values, x):
        '''
        the curve is parametrized with 
        r(t) = c + {a cos(2 pi t), b sin(2 pi t)}

        and 0 < t < 1
        
        '''

        t = 1.0/(2.0*np.pi) * cal.atan_quad([ (x[0] - rmsh.parameters['c'][0])/rmsh.parameters['a'], (x[1] - rmsh.parameters['c'][1])/rmsh.parameters['b'] ])

        values[0] = -2.0*np.pi * rmsh.parameters['a'] * np.sin(2.0 * np.pi * t)  
        values[1] = 2.0*np.pi * rmsh.parameters['b'] * np.cos(2.0 * np.pi * t) 

    def value_shape(self):
        return (2,)
    
    
class n_expression(UserExpression):
    def eval(self, values, x):

        t = 1.0/(2.0*np.pi) * cal.atan_quad([ (x[0] - rmsh.parameters['c'][0])/rmsh.parameters['a'], (x[1] - rmsh.parameters['c'][1])/rmsh.parameters['b'] ])

        norm = np.sqrt((2.0*np.pi * rmsh.parameters['a'] * np.sin(2*np.pi*t))**2 + (2.0*np.pi * rmsh.parameters['b'] * np.cos(2*np.pi*t))**2)

        values[0] = 2.0*np.pi * rmsh.parameters['b'] * np.cos(2*np.pi*t) / norm
        values[1] = 2.0*np.pi * rmsh.parameters['a'] * np.sin(2*np.pi*t) / norm

    def value_shape(self):
        return (2,)

    
msh.interpolate_dg(fsp.e, e_expression())
msh.interpolate_dg(fsp.n, n_expression())

# fsp.e.interpolate(t_expression(element=fsp.V.ufl_element()))
# fsp.n.interpolate(n_expression(element=fsp.V.ufl_element()))


# fsp.n_0.assign(bgeo.field_facet_normal_normalized(rmsh.lmsh.mesh[0], bgeo.facet_normal[0](sub_mesh_0_label),  rmsh.ds_mesh[0]['dS_shape'], interior=True))

# fsp.e_0.assign(bgeo.field_facet_tangent_normalized(rmsh.lmsh.mesh[0], bgeo.facet_normal[0](sub_mesh_0_label),  rmsh.ds_mesh[0]['dS_shape'], interior=True))


bcs = []

# # variational problem

F = (\
        (fsp.mu - 1.0/2.0 * (fsp.e[i].dx(j) * fsp.e[j] * fsp.n[i]) / dot(fsp.e, fsp.e)) * fsp.nu_mu \
    ) * rmsh.dx_mesh[0]['dx'] \
    + rpam.parameters['alpha']/rmsh.r_mesh[0] * (\
        msh.jump(fsp.mu, bgeo.facet_normal[0])[i] *  msh.jump(fsp.nu_mu, bgeo.facet_normal[0])[i] * (rmsh.ds_mesh[0]['dS_I_shape'] + rmsh.ds_mesh[0]['dS_I_square'] + rmsh.ds_mesh[0]['dS_shape']) \
    )