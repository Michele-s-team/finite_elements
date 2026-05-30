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

class t_expression(UserExpression):
    def eval(self, values, x):

        theta = cal.atan_quad(np.subtract(x, rmsh.parameters['c']))

        values[0] = -np.sin(theta)
        values[1] = np.cos(theta)

    def value_shape(self):
        return (2,)
    
class n_expression(UserExpression):
    def eval(self, values, x):

        theta = cal.atan_quad(np.subtract(x, rmsh.parameters['c']))

        values[0] = np.cos(theta)
        values[1] = np.sin(theta)

    def value_shape(self):
        return (2,)

    
# msh.interpolate_dg(fsp.t, t_expression())
# msh.interpolate_dg(fsp.n, n_expression())

fsp.t.interpolate(t_expression(element=fsp.V.ufl_element()))
fsp.n.interpolate(n_expression(element=fsp.V.ufl_element()))


# fsp.n_0.assign(bgeo.field_facet_normal_normalized(rmsh.lmsh.mesh[0], bgeo.facet_normal[0](sub_mesh_0_label),  rmsh.ds_mesh[0]['dS_shape'], interior=True))

# fsp.t_0.assign(bgeo.field_facet_tangent_normalized(rmsh.lmsh.mesh[0], bgeo.facet_normal[0](sub_mesh_0_label),  rmsh.ds_mesh[0]['dS_shape'], interior=True))


bcs = []

# # variational problem

F = (\
        (fsp.mu - fsp.t[i].dx(j) * fsp.t[j] * fsp.n[i]) * fsp.nu_mu \
    ) * rmsh.dx_mesh[0]['dx'] 
