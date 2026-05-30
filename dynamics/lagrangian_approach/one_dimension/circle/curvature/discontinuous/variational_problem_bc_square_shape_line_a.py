'''
NOTE: this VP is well posed only of the degree of the function space of u is <= 2. If it is > 2, some DOFs sitting within the mesh triangles will appear, and these will not be constrained by the VP F
'''

from fenics import *
import importlib
import numpy as np
import ufl as ufl

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



class TangentFromProjection(UserExpression):

    def __init__(self, mesh, mf, boundary_id, t_field, **kwargs):
        super().__init__(**kwargs)
        self.facets = []
        for facet in facets(mesh):

            if mf[facet] == boundary_id:
                
                verts = [v.point().array()[:2] for v in vertices(facet)]
                p0, p1 = np.array(verts[0]), np.array(verts[1])
                self.facets.append((p0, p1))

        self.t_field = t_field

    def eval(self, values, x):

        x_projected = np.array(x[:2])
        best_dist = np.inf
        best_proj = None

        for p0, p1 in self.facets:

            t_vec = p1 - p0
            L = np.linalg.norm(t_vec)
            t_hat = t_vec / L
            s = np.clip(np.dot(x_projected - p0, t_hat), 0.0, L)
            x_proj = p0 + s * t_hat
            dist = np.linalg.norm(x_projected - x_proj)
            if dist < best_dist:
                best_dist = dist
                best_proj = x_proj

        values[:] = self.t_field(Point(best_proj[0], best_proj[1]))[:2]
    
    def value_shape(self):
        return (2,)



# 1. define expressions for BCs

'''class n_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 1.0

    def value_shape(self):
        return (1,)

    
msh.interpolate_dg(fsp.n, n_expression())'''

fsp.n_0.assign(bgeo.field_facet_normal_normalized(rmsh.lmsh.mesh[0], bgeo.facet_normal[0](sub_mesh_0_label),  rmsh.ds_mesh[0]['dS_shape'], interior=True))

fsp.t_0.assign(bgeo.field_facet_tangent_normalized(rmsh.lmsh.mesh[0], bgeo.facet_normal[0](sub_mesh_0_label),  rmsh.ds_mesh[0]['dS_shape'], interior=True))

msh.interpolate_dg(fsp.n, TangentFromProjection(rmsh.lmsh.mesh[0], rmsh.mf_I[0], rmsh.lmsh.parameters['shape_id'], fsp.n_0))
msh.interpolate_dg(fsp.t, TangentFromProjection(rmsh.lmsh.mesh[0], rmsh.mf_I[0], rmsh.lmsh.parameters['shape_id'], fsp.t_0))

bcs = []

# # variational problem

F = (\
        (fsp.mu - 1.0/2.0 * fsp.t[i].dx(j) * fsp.t[j] * fsp.n[i]) * fsp.nu_mu \
    ) * rmsh.dx_mesh[0]['dx'] 
