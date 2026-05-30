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



'''# print facet_normal to check sub_mesh_0_label and sub_mesh_1_label
import input_output as io 
import solution_paths as solpath

n_1 = bgeo.field_facet_normal(bgeo.facet_normal[0](sub_mesh_1_label), rmsh.lmsh.mesh[0], rmsh.ds_mesh[0]['dS_shape'], interior=True)

io.full_print(n_1, 'n_1', \
                  solpath.snapshots_path, solpath.snapshots_h5_path, solpath.snapshots_csv_path, solpath.snapshots_csv_nodal_values_path, rmsh.sf[0])
'''


# 1. define expressions for BCs

'''class n_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 1.0

    def value_shape(self):
        return (1,)

    
msh.interpolate_dg(fsp.n, n_expression())'''

fsp.n.assign(bgeo.field_facet_normal_normalized(rmsh.lmsh.mesh[0], bgeo.facet_normal[0](sub_mesh_0_label),  rmsh.ds_mesh[0]['dS_shape'], interior=True))

fsp.t.assign(bgeo.field_facet_tangent_normalized(rmsh.lmsh.mesh[0], bgeo.facet_normal[0](sub_mesh_0_label),  rmsh.ds_mesh[0]['dS_shape'], interior=True))


bcs = []

# variational problem

F = (\
        (fsp.u[i] - fsp.n[i]) * fsp.nu_u[i] \
    ) * rmsh.dx_mesh[0]['dx'] 
