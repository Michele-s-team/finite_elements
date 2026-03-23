from fenics import *
import importlib
import numpy as np
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)

fsp.u_exact[0][1].interpolate(fsp.u_exact_expression(element=fsp.Q[0][1].ufl_element()))
fsp.f[0][1].interpolate(fsp.laplacian_u_exact_expression(element=fsp.Q[0][1].ufl_element()))

# boundary conditions for sub_mesh[0][1]
bcs = [ \
    DirichletBC(fsp.Q[0][1], fsp.u_exact[0][1], rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["line_l_id"]),\
    DirichletBC(fsp.Q[0][1], fsp.u_exact[0][1], rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["line_r_id"]),\
    DirichletBC(fsp.Q[0][1], fsp.u_exact[0][1], rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["line_t_id"]),\
    DirichletBC(fsp.Q[0][1], fsp.u_exact[0][1], rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["line_b_id"]),\
    DirichletBC(fsp.Q[0][1], fsp.u_exact[0][1], rmsh.lmsh.mf_sub_meshes[0][1], rmsh.lmsh.mesh_parameters[0]["shape_id"])
    ]

# variational functional for sub_mesh[1]
F = (fsp.u[0][1].dx(i) * fsp.nu_u[0][1].dx(i) + fsp.f[0][1] * fsp.nu_u[0][1]) * rmsh.dx_sub_mesh[0][1] \
    - bgeo.sub_mesh_facet_normal[0][1][i] * (fsp.u[0][1].dx(i)) * fsp.nu_u[0][1] * rmsh.ds_sub_mesh[0][1]['ds_lrtb']\
    - bgeo.sub_mesh_facet_normal[0][1][i] * (fsp.u[0][1].dx(i)) * fsp.nu_u[0][1] * rmsh.ds_sub_mesh[0][1]['ds_shape']
