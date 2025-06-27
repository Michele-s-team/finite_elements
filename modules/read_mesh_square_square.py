'''
Notation:
- dx_submesh_out : surface element of the outer submesh
- ds_submesh_out_out_l : surface element corresponding to the 'l' line on the outer boundary of submesh_out,
- ds_submesh_out_in_l : surface element corresponding to the 'l' line on the inner boundary of submesh_out,
- ...
'''

from fenics import *

import input_output as io
import load_mesh as lmsh
import mesh as msh
import runtime_arguments as rarg

# read the triangles
sf = msh.read_mesh_components(lmsh.mesh, lmsh.mesh.topology().dim(), rarg.args.input_directory + "/triangle_mesh.xdmf")
# read the lines
mf = msh.read_mesh_components(lmsh.mesh, lmsh.mesh.topology().dim() - 1, rarg.args.input_directory + "/line_mesh.xdmf")

parameters = io.read_parameters_from_csv_file(rarg.args.input_directory + "/mesh_metadata.csv")

# create a list of map functions for triangles and lines for each sub_mesh
sf_sub_mesh = []
mf_sub_mesh = []
for sub_mesh in lmsh.sub_meshes:
    sf_sub_mesh.append(msh.transfer_cell_tags_to_sub_mesh(sub_mesh, sf))
    mf_sub_mesh.append(msh.transfer_facet_tags_to_sub_mesh(lmsh.mesh, sub_mesh, mf))

# radius of the smallest cell in the mesh
r_mesh = lmsh.mesh.hmin()

# 1. create line and surface elements for mesh
# test for surface elements
dx_parent_mesh, ds_parent_mesh_l, ds_parent_mesh_r, ds_parent_mesh_t, ds_parent_mesh_b, ds_parent_mesh_lr, ds_parent_mesh_tb, ds_parent_mesh = [], [], [], [], [], [], [], []

# line elements for in square
ds_parent_mesh_l.append(Measure("dS", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=parameters["line_in_l_id"]))
ds_parent_mesh_l.append(Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=parameters["line_out_l_id"]))

ds_parent_mesh_r.append(Measure("dS", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=parameters["line_in_r_id"]))
ds_parent_mesh_r.append(Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=parameters["line_out_r_id"]))

ds_parent_mesh_t.append(Measure("dS", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=parameters["line_in_t_id"]))
ds_parent_mesh_t.append(Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=parameters["line_out_t_id"]))

ds_parent_mesh_b.append(Measure("dS", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=parameters["line_in_b_id"]))
ds_parent_mesh_b.append(Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=parameters["line_out_b_id"]))

for i in range(len(lmsh.sub_meshes)):
    dx_parent_mesh.append(Measure("dx", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=parameters[f"sub_mesh_{i}_id"]))

    ds_parent_mesh_lr.append(ds_parent_mesh_l[i] + ds_parent_mesh_r[i])
    ds_parent_mesh_tb.append(ds_parent_mesh_t[i] + ds_parent_mesh_b[i])

    ds_parent_mesh.append(ds_parent_mesh_lr[i] + ds_parent_mesh_tb[i])

ds = ds_parent_mesh[0] + ds_parent_mesh[1]

# create line and surface elements for sub_meshes
# create the measure dx_sub_mesh_ correspnding to the triangles of sub_mesh

dx_sub_mesh = []
for i in range(len(lmsh.sub_meshes)):
    dx_sub_mesh.append(Measure("dx", domain=lmsh.sub_meshes[i], subdomain_data=sf_sub_mesh[i], subdomain_id=parameters[f"sub_mesh_{i}_id"]))

    # # line elements for out square
    # ds_sub_mesh_out_l = Measure("ds", domain=lmsh.sub_meshes[i], subdomain_data=mf_sub_mesh, subdomain_id=parameters["line_out_l_id"])
    # ds_sub_mesh_out_r = Measure("ds", domain=lmsh.sub_meshes[i], subdomain_data=mf_sub_mesh, subdomain_id=parameters["line_out_r_id"])
    # ds_sub_mesh_out_t = Measure("ds", domain=lmsh.sub_meshes[i], subdomain_data=mf_sub_mesh, subdomain_id=parameters["line_out_t_id"])
    # ds_sub_mesh_out_b = Measure("ds", domain=lmsh.sub_meshes[i], subdomain_data=mf_sub_mesh, subdomain_id=parameters["line_out_b_id"])
    #
    # # line elements for in square
    # ds_sub_mesh_in_l = Measure("ds", domain=lmsh.sub_meshes[i], subdomain_data=mf_sub_mesh, subdomain_id=parameters["line_in_l_id"])
    # ds_sub_mesh_in_r = Measure("ds", domain=lmsh.sub_meshes[i], subdomain_data=mf_sub_mesh, subdomain_id=parameters["line_in_r_id"])
    # ds_sub_mesh_in_t = Measure("ds", domain=lmsh.sub_meshes[i], subdomain_data=mf_sub_mesh, subdomain_id=parameters["line_in_t_id"])
    # ds_sub_mesh_in_b = Measure("ds", domain=lmsh.sub_meshes[i], subdomain_data=mf_sub_mesh, subdomain_id=parameters["line_in_b_id"])
    #
    # ds_sub_mesh_out_lr = ds_sub_mesh_out_l + ds_sub_mesh_out_r
    # ds_sub_mesh_out_tb = ds_sub_mesh_out_t + ds_sub_mesh_out_b
    #
    # ds_sub_mesh_in_lr = ds_sub_mesh_in_l + ds_sub_mesh_in_r
    # ds_sub_mesh_in_tb = ds_sub_mesh_in_t + ds_sub_mesh_in_b
    #
    # ds_sub_mesh_out = ds_sub_mesh_out_lr + ds_sub_mesh_out_tb
    # ds_sub_mesh_in = ds_sub_mesh_in_lr + ds_sub_mesh_in_tb
    #
    # ds_sub_mesh = ds_sub_mesh_in + ds_sub_mesh_out

import check_mesh_tags_square_square

print(f'Module {__file__} called {check_mesh_tags_square_square.__file__}', flush=True)

# 1.  Define boundaries
boundary = 'on_boundary'

# outer boundaries
boundary_out_l = f'near(x[0], {0})'
boundary_out_r = f'near(x[0], {parameters["L"]})'
boundary_out_t = f'near(x[1], {parameters["h"]})'
boundary_out_b = f'near(x[1], {0})'
boundary_out_lr = f'({boundary_out_l}) || ({boundary_out_r})'
boundary_out_tb = f'({boundary_out_t}) || ({boundary_out_b})'
boundary_out = f'({boundary_out_lr}) || ({boundary_out_tb})'

# inner boundaries
boundary_in_l = f'on_boundary && near(x[0], {parameters["p"][0]}) && !{boundary_out_t} && !{boundary_out_b}'
boundary_in_r = f'on_boundary && near(x[0], {parameters["p"][0] + parameters["L_in"]}) && !{boundary_out_t} && !{boundary_out_b}'
boundary_in_t = f'on_boundary && near(x[1], {parameters["p"][1] + parameters["h_in"]}) && !{boundary_out_l} && !{boundary_out_r}'
boundary_in_b = f'on_boundary && near(x[1], {parameters["p"][1]}) && !{boundary_out_l} && !{boundary_out_r}'
boundary_in_lr = f'({boundary_in_l}) || ({boundary_in_r})'
boundary_in_tb = f'({boundary_in_t}) || ({boundary_in_b})'
boundary_in = f'({boundary_in_lr}) || ({boundary_in_tb})'
