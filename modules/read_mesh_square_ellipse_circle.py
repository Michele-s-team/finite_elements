'''
Notation:
- sub_mesh: either of the parts of the total mesh

- sf_sub_mesh: a list of map functions, where sf_sub_mesh[i] is the map function for the triangles of the i-th sub_mesh
- mf_sub_mesh: a list of map functions, where mf_sub_mesh[i] is the map function for the lines of the i-th sub_mesh
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

# create line and surface elements for sub_meshes
dx_sub_mesh = []

for p in range(len(lmsh.sub_meshes)):
    dx_sub_mesh.append(Measure("dx", domain=lmsh.sub_meshes[p], subdomain_data=sf_sub_mesh[p], subdomain_id=parameters[f"sub_mesh_{p}_id"]))

ds_sub_mesh = [''] * len(lmsh.sub_meshes)
ds_sub_mesh[0] = dict([ \
    ('ds_ellipse', Measure("ds", domain=lmsh.sub_meshes[0], subdomain_data=mf_sub_mesh[0], subdomain_id=parameters[f"ellipse_loop_id"])) \
    ])
ds_sub_mesh[1] = dict([ \
    ('ds_l', Measure("ds", domain=lmsh.sub_meshes[1], subdomain_data=mf_sub_mesh[1], subdomain_id=parameters[f"line_sub_mesh_{1}_l_id"])), \
    ('ds_r', Measure("ds", domain=lmsh.sub_meshes[1], subdomain_data=mf_sub_mesh[1], subdomain_id=parameters[f"line_sub_mesh_{1}_r_id"])), \
    ('ds_t', Measure("ds", domain=lmsh.sub_meshes[1], subdomain_data=mf_sub_mesh[1], subdomain_id=parameters[f"line_sub_mesh_{1}_t_id"])), \
    ('ds_b', Measure("ds", domain=lmsh.sub_meshes[1], subdomain_data=mf_sub_mesh[1], subdomain_id=parameters[f"line_sub_mesh_{1}_b_id"])), \
    ('ds_ellipse', Measure("ds", domain=lmsh.sub_meshes[1], subdomain_data=mf_sub_mesh[1], subdomain_id=parameters[f"ellipse_loop_id"]))\
    ])
ds_sub_mesh[1]['ds_lr'] = ds_sub_mesh[1]['ds_l'] + ds_sub_mesh[1]['ds_r']
ds_sub_mesh[1]['ds_tb'] = ds_sub_mesh[1]['ds_t'] + ds_sub_mesh[1]['ds_b']
ds_sub_mesh[1]['ds_lrtb'] = ds_sub_mesh[1]['ds_lr'] + ds_sub_mesh[1]['ds_tb']
ds_sub_mesh[1]['ds'] = ds_sub_mesh[1]['ds_lrtb'] + ds_sub_mesh[1]['ds_ellipse']

import check_mesh_tags_square_ellipse_circle

'''

print(f'Module {__file__} called {check_mesh_tags_square_square.__file__}', flush=True)

#Define boundaries
boundary = 'on_boundary'

boundary_l  = [''] * len(lmsh.sub_meshes)
boundary_r  = [''] * len(lmsh.sub_meshes)
boundary_t  = [''] * len(lmsh.sub_meshes)
boundary_b  = [''] * len(lmsh.sub_meshes)
boundary_lr = [''] * len(lmsh.sub_meshes)
boundary_tb = [''] * len(lmsh.sub_meshes)
boundary_lrtb = [''] * len(lmsh.sub_meshes)


# outer boundaries (sub_mesh_1)
boundary_l[1] = f'near(x[0], {0})'
boundary_r[1] = f'near(x[0], {parameters["L"]})'
boundary_t[1] = f'near(x[1], {parameters["h"]})'
boundary_b[1] = f'near(x[1], {0})'
boundary_lr[1] = f'({boundary_l[1]}) || ({boundary_r[1]})'
boundary_tb[1] = f'({boundary_t[1]}) || ({boundary_b[1]})'
boundary_lrtb[1] = f'({boundary_lr[1]}) || ({boundary_tb[1]})'

# inner boundaries (sub_mesh_0)
boundary_l[0] = f'on_boundary && near(x[0], {parameters["p"][0]}) && !{boundary_t[1]} && !{boundary_b[1]}'
boundary_r[0] = f'on_boundary && near(x[0], {parameters["p"][0] + parameters["L_in"]}) && !{boundary_t[1]} && !{boundary_b[1]}'
boundary_t[0] = f'on_boundary && near(x[1], {parameters["p"][1] + parameters["h_in"]}) && !{boundary_l[1]} && !{boundary_r[1]}'
boundary_b[0] = f'on_boundary && near(x[1], {parameters["p"][1]}) && !{boundary_l[1]} && !{boundary_r[1]}'
boundary_lr[0] = f'({boundary_l[0]}) || ({boundary_r[0]})'
boundary_tb[0] = f'({boundary_t[0]}) || ({boundary_b[0]})'
boundary_lrtb[0] = f'({boundary_lr[0]}) || ({boundary_tb[0]})'
'''
