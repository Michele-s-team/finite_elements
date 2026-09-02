from fenics import *
import importlib

import input_output as io
import mesh.load as lmsh
import mesh.utils as msh
import os
import runtime_arguments as rarg

parameters = io.read_parameters_from_csv_file(os.path.join(rarg.args.input_directory, "mesh_metadata.csv"))



# read the triangles
sf = msh.read_mesh_components(lmsh.mesh, lmsh.mesh.topology().dim(), rarg.args.input_directory + "/triangle_mesh.xdmf")
# read the lines
mf = msh.read_mesh_components(lmsh.mesh, lmsh.mesh.topology().dim() - 1, rarg.args.input_directory + "/line_mesh.xdmf")

# r_mesh[i] is the radius of the smallest cell in sub_meshes[i]
r_mesh =  [lmsh.sub_meshes[i].hmin() for i in range(len(lmsh.sub_meshes))]

# create line and surface elements for sub_meshes
dx_sub_mesh = []

for p in range(len(lmsh.sub_meshes)):
    dx_sub_mesh.append(Measure("dx", domain=lmsh.sub_meshes[p], subdomain_data=lmsh.sf_sub_meshes[p], subdomain_id=parameters[f"sub_mesh_{p}_id"]))


ds_sub_mesh = [''] * len(lmsh.sub_meshes)

ds_sub_mesh[0] = dict([ \
    ('ds_l', Measure("ds", domain=lmsh.sub_meshes[0], subdomain_data=lmsh.mf_sub_meshes[0], subdomain_id=parameters[f"line_sub_mesh_{0}_l_id"])), \
    ('ds_r', Measure("ds", domain=lmsh.sub_meshes[0], subdomain_data=lmsh.mf_sub_meshes[0], subdomain_id=parameters[f"line_sub_mesh_{0}_r_id"])), \
    ('ds_t', Measure("ds", domain=lmsh.sub_meshes[0], subdomain_data=lmsh.mf_sub_meshes[0], subdomain_id=parameters[f"sub_mesh_{1}_id"])), \
    ('ds_b', Measure("ds", domain=lmsh.sub_meshes[0], subdomain_data=lmsh.mf_sub_meshes[0], subdomain_id=parameters[f"line_sub_mesh_{0}_b_id"])), \
    ])

ds_sub_mesh[0]['ds_lr'] = ds_sub_mesh[0]['ds_l'] + ds_sub_mesh[0]['ds_r']
ds_sub_mesh[0]['ds_tb'] = ds_sub_mesh[0]['ds_t'] + ds_sub_mesh[0]['ds_b']

ds_sub_mesh[0]['ds'] = ds_sub_mesh[0]['ds_lr'] + ds_sub_mesh[0]['ds_tb']

ds_sub_mesh[1] = dict([ \
    ('ds_l', Measure("ds", domain=lmsh.sub_meshes[1], subdomain_data=lmsh.mf_sub_meshes[1], subdomain_id=parameters[f"vertex_sub_mesh_{1}_l_id"])), \
    ('ds_r', Measure("ds", domain=lmsh.sub_meshes[1], subdomain_data=lmsh.mf_sub_meshes[1], subdomain_id=parameters[f"vertex_sub_mesh_{1}_r_id"])), \
    ('ds', Measure("ds", domain=lmsh.sub_meshes[1], subdomain_data=lmsh.mf_sub_meshes[1]))
])

check_mesh_module = importlib.import_module('mesh.check_tags.square_no_circle_line')

print(f'Module {__file__} called {check_mesh_module.__file__}', flush=True)


