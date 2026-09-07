from fenics import *
import importlib

import input_output as io
import mesh.load as lmsh
import mesh.utils as msh
import os
import runtime_arguments as rarg


# read parameters for the mesh ensemble

parameters = io.read_parameters_from_csv_file(os.path.join(rarg.args.input_directory, "mesh_metadata.csv"))


# read quantities for meshes

sf = [None] * lmsh.parameters['n_meshes']
mf = [None] * lmsh.parameters['n_meshes']


#1. read quantities for mesh[0]

# 1.1 read the triangles
sf[0] = msh.read_mesh_components(lmsh.mesh[0], (lmsh.mesh[0]).topology().dim(), os.path.join(rarg.args.input_directory, f'mesh_{0}', 'triangle_mesh.xdmf'))

# 1.2. read the lines
mf[0] = msh.read_mesh_components(lmsh.mesh[0], (lmsh.mesh[0]).topology().dim() - 1, os.path.join(rarg.args.input_directory, f'mesh_{0}', 'line_mesh.xdmf'))

# 2. read quantities for mesh[1]

# 2.1 read the lines
sf[1] = msh.read_mesh_components(lmsh.mesh[1], (lmsh.mesh[1]).topology().dim(), os.path.join(rarg.args.input_directory, f'mesh_{1}', "line_mesh.h5"), name_to_read="cf")

# 2.2 read the vertices
mf[1] = msh.read_mesh_components(lmsh.mesh[1], (lmsh.mesh[1]).topology().dim() - 1, os.path.join(rarg.args.input_directory, f'mesh_{1}', "vertex_mesh.h5"), name_to_read="vf")



# r_mesh[i] is the radius of the smallest cell in mesh[i]
r_mesh =  [lmsh.mesh[i].hmin() for i in range(len(lmsh.mesh))]

# create line and surface elements for meshes
dx_mesh = []
ds_mesh = [None] * lmsh.parameters['n_meshes']


for p in range(len(lmsh.mesh)):
    dx_mesh.append(Measure("dx", domain=lmsh.mesh[p], subdomain_data=lmsh.sf[p]))


ds_mesh[0] = dict([ \
    ('ds_l', Measure("ds", domain=lmsh.mesh[0], subdomain_data=mf[0], subdomain_id=parameters[f"line_l_id"])), \
    ('ds_r', Measure("ds", domain=lmsh.mesh[0], subdomain_data=mf[0], subdomain_id=parameters[f"line_r_id"])), \
    ('ds_t', Measure("ds", domain=lmsh.mesh[0], subdomain_data=mf[0], subdomain_id=parameters[f"mesh_{1}_id"])), \
    ('ds_b', Measure("ds", domain=lmsh.mesh[0], subdomain_data=mf[0], subdomain_id=parameters[f"line_b_id"])), \
    ])

ds_mesh[0]['ds_lr'] = ds_mesh[0]['ds_l'] + ds_mesh[0]['ds_r']
ds_mesh[0]['ds_tb'] = ds_mesh[0]['ds_t'] + ds_mesh[0]['ds_b']

ds_mesh[0]['ds'] = ds_mesh[0]['ds_lr'] + ds_mesh[0]['ds_tb']

ds_mesh[1] = dict([ \
    ('ds_l', Measure("ds", domain=lmsh.mesh[1], subdomain_data=mf[1], subdomain_id=parameters[f"vertex_l_id"])), \
    ('ds_r', Measure("ds", domain=lmsh.mesh[1], subdomain_data=mf[1], subdomain_id=parameters[f"vertex_r_id"])), \
    ('ds', Measure("ds", domain=lmsh.mesh[1], subdomain_data=mf[1]))
])

check_mesh_module = importlib.import_module('mesh.check_tags.square_no_circle_line')

print(f'Module {__file__} called {check_mesh_module.__file__}', flush=True)


