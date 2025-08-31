'''
Notation:
- sub_mesh: either of the parts of the total mesh

- sf_sub_mesh: a list of map functions, where sf_sub_mesh[i] is the map function for the triangles of the i-th sub_mesh
- mf_sub_mesh: a list of map functions, where mf_sub_mesh[i] is the map function for the lines of the i-th sub_mesh
'''

from fenics import *

import input_output as io
import mesh.load as lmsh
import mesh.utils as msh
import runtime_arguments as rarg

parameters = io.read_parameters_from_csv_file(rarg.args.input_directory + "/mesh_metadata.csv")

print(f'**** DIMENSION *** = {lmsh.mesh.topology().dim()}')

# read the triangles
sf = msh.read_mesh_components(lmsh.mesh, lmsh.mesh.topology().dim(), rarg.args.input_directory + "/triangle_mesh.xdmf")
# read the lines
mf = msh.read_mesh_components(lmsh.mesh, lmsh.mesh.topology().dim() - 1, rarg.args.input_directory + "/line_mesh.xdmf")

# create a list of map functions for triangles and lines for each sub_mesh
sf_sub_mesh = []
mf_sub_mesh = []

sub_mesh = lmsh.sub_meshes[0]
sf_sub_mesh.append(msh.transfer_cell_tags_to_sub_mesh(sub_mesh, sf))
mf_sub_mesh.append(msh.transfer_facet_tags_to_sub_mesh(lmsh.mesh, sub_mesh, mf))

sub_mesh = lmsh.sub_meshes[1]
sf_sub_mesh.append(lmsh.cf_sub_mesh_1d)
mf_sub_mesh.append(lmsh.vf_sub_mesh_1d)

# radius of the smallest cell in the mesh
r_mesh = lmsh.mesh.hmin()

# create line and surface elements for sub_meshes
dx_sub_mesh = []

for p in range(len(lmsh.sub_meshes)):
    dx_sub_mesh.append(Measure("dx", domain=lmsh.sub_meshes[p], subdomain_data=sf_sub_mesh[p], subdomain_id=parameters[f"sub_mesh_{p}_id"]))

print(f'test: {assemble(Constant(1) * dx_sub_mesh[1])}')

ds_sub_mesh = [''] * len(lmsh.sub_meshes)

ds_sub_mesh[0] = dict([ \
    ('ds_l', Measure("ds", domain=lmsh.sub_meshes[0], subdomain_data=mf_sub_mesh[0], subdomain_id=parameters[f"line_sub_mesh_{0}_l_id"])), \
    ('ds_r', Measure("ds", domain=lmsh.sub_meshes[0], subdomain_data=mf_sub_mesh[0], subdomain_id=parameters[f"line_sub_mesh_{0}_r_id"])), \
    ('ds_t', Measure("ds", domain=lmsh.sub_meshes[0], subdomain_data=mf_sub_mesh[0], subdomain_id=parameters[f"sub_mesh_{1}_id"])), \
    ('ds_b', Measure("ds", domain=lmsh.sub_meshes[0], subdomain_data=mf_sub_mesh[0], subdomain_id=parameters[f"line_sub_mesh_{0}_b_id"])), \
    ])

ds_sub_mesh[0]['ds_lr'] = ds_sub_mesh[0]['ds_l'] + ds_sub_mesh[0]['ds_r']
ds_sub_mesh[0]['ds_tb'] = ds_sub_mesh[0]['ds_t'] + ds_sub_mesh[0]['ds_b']

ds_sub_mesh[0]['ds'] = ds_sub_mesh[0]['ds_lr'] + ds_sub_mesh[0]['ds_tb']

ds_sub_mesh[1] = dict([ \
    ('ds_l', Measure("ds", domain=lmsh.sub_meshes[1], subdomain_data=mf_sub_mesh[1], subdomain_id=parameters[f"vertex_sub_mesh_{1}_l_id"])), \
    ('ds_r', Measure("ds", domain=lmsh.sub_meshes[1], subdomain_data=mf_sub_mesh[1], subdomain_id=parameters[f"vertex_sub_mesh_{1}_r_id"])), \
    ('ds', Measure("ds", domain=lmsh.sub_meshes[1], subdomain_data=mf_sub_mesh[1]))
])

import importlib
check_mesh_module = importlib.import_module('mesh.check_tags.square_no_circle_line')

print(f'Module {__file__} called {check_mesh_module.__file__}', flush=True)

# Define boundaries: it is important that these boundaries are defined in the right order, because a definition may call a preceeding one

boundary = [''] * len(lmsh.sub_meshes)

# sub_mesh 0
boundary[0] = dict([ \
    ('l', f'near(x[0], {0})'), \
    ('r', f'near(x[0], {parameters["L"]})'), \
    ('t', f'near(x[1], {parameters["h"]})'), \
    ('b', f'near(x[1], {0})') \
    ])

boundary[0]['lr'] = f"({boundary[0]['l']}) || ({boundary[0]['r']})"
boundary[0]['tb'] = f"({boundary[0]['t']}) || ({boundary[0]['b']})"

boundary[0]['lrtb'] = f"({boundary[0]['lr']}) || ({boundary[0]['tb']})"

# sub_mesh 1
boundary[1] = dict([ \
    ('l', f'near(x[0], {0})'), \
    ('r', f'near(x[0], {parameters["L"]})') \
    ])
boundary[1]['lr'] = f"({boundary[1]['l']}) || ({boundary[1]['r']})"
