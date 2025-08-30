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
'''
# create line and surface elements for the parent mesh
dx_parent_mesh, ds_parent_mesh_l, ds_parent_mesh_r, ds_parent_mesh_t, ds_parent_mesh_b, ds_parent_mesh_lr, ds_parent_mesh_tb, ds_parent_mesh_lrtb = [], [], [], [], [], [], [], []

ds_parent_mesh_l.append(Measure("dS", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=parameters["line_sub_mesh_0_l_id"]))
ds_parent_mesh_l.append(Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=parameters["line_sub_mesh_1_l_id"]))

ds_parent_mesh_r.append(Measure("dS", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=parameters["line_sub_mesh_0_r_id"]))
ds_parent_mesh_r.append(Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=parameters["line_sub_mesh_1_r_id"]))

ds_parent_mesh_t.append(Measure("dS", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=parameters["line_sub_mesh_0_t_id"]))
ds_parent_mesh_t.append(Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=parameters["line_sub_mesh_1_t_id"]))

ds_parent_mesh_b.append(Measure("dS", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=parameters["line_sub_mesh_0_b_id"]))
ds_parent_mesh_b.append(Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=parameters["line_sub_mesh_1_b_id"]))

for p in range(len(lmsh.sub_meshes)):
    dx_parent_mesh.append(Measure("dx", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=parameters[f"sub_mesh_{p}_id"]))

    ds_parent_mesh_lr.append(ds_parent_mesh_l[p] + ds_parent_mesh_r[p])
    ds_parent_mesh_tb.append(ds_parent_mesh_t[p] + ds_parent_mesh_b[p])

    ds_parent_mesh_lrtb.append(ds_parent_mesh_lr[p] + ds_parent_mesh_tb[p])

ds_parent_mesh = ds_parent_mesh_lrtb[0] + ds_parent_mesh_lrtb[1]
'''

# create line and surface elements for sub_meshes
dx_sub_mesh = []

for p in range(len(lmsh.sub_meshes)):
    dx_sub_mesh.append(Measure("dx", domain=lmsh.sub_meshes[p], subdomain_data=sf_sub_mesh[p], subdomain_id=parameters[f"sub_mesh_{p}_id"]))

ds_sub_mesh = [''] * len(lmsh.sub_meshes)

ds_sub_mesh[0] = dict([ \
    ('l', Measure("ds", domain=lmsh.sub_meshes[0], subdomain_data=mf_sub_mesh[0], subdomain_id=parameters[f"line_sub_mesh_{0}_l_id"])), \
    ('r', Measure("ds", domain=lmsh.sub_meshes[0], subdomain_data=mf_sub_mesh[0], subdomain_id=parameters[f"line_sub_mesh_{0}_r_id"])), \
    ('t', Measure("ds", domain=lmsh.sub_meshes[0], subdomain_data=mf_sub_mesh[0], subdomain_id=parameters[f"line_sub_mesh_{0}_t_id"])), \
    ('b', Measure("ds", domain=lmsh.sub_meshes[0], subdomain_data=mf_sub_mesh[0], subdomain_id=parameters[f"line_sub_mesh_{0}_b_id"])) \
    ])

ds_sub_mesh[0]['lr'] = ds_sub_mesh[0]['l'] + ds_sub_mesh[0]['r']
ds_sub_mesh[0]['tb'] = ds_sub_mesh[0]['t'] + ds_sub_mesh[0]['b']

ds_sub_mesh[0]['lrtb'] = ds_sub_mesh[0]['lr'] + ds_sub_mesh[0]['tb']

ds_sub_mesh[1] = dict([ \
    ('in_l', Measure("ds", domain=lmsh.sub_meshes[1], subdomain_data=mf_sub_mesh[1], subdomain_id=parameters[f"line_sub_mesh_{0}_l_id"])), \
    ('in_r', Measure("ds", domain=lmsh.sub_meshes[1], subdomain_data=mf_sub_mesh[1], subdomain_id=parameters[f"line_sub_mesh_{0}_r_id"])), \
    ('in_t', Measure("ds", domain=lmsh.sub_meshes[1], subdomain_data=mf_sub_mesh[1], subdomain_id=parameters[f"line_sub_mesh_{0}_t_id"])), \
    ('in_b', Measure("ds", domain=lmsh.sub_meshes[1], subdomain_data=mf_sub_mesh[1], subdomain_id=parameters[f"line_sub_mesh_{0}_b_id"])), \

    ('out_l', Measure("ds", domain=lmsh.sub_meshes[1], subdomain_data=mf_sub_mesh[1], subdomain_id=parameters[f"line_sub_mesh_{1}_l_id"])), \
    ('out_r', Measure("ds", domain=lmsh.sub_meshes[1], subdomain_data=mf_sub_mesh[1], subdomain_id=parameters[f"line_sub_mesh_{1}_r_id"])), \
    ('out_t', Measure("ds", domain=lmsh.sub_meshes[1], subdomain_data=mf_sub_mesh[1], subdomain_id=parameters[f"line_sub_mesh_{1}_t_id"])), \
    ('out_b', Measure("ds", domain=lmsh.sub_meshes[1], subdomain_data=mf_sub_mesh[1], subdomain_id=parameters[f"line_sub_mesh_{1}_b_id"])), \
    ])

ds_sub_mesh[1]['in_lr'] = ds_sub_mesh[1]['in_l'] + ds_sub_mesh[1]['in_r']
ds_sub_mesh[1]['in_tb'] = ds_sub_mesh[1]['in_t'] + ds_sub_mesh[1]['in_b']

ds_sub_mesh[1]['in_lrtb'] = ds_sub_mesh[1]['in_lr'] + ds_sub_mesh[1]['in_tb']


ds_sub_mesh[1]['out_lr'] = ds_sub_mesh[1]['out_l'] + ds_sub_mesh[1]['out_r']
ds_sub_mesh[1]['out_tb'] = ds_sub_mesh[1]['out_t'] + ds_sub_mesh[1]['out_b']

ds_sub_mesh[1]['out_lrtb'] = ds_sub_mesh[1]['out_lr'] + ds_sub_mesh[1]['out_tb']



import importlib
check_mesh_module = importlib.import_module('mesh.check_tags.square_square')

print(f'Module {__file__} called {check_mesh_module.__file__}', flush=True)

#Define boundaries
boundary = [''] * len(lmsh.sub_meshes)

boundary[0] = dict([])
boundary[1] = dict([])



boundary[1]['out_l'] = f'near(x[0], {0})'
boundary[1]['out_r'] = f'near(x[0], {parameters["L"]})'
boundary[1]['out_t'] = f'near(x[1], {parameters["h"]})'
boundary[1]['out_b'] = f'near(x[1], {0})'

boundary[1]['out_lr'] = f"({boundary[1]['out_l']}) || ({boundary[1]['out_r']})"
boundary[1]['out_tb'] = f"({boundary[1]['out_t']}) || ({boundary[1]['out_b']})"
boundary[1]['out_lrtb'] = f"({boundary[1]['out_lr']}) || ({boundary[1]['out_tb']})"


boundary[0]['l'] = f"on_boundary && near(x[0], {parameters['p'][0]}) && !{boundary[1]['out_t']} && !{boundary[1]['out_b']}"
boundary[0]['r'] = f"on_boundary && near(x[0], {parameters['p'][0] + parameters['L_in']}) && !{boundary[1]['out_t']} && !{boundary[1]['out_b']}"
boundary[0]['t'] = f"on_boundary && near(x[1], {parameters['p'][1] + parameters['h_in']}) && !{boundary[1]['out_l']} && !{boundary[1]['out_r']}"
boundary[0]['b'] = f"on_boundary && near(x[1], {parameters['p'][1]}) && !{boundary[1]['out_l']} && !{boundary[1]['out_r']}"

boundary[0]['lr'] = f"({boundary[0]['l']}) || ({boundary[0]['r']})"
boundary[0]['tb'] = f"({boundary[0]['t']}) || ({boundary[0]['b']})"
boundary[0]['lrtb'] = f"({boundary[0]['lr']}) || ({boundary[0]['tb']})"

boundary[1]['in_l'] = boundary[0]['l']
boundary[1]['in_r'] = boundary[0]['r']
boundary[1]['in_t'] = boundary[0]['t']
boundary[1]['in_b'] = boundary[0]['b']

boundary[1]['in_lr'] = f"({boundary[1]['in_l']}) || ({boundary[1]['in_r']})"
boundary[1]['in_tb'] = f"({boundary[1]['in_t']}) || ({boundary[1]['in_b']})"
boundary[1]['in_lrtb'] = f"({boundary[1]['in_lr']}) || ({boundary[1]['in_tb']})"



