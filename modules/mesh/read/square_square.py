from fenics import *
import importlib

import input_output as io
import mesh.load as lmsh
import mesh.utils as msh
import runtime_arguments as rarg

# read the triangles
sf = msh.read_mesh_components(lmsh.mesh, lmsh.mesh.topology().dim(), rarg.args.input_directory + "/triangle_mesh.xdmf")
# read the lines
mf = msh.read_mesh_components(lmsh.mesh, lmsh.mesh.topology().dim() - 1, rarg.args.input_directory + "/line_mesh.xdmf")

parameters = io.read_parameters_from_csv_file(rarg.args.input_directory + "/mesh_metadata.csv")

# radius of the smallest cell in the mesh
r_mesh = lmsh.mesh.hmin()


# create line and surface elements for sub_meshes
dx_sub_mesh = []

for p in range(len(lmsh.sub_meshes)):
    dx_sub_mesh.append(Measure("dx", domain=lmsh.sub_meshes[p], subdomain_data=lmsh.sf_sub_meshes[p], subdomain_id=parameters[f"sub_mesh_{p}_id"]))

ds_sub_mesh = [''] * len(lmsh.sub_meshes)

ds_sub_mesh[0] = dict([ \
    ('l', Measure("ds", domain=lmsh.sub_meshes[0], subdomain_data=lmsh.mf_sub_meshes[0], subdomain_id=parameters[f"line_sub_mesh_{0}_l_id"])), \
    ('r', Measure("ds", domain=lmsh.sub_meshes[0], subdomain_data=lmsh.mf_sub_meshes[0], subdomain_id=parameters[f"line_sub_mesh_{0}_r_id"])), \
    ('t', Measure("ds", domain=lmsh.sub_meshes[0], subdomain_data=lmsh.mf_sub_meshes[0], subdomain_id=parameters[f"line_sub_mesh_{0}_t_id"])), \
    ('b', Measure("ds", domain=lmsh.sub_meshes[0], subdomain_data=lmsh.mf_sub_meshes[0], subdomain_id=parameters[f"line_sub_mesh_{0}_b_id"])) \
    ])

ds_sub_mesh[0]['lr'] = ds_sub_mesh[0]['l'] + ds_sub_mesh[0]['r']
ds_sub_mesh[0]['tb'] = ds_sub_mesh[0]['t'] + ds_sub_mesh[0]['b']

ds_sub_mesh[0]['lrtb'] = ds_sub_mesh[0]['lr'] + ds_sub_mesh[0]['tb']

ds_sub_mesh[1] = dict([ \
    ('in_l', Measure("ds", domain=lmsh.sub_meshes[1], subdomain_data=lmsh.mf_sub_meshes[1], subdomain_id=parameters[f"line_sub_mesh_{0}_l_id"])), \
    ('in_r', Measure("ds", domain=lmsh.sub_meshes[1], subdomain_data=lmsh.mf_sub_meshes[1], subdomain_id=parameters[f"line_sub_mesh_{0}_r_id"])), \
    ('in_t', Measure("ds", domain=lmsh.sub_meshes[1], subdomain_data=lmsh.mf_sub_meshes[1], subdomain_id=parameters[f"line_sub_mesh_{0}_t_id"])), \
    ('in_b', Measure("ds", domain=lmsh.sub_meshes[1], subdomain_data=lmsh.mf_sub_meshes[1], subdomain_id=parameters[f"line_sub_mesh_{0}_b_id"])), \

    ('out_l', Measure("ds", domain=lmsh.sub_meshes[1], subdomain_data=lmsh.mf_sub_meshes[1], subdomain_id=parameters[f"line_sub_mesh_{1}_l_id"])), \
    ('out_r', Measure("ds", domain=lmsh.sub_meshes[1], subdomain_data=lmsh.mf_sub_meshes[1], subdomain_id=parameters[f"line_sub_mesh_{1}_r_id"])), \
    ('out_t', Measure("ds", domain=lmsh.sub_meshes[1], subdomain_data=lmsh.mf_sub_meshes[1], subdomain_id=parameters[f"line_sub_mesh_{1}_t_id"])), \
    ('out_b', Measure("ds", domain=lmsh.sub_meshes[1], subdomain_data=lmsh.mf_sub_meshes[1], subdomain_id=parameters[f"line_sub_mesh_{1}_b_id"])), \
    ])

ds_sub_mesh[1]['in_lr'] = ds_sub_mesh[1]['in_l'] + ds_sub_mesh[1]['in_r']
ds_sub_mesh[1]['in_tb'] = ds_sub_mesh[1]['in_t'] + ds_sub_mesh[1]['in_b']

ds_sub_mesh[1]['in_lrtb'] = ds_sub_mesh[1]['in_lr'] + ds_sub_mesh[1]['in_tb']


ds_sub_mesh[1]['out_lr'] = ds_sub_mesh[1]['out_l'] + ds_sub_mesh[1]['out_r']
ds_sub_mesh[1]['out_tb'] = ds_sub_mesh[1]['out_t'] + ds_sub_mesh[1]['out_b']

ds_sub_mesh[1]['out_lrtb'] = ds_sub_mesh[1]['out_lr'] + ds_sub_mesh[1]['out_tb']


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



