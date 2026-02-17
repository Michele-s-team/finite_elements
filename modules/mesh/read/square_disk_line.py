'''
Notation:

* Map functions:
    - sf: a list of map functions, where sf[i] is the map function for the triangles of the i-th mesh
    - mf: a list of map functions, where mf[i] is the map function for the lines of the i-th mesh

* Measures: 
    - dx_mesh[i] is the volume measure of the i-th mesh, and it includes all sub-meshes of the i-th mesh
    - dx_sub_mesh[i][j] is the volume measure of the j-th submesh of the i-th mesh. If the i-th mesh has no sub-meshes, then dx_sub_mesh[i] is empty. 
'''

from fenics import *
import os

import input_output as io
import mesh.load as lmsh
import mesh.utils as msh
import runtime_arguments as rarg

# read quantities for meshes

sf = [None] * lmsh.parameters['n_meshes']
mf = [None] * lmsh.parameters['n_meshes']
r_mesh = [None] * lmsh.parameters['n_meshes']

# read quantities for mesh[0]
# read the triangles
sf[0] = msh.read_mesh_components(lmsh.mesh[0], (lmsh.mesh[0]).topology().dim(), os.path.join(rarg.args.input_directory, f'mesh_{0}', 'triangle_mesh.xdmf'))
# read the lines
mf[0] = msh.read_mesh_components(lmsh.mesh[0], (lmsh.mesh[0]).topology().dim() - 1, os.path.join(rarg.args.input_directory, f'mesh_{0}', 'line_mesh.xdmf'))



# read quantities for mesh[1]
# read the lines
sf[1] = msh.read_mesh_components(lmsh.mesh[1], (lmsh.mesh[1]).topology().dim(), os.path.join(rarg.args.input_directory, f'mesh_{1}', "line_mesh.h5"), 
                                 name_to_read="cf")
# read the vertices
mf[1] = msh.read_mesh_components(lmsh.mesh[1], (lmsh.mesh[1]).topology().dim() - 1, os.path.join(rarg.args.input_directory, f'mesh_{1}', "vertex_mesh.h5"), 
                                 name_to_read="vf")
                                 

r_mesh[0] = lmsh.mesh[0].hmin()
r_mesh[1] = lmsh.mesh[1].hmin()




print(f'lmsh_sub_meshes: {lmsh.sub_meshes}')
print(f'sf_sub_meshes: {lmsh.sf_sub_meshes}')


#1.  define surface and line elements for meshes
dx_mesh = [[] for _ in range(lmsh.parameters['n_meshes'])]

dx_mesh[0] = Measure("dx", domain=lmsh.mesh[0], subdomain_data=lmsh.sf[0])
dx_mesh[1] = Measure("dx", domain=lmsh.mesh[1], subdomain_data=lmsh.sf[1])


#2. define surface and line elements for sub-meshes

dx_sub_mesh = [[] for _ in range(lmsh.parameters['n_meshes'])]

for p in range(len(lmsh.sub_meshes[0])):
    dx_sub_mesh[0].append(Measure("dx", domain=lmsh.sub_meshes[0][p], subdomain_data=lmsh.sf_sub_meshes[0][p], subdomain_id=lmsh.mesh_parameters[0][f"sub_mesh_{p}_id"]))


'''





# line elements
ds_sub_mesh = [''] * len(lmsh.sub_meshes)

ds_sub_mesh[0] = dict([ \
    ('ds_circle', Measure("ds", domain=lmsh.sub_meshes[0], subdomain_data=lmsh.mf_sub_meshes[0], subdomain_id=parameters[f"circle_loop_id"])), 
    ])

ds_sub_mesh[1] = dict([ \
    ('ds_l', Measure("ds", domain=lmsh.sub_meshes[1], subdomain_data=lmsh.mf_sub_meshes[1], subdomain_id=parameters[f"line_sub_mesh_{1}_l_id"])), \
    ('ds_r', Measure("ds", domain=lmsh.sub_meshes[1], subdomain_data=lmsh.mf_sub_meshes[1], subdomain_id=parameters[f"line_sub_mesh_{1}_r_id"])), \
    ('ds_t', Measure("ds", domain=lmsh.sub_meshes[1], subdomain_data=lmsh.mf_sub_meshes[1], subdomain_id=parameters[f"line_sub_mesh_{1}_t_id"])), \
    ('ds_b', Measure("ds", domain=lmsh.sub_meshes[1], subdomain_data=lmsh.mf_sub_meshes[1], subdomain_id=parameters[f"line_sub_mesh_{1}_b_id"])), \
    ('ds_circle', Measure("ds", domain=lmsh.sub_meshes[1], subdomain_data=lmsh.mf_sub_meshes[1], subdomain_id=parameters[f"circle_loop_id"])) \
    ])
ds_sub_mesh[1]['ds_lr'] = ds_sub_mesh[1]['ds_l'] + ds_sub_mesh[1]['ds_r']
ds_sub_mesh[1]['ds_tb'] = ds_sub_mesh[1]['ds_t'] + ds_sub_mesh[1]['ds_b']
ds_sub_mesh[1]['ds_lrtb'] = ds_sub_mesh[1]['ds_lr'] + ds_sub_mesh[1]['ds_tb']
ds_sub_mesh[1]['ds'] = ds_sub_mesh[1]['ds_lrtb'] + ds_sub_mesh[1]['ds_circle']
'''
import importlib
check_mesh_module = importlib.import_module('mesh.check_tags.square_disk_line')

print(f'Module {__file__} called {check_mesh_module.__file__}', flush=True)

