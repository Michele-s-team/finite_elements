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
import importlib
import os

import input_output as io
import mesh.load as lmsh
import mesh.utils as msh
import runtime_arguments as rarg


# read parameters for the mesh ensemble

parameters = io.read_parameters_from_csv_file(os.path.join(rarg.args.input_directory, "mesh_metadata.csv"))


# read quantities for meshes

sf = [None] * lmsh.parameters['n_meshes']
mf = [None] * lmsh.parameters['n_meshes']
mf_I = [None] * lmsh.parameters['n_meshes']
r_mesh = [None] * lmsh.parameters['n_meshes']

#1. read quantities for mesh[0]

# 1.1 read the triangles
sf[0] = msh.read_mesh_components(lmsh.mesh[0], (lmsh.mesh[0]).topology().dim(), os.path.join(rarg.args.input_directory, f'mesh_{0}', 'triangle_mesh.xdmf'))


# 1.2. read the lines

# 1.2.1 read the boundary lines
mf[0] = msh.read_mesh_components(lmsh.mesh[0], (lmsh.mesh[0]).topology().dim() - 1, os.path.join(rarg.args.input_directory, f'mesh_{0}', 'line_mesh.xdmf'))

# 1.2.2 read the inner (I) lines

mf_I[0] = msh.read_mesh_internal_components(lmsh.mesh[0], sf[0], lmsh.parameters['sub_mesh_0_0_id'], lmsh.parameters['sub_mesh_0_1_id'], lmsh.parameters['shape_id'])

    
    

# 2. read quantities for mesh[1]

# 2.1 read the lines
sf[1] = msh.read_mesh_components(lmsh.mesh[1], (lmsh.mesh[1]).topology().dim(), os.path.join(rarg.args.input_directory, f'mesh_{1}', "line_mesh.h5"), name_to_read="cf")

# 2.2 read the vertices
mf[1] = msh.read_mesh_components(lmsh.mesh[1], (lmsh.mesh[1]).topology().dim() - 1, os.path.join(rarg.args.input_directory, f'mesh_{1}', "vertex_mesh.h5"), name_to_read="vf")
                                 

# minimal mesh size for meshes
r_mesh[0] = lmsh.mesh[0].hmin()
r_mesh[1] = lmsh.mesh[1].hmin()

# minimal mesh size for sub_meshes
r_sub_mesh = [[None] * 2, None]
r_sub_mesh[0][0] = lmsh.sub_meshes[0][0].hmin()
r_sub_mesh[0][1] = lmsh.sub_meshes[0][1].hmin()


# 3. define measures

#3.1.  define bulk and boundary measures for meshes
dx_mesh = [[] for _ in range(lmsh.parameters['n_meshes'])]
ds_mesh = [None] * lmsh.parameters['n_meshes']

# 3.1.1 mesh 0
# 3.1.1.1 bulk measures
dx_mesh[0] = dict([
    ('dx', Measure("dx", domain=lmsh.mesh[0], subdomain_data=lmsh.sf[0])),\
    ('dx_shape', Measure("dx", domain=lmsh.mesh[0], subdomain_data=lmsh.sf[0], subdomain_id=lmsh.parameters[f"sub_mesh_{0}_{0}_id"])),\
    ('dx_square', Measure("dx", domain=lmsh.mesh[0], subdomain_data=lmsh.sf[0], subdomain_id=lmsh.parameters[f"sub_mesh_{0}_{1}_id"]))
])

# 3.1.1.2 boundary measures
ds_mesh[0] = dict([ \
    ('ds_l', Measure("ds", domain=lmsh.mesh[0], subdomain_data=mf[0], subdomain_id=lmsh.parameters[f"line_l_id"])), \
    ('ds_r', Measure("ds", domain=lmsh.mesh[0], subdomain_data=mf[0], subdomain_id=lmsh.parameters[f"line_r_id"])), \
    ('ds_t', Measure("ds", domain=lmsh.mesh[0], subdomain_data=mf[0], subdomain_id=lmsh.parameters[f"line_t_id"])), \
    ('ds_b', Measure("ds", domain=lmsh.mesh[0], subdomain_data=mf[0], subdomain_id=lmsh.parameters[f"line_b_id"])), \
    ('dS_shape', Measure("dS", domain=lmsh.mesh[0], subdomain_data=mf_I[0], subdomain_id=lmsh.parameters[f"shape_id"])), \
    ('dS_I_shape', Measure("dS", domain=lmsh.mesh[0], subdomain_data=mf_I[0], subdomain_id=lmsh.parameters[f"sub_mesh_0_0_id"])),\
    ('dS_I_square', Measure("dS", domain=lmsh.mesh[0], subdomain_data=mf_I[0], subdomain_id=lmsh.parameters[f"sub_mesh_0_1_id"]))
    ])

ds_mesh[0]['ds_lr'] = ds_mesh[0]['ds_l'] + ds_mesh[0]['ds_r']
ds_mesh[0]['ds_tb'] = ds_mesh[0]['ds_t'] + ds_mesh[0]['ds_b']
ds_mesh[0]['ds'] = ds_mesh[0]['ds_lr'] + ds_mesh[0]['ds_tb']

# all internal facets in the region inside the square (including the shape and the facets at the shape boundary)
ds_mesh[0]['dS_I'] = ds_mesh[0]['dS_shape'] + ds_mesh[0]['dS_I_shape'] + ds_mesh[0]['dS_I_square']

# 3.1.2 mesh 1
# 3.1.2.1 bulk measures
dx_mesh[1] = Measure("dx", domain=lmsh.mesh[1], subdomain_data=lmsh.sf[1])

# 3.1.2.2 boundary measures
ds_mesh[1] = dict([ \
    ('ds_l', Measure("ds", domain=lmsh.mesh[1], subdomain_data=mf[1], subdomain_id=lmsh.mesh_parameters[1][f"vertex_l_id"])), \
    ('ds_r', Measure("ds", domain=lmsh.mesh[1], subdomain_data=mf[1], subdomain_id=lmsh.mesh_parameters[1][f"vertex_r_id"]))
    ])

ds_mesh[1]['ds'] = ds_mesh[1]['ds_l'] + ds_mesh[1]['ds_r']


#4. define bulk and boundary measures for sub-meshes
dx_sub_mesh = [[] for _ in range(lmsh.parameters['n_meshes'])]
ds_sub_mesh = [[None, None], None]

# 4.1 sub_meshes of mesh 0
# 4.1.1 bulk measures
for p in range(len(lmsh.sub_meshes[0])):
    dx_sub_mesh[0].append(Measure("dx", domain=lmsh.sub_meshes[0][p], subdomain_data=lmsh.sf_sub_meshes[0][p], subdomain_id=lmsh.mesh_parameters[0][f"sub_mesh_{p}_id"]))

# 4.1.2 boundary measures
# 4.1.2.1 boundary measures of sub_mesh 0 of mesh 0
ds_sub_mesh[0][0] = dict([
        ('ds', Measure("ds", domain=lmsh.sub_meshes[0][0], subdomain_data=lmsh.mf_sub_meshes[0][0], subdomain_id=lmsh.mesh_parameters[0][f"shape_id"]))
])

# 4.1.2.1 boundary measures of sub_mesh 1 of mesh 0
ds_sub_mesh[0][1] = dict([
        ('ds_l', Measure("ds", domain=lmsh.sub_meshes[0][1], subdomain_data=lmsh.mf_sub_meshes[0][1], subdomain_id=lmsh.mesh_parameters[0]["line_l_id"])),\
        ('ds_r', Measure("ds", domain=lmsh.sub_meshes[0][1], subdomain_data=lmsh.mf_sub_meshes[0][1], subdomain_id=lmsh.mesh_parameters[0]["line_r_id"])),\
        ('ds_t', Measure("ds", domain=lmsh.sub_meshes[0][1], subdomain_data=lmsh.mf_sub_meshes[0][1], subdomain_id=lmsh.mesh_parameters[0]["line_t_id"])),\
        ('ds_b', Measure("ds", domain=lmsh.sub_meshes[0][1], subdomain_data=lmsh.mf_sub_meshes[0][1], subdomain_id=lmsh.mesh_parameters[0]["line_b_id"])),\
        ('ds_shape', Measure("ds", domain=lmsh.sub_meshes[0][1], subdomain_data=lmsh.mf_sub_meshes[0][1], subdomain_id=lmsh.mesh_parameters[0]["shape_id"]))
])

ds_sub_mesh[0][1]['ds_lr'] = ds_sub_mesh[0][1]['ds_l'] + ds_sub_mesh[0][1]['ds_r']
ds_sub_mesh[0][1]['ds_tb'] = ds_sub_mesh[0][1]['ds_t'] + ds_sub_mesh[0][1]['ds_b']

ds_sub_mesh[0][1]['ds_lrtb'] = ds_sub_mesh[0][1]['ds_lr'] + ds_sub_mesh[0][1]['ds_tb']


ds_sub_mesh[0][1]['ds'] = ds_sub_mesh[0][1]['ds_lrtb'] + ds_sub_mesh[0][1]['ds_shape']


check_mesh_module = importlib.import_module('mesh.check_tags.square_shape_line')

print(f'Module {__file__} called {check_mesh_module.__file__}', flush=True)