'''
This code reads the 3d mesh generated from generate_mesh.py and it creates dvs and dss from labelled components of the mesh
'''

from fenics import *
import importlib
import os
import sys

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import input_output as io
import mesh.load as lmsh
import mesh.utils as msh
import runtime_arguments as rarg


# read the tetrahedra
cf = msh.read_mesh_components(lmsh.mesh, lmsh.mesh.topology().dim(), os.path.join(rarg.args.input_directory, 'tetra_mesh.xdmf'))
# read the triangles
sf = msh.read_mesh_components(lmsh.mesh, lmsh.mesh.topology().dim() - 1, os.path.join(rarg.args.input_directory, 'triangle_mesh.xdmf'))

# radius of the smallest cell in the mesh
r_mesh = lmsh.mesh.hmin()

parameters = io.read_parameters_from_csv_file(os.path.join(rarg.args.input_directory, 'mesh_metadata.csv'))


dx_surface = Measure("dx", domain=lmsh.mesh, subdomain_data=cf, subdomain_id=lmsh.parameters['surface_volume_id'])  
dx_box = Measure("dx", domain=lmsh.mesh, subdomain_data=cf, subdomain_id=lmsh.parameters['box_volume_id'])  

dx = dx_box + dx_surface

ds_le = Measure("ds", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=lmsh.parameters['boundary_le_id'])
ds_ri = Measure("ds", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=lmsh.parameters['boundary_ri_id'])
ds_to = Measure("ds", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=lmsh.parameters['boundary_to_id'])
ds_bo = Measure("ds", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=lmsh.parameters['boundary_bo_id'])
ds_fr = Measure("ds", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=lmsh.parameters['boundary_fr_id'])
ds_ba = Measure("ds", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=lmsh.parameters['boundary_ba_id'])

ds_leri = ds_le + ds_ri
ds_tobo = ds_to + ds_bo
ds_frba = ds_fr + ds_ba

ds = ds_leri + ds_tobo + ds_frba

check_mesh_module = importlib.import_module('mesh.check_tags.box_surface')
print(f'Module {__file__} called {check_mesh_module.__file__}', flush=True)
